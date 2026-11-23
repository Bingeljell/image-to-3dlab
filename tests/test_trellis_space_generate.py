"""Pure contract tests for the clean-port full-generation wrapper.

These import the real module (no re-derived copies) and exercise only the torch-free helpers,
so they run in the dev venv as the cheap gate before any 25-minute run. The heavy ``generate``
path is exercised manually via ``--check`` then a real run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from image_to_3dlab.provenance import read_model_record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_space_generate.py"
SPEC = importlib.util.spec_from_file_location("trellis_space_generate", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


# --- resolution -> pipeline_type ---
def test_pipeline_type_mapping():
    assert gen.pipeline_type_for_resolution("512") == "512"
    assert gen.pipeline_type_for_resolution("1024") == "1024_cascade"
    assert gen.pipeline_type_for_resolution("1536") == "1536_cascade"


def test_pipeline_type_rejects_unknown():
    with pytest.raises(ValueError):
        gen.pipeline_type_for_resolution("2048")


def test_demo_resolution_is_1024_cascade():
    assert gen.pipeline_type_for_resolution(gen.DEMO_PARAMS["resolution"]) == "1024_cascade"


# --- sampler params come straight from the demo defaults, as fresh copies ---
@pytest.mark.parametrize(
    "stage,expected",
    [
        ("sparse_structure", {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}),
        ("shape_slat", {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}),
        ("tex_slat", {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}),
    ],
)
def test_sampler_params_match_demo(stage, expected):
    assert gen.sampler_params(stage) == expected


def test_sampler_params_returns_a_copy():
    p = gen.sampler_params("shape_slat")
    p["steps"] = 999
    assert gen.DEMO_PARAMS["shape_slat"]["steps"] == 12  # source of truth untouched


def test_sampler_params_rejects_unknown_stage():
    with pytest.raises(ValueError):
        gen.sampler_params("nonsense")


# --- alpha guardrail logic ---
@pytest.mark.parametrize(
    "mode,alpha_min,expected",
    [
        ("RGBA", 0, True),
        ("RGBA", 254, True),
        ("RGBA", 255, False),   # fully opaque = no real foreground mask
        ("RGB", None, False),
        ("LA", 0, False),       # not RGBA
    ],
)
def test_alpha_is_transparent(mode, alpha_min, expected):
    assert gen.alpha_is_transparent(mode, alpha_min) is expected


# --- "has alpha" is not "is cut out" ---
# 2026-09-03: 2dDog.png passed alpha_is_transparent on its letterbox bars alone while its
# subject sat on an opaque white backdrop. TRELLIS duly rebuilt the backdrop as geometry --
# 45 minutes for a slab. The border ring is what separates the two cases.
def _letterboxed(size=64, bar=8):
    """Opaque artwork with only transparent bars top and bottom -- the 2dDog.png shape."""
    import numpy as np

    a = np.full((size, size), 255, dtype=np.uint8)
    a[:bar] = 0
    a[-bar:] = 0
    return a


def _real_cutout(size=64, margin=8):
    """An opaque subject with a transparent frame all the way round."""
    import numpy as np

    a = np.zeros((size, size), dtype=np.uint8)
    a[margin:-margin, margin:-margin] = 255
    return a


def test_border_opaque_fraction_flags_letterboxed_image():
    # left/right edges of the un-barred rows are still opaque, so the ring is far from clear
    assert gen.border_opaque_fraction(_letterboxed()) > gen.BORDER_OPAQUE_LIMIT


def test_border_opaque_fraction_passes_real_cutout():
    assert gen.border_opaque_fraction(_real_cutout()) == 0.0


def test_border_opaque_fraction_ignores_interior_opacity():
    """A subject filling most of the frame is fine so long as the frame itself is clear."""
    import numpy as np

    a = np.full((64, 64), 255, dtype=np.uint8)
    a[:3] = a[-3:] = 0
    a[:, :3] = a[:, -3:] = 0
    assert a.mean() > 200  # overwhelmingly opaque overall...
    assert gen.border_opaque_fraction(a) == 0.0  # ...but properly framed


def test_border_opaque_fraction_survives_degenerate_input():
    import numpy as np

    assert gen.border_opaque_fraction(np.zeros((3, 3), dtype=np.uint8)) == 0.0
    assert gen.border_opaque_fraction(np.zeros((8, 8, 3), dtype=np.uint8)) == 0.0


def test_uncut_foreground_message_is_actionable():
    msg = gen.uncut_foreground_message("dog.png", 0.39)
    assert "dog.png" in msg
    assert "39%" in msg           # the measurement, so the user can judge it
    assert "--allow-uncut" in msg  # the override, so it is not a dead end


# --- degenerate-face filter (the MPS decode -1 index crash) ---
def test_valid_face_mask_drops_out_of_range():
    faces = [[0, 1, 2], [59990, 59991, -1], [3, 4, 5], [1, 2, 10]]
    # V=10 -> index 10 is out of range, and -1 is the degenerate marker
    mask = gen.valid_face_mask(faces, num_vertices=10)
    assert list(mask) == [True, False, True, False]


def test_filter_degenerate_faces_removes_bad_rows_and_keeps_device():
    """The filter must drop out-of-range faces without relying on MPS mask indexing.

    Indexing a large boolean mask on Metal returned a garbage index on 2026-09-20 and
    killed a run after sampling had succeeded, so the gather now happens on CPU.
    """
    torch = pytest.importorskip("torch")

    class _Mesh:
        pass

    mesh = _Mesh()
    mesh.vertices = torch.zeros((5, 3))
    mesh.faces = torch.tensor([[0, 1, 2], [1, 2, 3], [-1, 0, 1], [0, 1, 99]])

    removed = gen.filter_degenerate_faces(mesh)

    assert removed == 2
    assert mesh.faces.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert mesh.faces.device == torch.device("cpu")


def test_filter_degenerate_faces_is_a_noop_when_every_face_is_valid():
    torch = pytest.importorskip("torch")

    class _Mesh:
        pass

    mesh = _Mesh()
    mesh.vertices = torch.zeros((4, 3))
    original = torch.tensor([[0, 1, 2], [1, 2, 3]])
    mesh.faces = original

    assert gen.filter_degenerate_faces(mesh) == 0
    # Untouched, not merely equal: no needless copy when there is nothing to drop.
    assert mesh.faces is original


def test_valid_face_mask_boundary_index_is_valid():
    # index num_vertices-1 is the last valid vertex
    assert list(gen.valid_face_mask([[0, 9, 9]], num_vertices=10)) == [True]
    # all-good faces stay
    assert gen.valid_face_mask([[0, 1, 2], [2, 3, 4]], num_vertices=5).all()


# --- sampling checkpoint boundary ----------------------------------------------------
def test_sample_without_decode_returns_latents_and_restores_decoder():
    class FakePipeline:
        def __init__(self):
            self.decode_calls = 0

        def decode_latent(self, shape, texture, resolution):
            self.decode_calls += 1
            return [f"mesh-{shape}-{texture}-{resolution}"]

        def run(self, image, **kwargs):
            assert image == "image"
            assert kwargs["return_latent"] is True
            output = self.decode_latent("shape", "texture", 512)
            return output, ("shape", "texture", 512)

    pipeline = FakePipeline()
    latents = gen.sample_latents_without_decode(pipeline, "image", seed=7)

    assert latents == ("shape", "texture", 512)
    assert pipeline.decode_calls == 0
    assert pipeline.decode_latent("shape", "texture", 512) == ["mesh-shape-texture-512"]
    assert pipeline.decode_calls == 1


def test_sample_without_decode_restores_decoder_after_sampling_error():
    class FakePipeline:
        def decode_latent(self, *_args):
            return ["mesh"]

        def run(self, *_args, **_kwargs):
            raise RuntimeError("sampling broke")

    pipeline = FakePipeline()
    original = pipeline.decode_latent
    with pytest.raises(RuntimeError, match="sampling broke"):
        gen.sample_latents_without_decode(pipeline, "image")

    assert pipeline.decode_latent == original


def test_generate_checkpoints_latents_before_decode_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    image_path = tmp_path / "cutout.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y in range(3, 13):
        for x in range(3, 13):
            image.putpixel((x, y), (80, 120, 40, 255))
    image.save(image_path)
    output_path = tmp_path / "model.glb"

    class FakePipeline:
        def preprocess_image(self, source):
            return source

    shape = SimpleNamespace(
        feats=torch.tensor([[1.0, 2.0]]),
        coords=torch.tensor([[0, 1, 2, 3]]),
    )
    texture = SimpleNamespace(feats=torch.tensor([[4.0, 5.0]]))
    monkeypatch.setattr(gen, "load_pipeline", lambda *_args: FakePipeline())
    monkeypatch.setattr(
        gen,
        "sample_latents_without_decode",
        lambda *_args, **_kwargs: (shape, texture, 512),
    )

    checkpoint = tmp_path / "model_latents.pt"

    def fail_decode(*_args, **_kwargs):
        assert checkpoint.is_file(), "recovery checkpoint must exist before decode starts"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["seed"] == 19
        raise RuntimeError("decode failed")

    monkeypatch.setattr(gen, "_decode_and_cache", fail_decode)

    with pytest.raises(RuntimeError, match="decode failed"):
        gen.generate(
            image_path,
            output_path,
            tmp_path,
            seed=19,
            sparse_attn_backend="sdpa",
            matted_with=None,
            save_latents=False,
            save_decode=False,
            pre_cap=4_000_000,
            resolution="512",
            decimation_target=300_000,
            texture_size=1024,
        )

    assert checkpoint.is_file(), "failed non-debug runs must retain their recovery checkpoint"


def _finished_run(monkeypatch, tmp_path, *, matted_with=None, save_latents=False, told_to_decode=None):
    """The real `generate`, picture to files, with the model, the decode and the bake stubbed."""
    from types import SimpleNamespace

    import torch
    from PIL import Image

    image_path = tmp_path / "cutout.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y in range(3, 13):
        for x in range(3, 13):
            image.putpixel((x, y), (80, 120, 40, 255))
    image.save(image_path)
    output_path = tmp_path / "model.glb"

    class FakePipeline:
        def preprocess_image(self, source):
            return source

    shape = SimpleNamespace(
        feats=torch.tensor([[1.0, 2.0]]),
        coords=torch.tensor([[0, 1, 2, 3]]),
    )
    texture = SimpleNamespace(feats=torch.tensor([[4.0, 5.0]]))
    monkeypatch.setattr(gen, "load_pipeline", lambda *_args: FakePipeline())
    monkeypatch.setattr(
        gen,
        "sample_latents_without_decode",
        lambda *_args, **_kwargs: (shape, texture, 512),
    )
    def decode(*_args, **told):
        if told_to_decode is not None:
            told_to_decode.update(told)
        return {"vertices": None, "faces": None, "attrs": None, "coords": None, "attr_layout": None, "res": 512}, 0.1

    monkeypatch.setattr(gen, "_decode_and_cache", decode)
    monkeypatch.setattr(gen, "_release_mps_memory", lambda: None)

    def finish_bake(*_args, **_kwargs):
        output_path.write_bytes(b"glTF")
        return {"bake": 0.2}, {"glb": gen._artifact(output_path)}

    monkeypatch.setattr(gen, "_bake_export", finish_bake)

    manifest = gen.generate(
        image_path,
        output_path,
        tmp_path,
        seed=19,
        sparse_attn_backend="sdpa",
        matted_with=matted_with,
        save_latents=save_latents,
        save_decode=False,
        pre_cap=4_000_000,
        resolution="512",
        decimation_target=300_000,
        texture_size=1024,
    )
    return image_path, output_path, manifest


def test_generate_removes_temporary_checkpoint_after_success(monkeypatch, tmp_path):
    _, output_path, manifest = _finished_run(monkeypatch, tmp_path)

    assert output_path.is_file()
    assert not (tmp_path / "model_latents.pt").exists()
    assert "latents" not in manifest["artifacts"]


# --- the licence record every way out of the script leaves beside the model ---

def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _licence_parts(record):
    return {part["component"]: part for part in record["components"]}


def _stub_the_bake(monkeypatch, output_path):
    def finish_bake(*_args, **_kwargs):
        output_path.write_bytes(b"glTF")
        return {"bake": 0.2}, {"glb": gen._artifact(output_path)}

    monkeypatch.setattr(gen, "_bake_export", finish_bake)


def test_a_finished_run_leaves_its_licence_record_beside_the_model(monkeypatch, tmp_path):
    # Every other route wrote <name>.provenance.json. This one wrote only its run file, so a Mac
    # TRELLIS.2 model reached the Library with no licence, and Finish called it unverified.
    image_path, output_path, _ = _finished_run(monkeypatch, tmp_path, matted_with="birefnet-general-lite")

    record = read_model_record(output_path)     # what Finish and Fix a spot read
    assert record is not None, "no licence record beside the model"
    assert json.loads((tmp_path / "model.provenance.json").read_text(encoding="utf-8")) == record
    assert record["license"]["name"] == "MIT (TRELLIS.2) + DINOv3 License"
    assert record["model"]["backend"] == "trellis2" and record["model"]["route"] == "apple-mps"
    assert record["model"]["parameters"]["resolution"] == "512"
    assert record["input"] == {"path": str(image_path), "sha256": _sha256(image_path),
                               "source": "user-provided"}
    assert record["output"]["sha256"] == _sha256(output_path)
    parts = _licence_parts(record)
    assert parts["rembg/birefnet-general-lite"]["purpose"] == "background removal"
    assert parts["facebook/dinov3-vitl16-pretrain-lvd1689m"]["license"] == "DINOv3 License"
    assert parts["BRIA RMBG-2.0"]["loaded"] is False
    assert "scripts/patch_trellis_space_core.py" in parts["BRIA RMBG-2.0"]["note"]


def test_a_run_resumed_from_a_decode_bundle_leaves_its_licence_record_too(monkeypatch, tmp_path):
    import torch

    decode_path, output_path = tmp_path / "model_decode.pt", tmp_path / "model.glb"
    # The picture this bundle came from is no longer on disk, so the bundle is the input on record.
    torch.save({"vertices": None, "faces": None, "attrs": None, "coords": None, "attr_layout": None,
                "res": 512, "seed": 7, "images": [str(tmp_path / "gone.png")]}, decode_path)
    monkeypatch.setattr(gen, "configure_environment", lambda *_args: None)
    _stub_the_bake(monkeypatch, output_path)

    gen.generate_from_decode(decode_path, output_path, tmp_path,
                             decimation_target=300_000, texture_size=1024, pre_cap=4_000_000)

    record = read_model_record(output_path)
    assert record is not None, "no licence record beside the model"
    assert record["license"]["name"] == "MIT (TRELLIS.2) + DINOv3 License"
    assert record["input"] == {"path": str(decode_path), "sha256": _sha256(decode_path),
                               "source": "resumed-bundle"}        # not a picture the user gave
    assert record["output"]["sha256"] == _sha256(output_path)
    assert _licence_parts(record)["BRIA RMBG-2.0"]["loaded"] is False


def _resume_from_latents(monkeypatch, tmp_path, kept):
    """The real `generate_from_latents` over a bundle that kept `kept` beside the latents, with the
    model, the decode and the bake stubbed. Returns (the latents file, the model, the decode
    bundle the run would save)."""
    import torch

    latents_path, output_path = tmp_path / "model_latents.pt", tmp_path / "model.glb"
    latents_path.write_bytes(b"latents")

    class OnDevice:     # stands in for a tensor: it is only moved to the device, never read
        def to(self, _device):
            return self

    class SparseTensor:
        def __init__(self, feats, coords):
            pass

        def replace(self, _feats):
            return self

    sparse = types.ModuleType("trellis2.modules.sparse")
    sparse.SparseTensor = SparseTensor
    for name, module in (("trellis2", types.ModuleType("trellis2")),
                         ("trellis2.modules", types.ModuleType("trellis2.modules")),
                         ("trellis2.modules.sparse", sparse)):
        monkeypatch.setitem(sys.modules, name, module)
    bundle = {"shape_slat_feats": OnDevice(), "coords": OnDevice(), "tex_slat_feats": OnDevice(),
              "res": 512, "seed": 7, **kept}
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(gen, "load_pipeline", lambda *_args, **_kwargs: object())
    handed_on = {}

    def decode(*_args, **told):
        handed_on.update(told)
        return {"vertices": None, "faces": None, "attrs": None, "coords": None, "attr_layout": None, "res": 512}, 0.1
    monkeypatch.setattr(gen, "_decode_and_cache", decode)
    monkeypatch.setattr(gen, "_release_mps_memory", lambda: None)
    _stub_the_bake(monkeypatch, output_path)

    gen.generate_from_latents(latents_path, output_path, tmp_path, sparse_attn_backend="sdpa",
                              save_decode=False, pre_cap=4_000_000)
    return latents_path, output_path, handed_on


def _resume_from_decode(monkeypatch, tmp_path, kept):
    import torch

    decode_path, output_path = tmp_path / "model_decode.pt", tmp_path / "model.glb"
    torch.save({"vertices": None, "faces": None, "attrs": None, "coords": None, "attr_layout": None,
                "res": 512, "seed": 7, **kept}, decode_path)
    monkeypatch.setattr(gen, "configure_environment", lambda *_args: None)
    _stub_the_bake(monkeypatch, output_path)

    gen.generate_from_decode(decode_path, output_path, tmp_path,
                             decimation_target=300_000, texture_size=1024, pre_cap=4_000_000)
    return decode_path, output_path


def test_a_run_resumed_from_latents_leaves_its_licence_record_too(monkeypatch, tmp_path):
    picture = tmp_path / "cutout.png"
    picture.write_bytes(b"png")

    _, output_path, _ = _resume_from_latents(monkeypatch, tmp_path, {"images": [str(picture)]})

    record = read_model_record(output_path)
    assert record is not None, "no licence record beside the model"
    assert record["license"]["name"] == "MIT (TRELLIS.2) + DINOv3 License"
    assert record["input"]["path"] == str(picture)      # still on disk, so it is the input on record
    assert record["input"]["sha256"] == _sha256(picture)
    assert record["output"]["sha256"] == _sha256(output_path)


# --- a resumed run says what the first run knew ---------------------------------------------
#
# The remover that cut the picture out and the picture itself are known to the run that
# samples. A run resumed from its latents or its decode bundle wrote a record without the
# remover, and took whatever file sat at the picture's old path as the input.

def test_a_fresh_run_keeps_the_remover_and_the_pictures_hash_in_its_checkpoint(monkeypatch, tmp_path):
    import torch

    kept_by_the_decode = {}
    image_path, _, _ = _finished_run(monkeypatch, tmp_path, matted_with="birefnet-general-lite",
                                     save_latents=True, told_to_decode=kept_by_the_decode)

    latents = torch.load(tmp_path / "model_latents.pt", weights_only=False)
    wanted = {"images": [str(image_path)], "image_sha256": _sha256(image_path), "matte_model": "birefnet-general-lite"}
    assert {key: latents.get(key) for key in wanted} == wanted
    assert kept_by_the_decode["origin"] == wanted       # the decode bundle is told the same


@pytest.mark.parametrize("resume", ["latents", "decode"])
def test_a_resumed_run_names_the_remover_and_says_it_was_resumed(monkeypatch, tmp_path, resume):
    picture = tmp_path / "cutout.png"
    picture.write_bytes(b"png")
    kept = {"images": [str(picture)], "image_sha256": _sha256(picture), "matte_model": "birefnet-general-lite"}

    if resume == "latents":
        _, output_path, handed_on = _resume_from_latents(monkeypatch, tmp_path, kept)
        assert handed_on["origin"] == kept      # so a decode bundle saved now can be resumed the same way
    else:
        _, output_path = _resume_from_decode(monkeypatch, tmp_path, kept)

    record = read_model_record(output_path)
    assert _licence_parts(record)["rembg/birefnet-general-lite"]["purpose"] == "background removal"
    assert record["model"]["resumed_from"] == resume
    assert record["input"] == {"path": str(picture), "sha256": _sha256(picture), "source": "user-provided"}
    assert json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))["matte_model"] == "birefnet-general-lite"


def test_a_resumed_run_does_not_take_another_file_at_the_pictures_old_path_for_the_picture(monkeypatch, tmp_path):
    picture = tmp_path / "cutout.png"
    picture.write_bytes(b"the picture the model was made from")
    kept = {"images": [str(picture)], "image_sha256": _sha256(picture), "matte_model": None}
    picture.write_bytes(b"something else saved under the same name since")

    decode_path, output_path = _resume_from_decode(monkeypatch, tmp_path, kept)

    assert read_model_record(output_path)["input"] == {
        "path": str(decode_path), "sha256": _sha256(decode_path), "source": "resumed-bundle"}


def test_an_older_bundle_with_a_relative_picture_path_is_not_looked_up_from_the_current_folder(monkeypatch, tmp_path):
    # Bundles made before the hash was kept: a relative path is tried against wherever the
    # resume happens to run, and a file of that name there need not be the picture.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "cutout.png").write_bytes(b"not the picture")
    monkeypatch.chdir(elsewhere)

    decode_path, output_path = _resume_from_decode(monkeypatch, tmp_path, {"images": ["cutout.png"]})

    assert read_model_record(output_path)["input"] == {
        "path": str(decode_path), "sha256": _sha256(decode_path), "source": "resumed-bundle"}


def test_an_older_bundle_takes_the_remover_from_the_first_runs_own_file(monkeypatch, tmp_path):
    # Before the bundle kept it, the remover's name was only in <name>.json, which a finished
    # debug run leaves beside <name>_latents.pt and <name>_decode.pt.
    (tmp_path / "model.json").write_text(json.dumps({"matte_model": "u2net"}), encoding="utf-8")

    _, output_path = _resume_from_decode(monkeypatch, tmp_path, {"images": [str(tmp_path / "gone.png")]})

    assert "rembg/u2net" in _licence_parts(read_model_record(output_path))


def _git(folder, *args):
    return subprocess.run(
        ["git", "-C", str(folder), "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True).stdout.strip()


def test_backend_revision_is_the_commit_of_the_trellis_checkout(tmp_path):
    checkout = tmp_path / "TRELLIS.2"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "commit", "-q", "--allow-empty", "-m", "pin")

    assert gen.backend_revision(tmp_path) == _git(checkout, "rev-parse", "HEAD")


def test_backend_revision_is_nothing_when_the_checkout_is_not_a_clone(tmp_path):
    # vendor/ sits inside this repository: asking git about a plain folder there would answer
    # with this repository's commit, and the record would name the wrong code.
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "the repository around it")
    (tmp_path / "TRELLIS.2").mkdir()

    assert gen.backend_revision(tmp_path) is None
    assert gen.backend_revision(tmp_path / "nowhere") is None


# --- environment configuration ---
def test_configure_environment_sets_sdpa(monkeypatch, tmp_path):
    for key in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "PYTORCH_ENABLE_MPS_FALLBACK",
                "FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
        monkeypatch.delenv(key, raising=False)
    gen.configure_environment(tmp_path, "sdpa")
    import os
    assert os.environ["ATTN_BACKEND"] == "sdpa"
    assert os.environ["SPARSE_ATTN_BACKEND"] == "sdpa"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_configure_environment_overrides_stale_attention_env(monkeypatch, tmp_path):
    """A stale inherited ATTN_BACKEND must not silently win (the conv_none crash class)."""
    import os

    monkeypatch.setenv("ATTN_BACKEND", "flash_attn")
    gen.configure_environment(tmp_path, "sdpa")
    assert os.environ["ATTN_BACKEND"] == "sdpa"


def test_require_flex_gemm_pins_flex_gemm(monkeypatch):
    import os
    import sys
    import types

    monkeypatch.setitem(sys.modules, "flex_gemm", types.ModuleType("flex_gemm"))
    monkeypatch.delenv("SPARSE_CONV_BACKEND", raising=False)
    gen.require_flex_gemm()
    assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"


def test_require_flex_gemm_raises_instead_of_invalid_none(monkeypatch):
    """flex_gemm is mandatory on MPS; 'none' is not a real backend, so fail loudly."""
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "flex_gemm":
            raise ImportError("dlopen: library not loaded, stale rpath")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(RuntimeError, match="flex_gemm"):
        gen.require_flex_gemm()


# --- filesystem verification ---
def test_load_pipeline_refuses_when_the_snapshot_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="about 16.5 GB") as raised:
        gen.load_pipeline(tmp_path, "sdpa", False)
    assert "Mac clean port" in str(raised.value)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _space_tree(tmp_path: Path, *, patched: bool) -> Path:
    vendor = tmp_path / "trellis-space-mac"
    pipeline = vendor / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    _touch(pipeline)
    if patched:
        pipeline.write_text("        new_pipeline.rembg_model = None\n")
    else:
        pipeline.write_text(
            "        new_pipeline.rembg_model = getattr(rembg, "
            "args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
        )
    for relative in (
        ".venv/bin/python",
        "TRELLIS.2/trellis2/representations/mesh/base.py",
        "deps/trellis2-apple/o-voxel/o_voxel/postprocess.py",
        "deps/mtlmesh/src/metal/remesh.metal",
    ):
        _touch(vendor / relative)
    return vendor


def test_bria_refusal_names_the_patch_until_the_tree_is_disabled(tmp_path):
    refusal = gen.bria_refusal(_space_tree(tmp_path, patched=False))
    assert refusal and "BRIA RMBG-2.0" in refusal
    assert "patch_trellis_space_core.py" in refusal
    assert gen.bria_refusal(_space_tree(tmp_path / "ok", patched=True)) is None


def test_load_pipeline_refuses_an_unpatched_tree_before_torch(tmp_path, monkeypatch):
    vendor = _space_tree(tmp_path, patched=False)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "pipeline.json").write_text("{}\n")
    from image_to_3dlab import weight_gate

    monkeypatch.setattr(weight_gate, "snapshot_dir", lambda *a, **k: snap)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="BRIA RMBG-2.0"):
        gen.load_pipeline(vendor, "sdpa", False)


def test_check_refuses_an_unpatched_tree_before_torch(tmp_path, monkeypatch, capsys):
    vendor = _space_tree(tmp_path, patched=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert gen.check_environment(vendor, "sdpa") == 1
    assert "BRIA RMBG-2.0" in capsys.readouterr().out


def test_verify_paths_flags_missing_build(tmp_path):
    problems = gen.verify_paths(tmp_path / "does-not-exist")
    assert problems  # every expected artifact is missing
    assert any(".venv python" in p for p in problems)


# --- manifest builder ---
def test_build_manifest_shape():
    m = gen.build_manifest(
        image="in.png", output="out.glb",
        params={"decimation_target": 300000}, pipeline_type="1024_cascade", seed=0,
        timings={"total": 1.0}, artifacts={"glb": {"bytes": 10}},
        load_rembg=False, sparse_attn_backend="sdpa",
    )
    assert m["schema_version"] == 1
    assert m["attn_backend"] == "sdpa"
    assert m["device"] == "mps"
    assert m["load_rembg"] is False
    assert m["pipeline_type"] == "1024_cascade"
    assert m["params"]["decimation_target"] == 300000
    assert m["timings_seconds"]["total"] == 1.0


@pytest.mark.parametrize("backend,env,expected", [
    ("mlx", "fp16", "fp16"),
    ("mlx", "FP16", "fp16"),
    ("mlx", "fp32", "fp32"),
    ("mlx", None, "fp32"),
    ("sdpa", "fp16", None),
    ("metal_flash", None, None),
])
def test_resolved_attention_dtype(backend, env, expected):
    assert gen.resolved_attention_dtype(backend, env) == expected


def test_manifest_records_the_attention_precision():
    """Precision travels by environment, not by flag, so it must be recorded explicitly.

    Without it two runs that computed different things produce identical manifests, and a
    comparison made days later cannot be interpreted.
    """
    m = gen.build_manifest(
        image="in.png", output="out.glb", params={}, pipeline_type="512", seed=1,
        timings={}, artifacts={}, load_rembg=False, sparse_attn_backend="mlx",
        sparse_attn_dtype=gen.resolved_attention_dtype("mlx", "fp16"),
    )
    assert m["sparse_attn_dtype"] == "fp16"


def test_manifest_reports_no_precision_for_non_mlx_backends():
    m = gen.build_manifest(
        image="in.png", output="out.glb", params={}, pipeline_type="512", seed=1,
        timings={}, artifacts={}, load_rembg=False, sparse_attn_backend="sdpa",
        sparse_attn_dtype=gen.resolved_attention_dtype("sdpa", "fp16"),
    )
    # None, not "fp32": the stock path has no MLX precision to report, and a default here
    # would be a plausible-looking lie in the provenance record.
    assert m["sparse_attn_dtype"] is None


# --- CPU pre-cap ratio (the MPS decode->GLB path: fast_simplification before Metal to_glb) ---
def test_precap_ratio_math():
    # 20M faces -> 4M cap means removing 80% of faces
    assert gen.precap_ratio(20_000_000, 4_000_000) == 0.8
    assert gen.precap_ratio(8_000_000, 4_000_000) == 0.5
    # already at or under the cap -> no-op (0.0 = fast_simplification keeps everything)
    assert gen.precap_ratio(4_000_000, 4_000_000) == 0.0
    assert gen.precap_ratio(1_000_000, 4_000_000) == 0.0


def test_precap_ratio_rejects_bad_cap():
    with pytest.raises(ValueError):
        gen.precap_ratio(100, 0)
    with pytest.raises(ValueError):
        gen.precap_ratio(100, -5)


# --- guard the demo constants against accidental drift ---
def test_demo_params_integrity():
    assert gen.DEMO_PARAMS["seed"] == 0
    assert gen.DEMO_PARAMS["texture_size"] == 2048
    # decimation_target is under active review (300k app.py vs 3M live demo); assert it is a
    # positive int rather than pinning a value we may deliberately change.
    assert isinstance(gen.DEMO_PARAMS["decimation_target"], int)
    assert gen.DEMO_PARAMS["decimation_target"] > 0
    assert gen.DEMO_PARAMS["remesh"] == {"remesh": True, "remesh_band": 1, "remesh_project": 0}
    for stage in ("sparse_structure", "shape_slat", "tex_slat"):
        assert gen.DEMO_PARAMS[stage]["steps"] == 12


# --- post-pre-cap corruption filter (fast_simplification emits stray out-of-range indices) ---
def test_filter_out_of_range_faces_drops_bad():
    import torch

    faces = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 2, 99], [4, 5, 6]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 1
    assert kept.tolist() == [[0, 1, 2], [3, 4, 5], [4, 5, 6]]


def test_filter_out_of_range_faces_clean_input_untouched():
    import torch

    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 0
    assert kept.shape[0] == 2
    assert kept.dtype == faces.dtype


def test_filter_out_of_range_faces_negative_index():
    import torch

    faces = torch.tensor([[0, 1, 2], [-1, 4, 5]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 1
    assert kept.tolist() == [[0, 1, 2]]


def test_manifest_records_our_matte_and_never_bria():
    m = gen.build_manifest(image="i.png", output="o.glb", params={}, pipeline_type="512", seed=0,
                           timings={}, artifacts={}, load_rembg=False, sparse_attn_backend="sdpa",
                           matte_model="birefnet-general-lite")
    assert m["matte_model"] == "birefnet-general-lite" and m["load_rembg"] is False


def test_wrapper_no_longer_offers_bria():
    source = Path(gen.__file__).read_text()
    assert "--allow-rembg" not in source and "--matted-with" in source


def test_an_install_with_the_older_patch_is_upgraded_not_refused(tmp_path, capsys):
    """Macs patched before the guardrail tightened kept BRIA behind `if load_rembg`.
    Refusing them stopped every TRELLIS run while Setup still said ready."""
    vendor = _space_tree(tmp_path, patched=True)
    pipeline = vendor / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    pipeline.write_text(
        '    def from_pretrained(path: str, *, load_rembg: bool = True) '
        '-> "Trellis2ImageTo3DPipeline":\n'
        "        new_pipeline.rembg_model = (\n"
        "            getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
        "            if load_rembg else None\n"
        "        )\n"
    )
    assert gen.bria_refusal(vendor) is None
    text = pipeline.read_text()
    assert "new_pipeline.rembg_model = None" in text and "getattr(rembg" not in text
    assert "upgraded" in capsys.readouterr().out


def test_a_pristine_tree_is_still_refused_not_patched_behind_the_users_back(tmp_path):
    vendor = _space_tree(tmp_path, patched=False)
    pipeline = vendor / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    before = pipeline.read_text()
    assert "BRIA RMBG-2.0" in gen.bria_refusal(vendor)
    assert pipeline.read_text() == before
