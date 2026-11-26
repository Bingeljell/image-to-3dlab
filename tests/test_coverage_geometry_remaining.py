"""Local fixtures for command safeguards and error paths."""

import io
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_model_weights as audit


@pytest.mark.parametrize("payload", [b"short", (8).to_bytes(8, "little") + b"short"])
def test_checkpoint_audit_rejects_file_truncated_during_read(payload):
    path = SimpleNamespace(
        stat=lambda: SimpleNamespace(st_size=16),
        open=lambda mode: io.BytesIO(payload),
    )
    assert not audit.conversion_proven(path)


def test_checkpoint_audit_skips_missing_roots_and_symlinks(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    model = local / "local.safetensors"
    model.write_bytes(b"model")
    (local / "nested").mkdir()
    assert audit.find_superseded_checkpoints([tmp_path / "missing"]) == []
    assert audit._local_files([local]) == {(model.name, 5): model}


def test_audit_only_collects_requested_rule_and_removes_local_fixture(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "model").write_bytes(b"123")
    assert audit.collect((), hub=tmp_path / "absent") == []
    assert audit.report([audit.Candidate("cached", local, 3, "fixture")], apply=True) == 0
    assert not local.exists()


@pytest.mark.parametrize("name", [
    "audit_model_weights.py", "pod_smoke_test.py", "trellis_space_generate.py",
    "trellis_cuda_generate.py", "hunyuan_cuda_generate.py", "trellis_stage3.py",
    "hero_route.py", "repair_decode.py", "export_decode_highpoly.py", "make_side_views.py",
    "attach_normal_map.py", "fix_winding.py", "feature_mask.py", "soften_markings.py",
    "remove_loose_parts.py", "classify_thickness.py",
])
def test_command_help_exits_without_starting_work(monkeypatch, name, capsys, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    original_path = sys.path
    original_entries = sys.path.copy()
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        root_paths = {str(scripts), str(scripts.parent), ""}
        isolated_path = [entry for entry in sys.path if entry not in root_paths]
        if name not in {"trellis_space_generate.py", "trellis_cuda_generate.py", "hunyuan_cuda_generate.py",
                        "trellis_stage3.py"}:
            isolated_path.insert(0, str(scripts))
        patch.setattr(sys, "path", isolated_path)
        patch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as status:
            run_path(str(scripts / name), run_name="__main__")
    assert sys.path is original_path
    assert sys.path == original_entries
    assert status.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("enabled", [False, True])
def test_trellis_cuda_check_requires_device_and_pipeline_surface(monkeypatch, tmp_path, capsys, enabled):
    from scripts import trellis_cuda_generate as worker

    for path in worker.bria_patch.pipeline_files(worker.checkout(tmp_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(worker.bria_patch.REPLACEMENT)
    monkeypatch.setattr(worker, "verify_paths", lambda _root, _route: [])
    monkeypatch.setattr(worker, "configure_environment", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    cuda = SimpleNamespace(is_available=lambda: enabled, get_device_name=lambda _id: "GPU fixture")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    pipeline = SimpleNamespace(run=True, decode_latent=True, preprocess_image=True)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines.trellis2_image_to_3d",
                        SimpleNamespace(Trellis2ImageTo3DPipeline=pipeline))
    monkeypatch.setitem(sys.modules, "o_voxel", SimpleNamespace(postprocess=SimpleNamespace(to_glb=True)))
    routes = [r for r in vars(worker.trellis_route).values() if isinstance(r, worker.trellis_route.Route)]
    route = next(r for r in routes if not worker.uses_aule(r))
    assert worker.check_environment(tmp_path, route) == (0 if enabled else 1)
    text = capsys.readouterr().out
    assert ("run/decode_latent/to_glb resolved" if enabled else "torch cannot see a CUDA device") in text
    if enabled:
        del pipeline.run
        assert worker.check_environment(tmp_path, route) == 1
        assert "TRELLIS.2 surface missing: ['run']" in capsys.readouterr().out


@pytest.mark.parametrize("available", [None, False, True])
def test_hunyuan_check_validates_weights_and_cuda_without_loading_model(monkeypatch, tmp_path, capsys, available):
    from scripts import hunyuan_cuda_generate as worker

    model = tmp_path / "realesrgan.pth"
    monkeypatch.setattr(worker, "verify_paths", lambda _root: [])
    monkeypatch.setattr(worker, "realesrgan_path", lambda _root: model)
    entered = []
    monkeypatch.setattr(worker, "_enter_upstream", entered.append)
    cuda = SimpleNamespace(is_available=lambda: available, get_device_name=lambda _id: "GPU fixture")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    modules = {
        "custom_rasterizer": {},
        "DifferentiableRenderer": {"mesh_inpaint_processor": object()},
        "hy3dshape.pipelines": {"Hunyuan3DDiTFlowMatchingPipeline": object()},
        "textureGenPipeline": {"Hunyuan3DPaintConfig": object(), "Hunyuan3DPaintPipeline": object()},
    }
    for name, features in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**features))
    if available is not None:
        model.touch()
    assert worker.main(["--check", "--vendor-root", str(tmp_path)]) == (0 if available else 1)
    assert entered == ([] if available is None else [tmp_path])
    expected = {None: "missing RealESRGAN", False: "torch cannot see", True: "shape + paint"}
    assert expected[available] in capsys.readouterr().out


def test_trellis_reports_when_sampling_never_reaches_decode():
    from scripts import trellis_space_generate as worker

    def decode(*args):
        pytest.fail("sampling should not decode")

    pipeline = SimpleNamespace(decode_latent=decode, run=lambda *args, **kwargs: (object(), ()))
    with pytest.raises(RuntimeError, match="did not reach the postponed decode boundary"):
        worker.sample_latents_without_decode(pipeline, object())
    assert pipeline.decode_latent is decode


def test_trellis_mps_cache_release_tolerates_cpu_torch(monkeypatch):
    from scripts import trellis_space_generate as worker

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    worker._release_mps_memory()


@pytest.mark.parametrize("problem", ["missing", "rank", "columns", "resolution", "model"])
def test_trellis_stage3_rejects_incompatible_latent_shapes(problem):
    from scripts import trellis_stage3 as stage

    bundle = {
        "shape_slat_feats": np.zeros((2, 3)), "coords": np.zeros((2, 4)),
        "res": 512, "pipeline_type": "512", "images": ["fixture.png"],
    }
    if problem == "missing":
        bundle.pop("images")
        expected = "latent bundle missing: images"
    elif problem == "rank":
        bundle["coords"] = np.zeros(2)
        expected = "rank-2"
    elif problem == "columns":
        bundle["coords"] = np.zeros((2, 3))
        expected = "4 columns"
    else:
        bundle["res"] = 256
        expected = "unsupported latent resolution"
    with pytest.raises(ValueError, match=expected):
        if problem == "model":
            stage.material_stage_contract(bundle)
        else:
            stage.validate_bundle(bundle)


def test_trellis_stage3_checks_latent_input_before_loading_weights(monkeypatch, tmp_path):
    from scripts import trellis_stage3 as stage

    monkeypatch.setattr(sys, "argv", ["stage3", str(tmp_path / "absent.pt"), str(tmp_path / "out.pt"), "--texture-seed", "1"])
    with pytest.raises(SystemExit, match="missing latent bundle"):
        stage.main()


def test_trellis_stage3_derives_decode_names_for_unlabelled_legacy_cache(tmp_path):
    from scripts import trellis_stage3 as stage

    assert stage.default_geometry_decode(tmp_path / "shape.pt") == tmp_path / "shape_decode.pt"


@pytest.mark.parametrize("fallback", [False, True])
def test_hunyuan_and_trellis_announce_background_remover_fallback(monkeypatch, tmp_path, capsys, fallback):
    from PIL import Image

    from image_to_3dlab import matte
    from scripts import hunyuan_cuda_generate, trellis_cuda_generate

    source = tmp_path / "source.png"
    target = tmp_path / "finished.glb"
    Image.new("RGB", (3, 3), "white").save(source)
    cutout = Image.new("RGBA", (3, 3))
    cutout.putpixel((1, 1), (80, 0, 0, 255))
    remover = matte.FALLBACK_MODEL if fallback else matte.LEAN_MODEL
    monkeypatch.setattr(matte, "cut_out", lambda image: (cutout, remover))
    for worker, options in ((hunyuan_cuda_generate, {}), (trellis_cuda_generate, {"allow_uncut": False})):
        path, model = worker.prepare_image(source, target, force_matte=True, **options)
        assert model == remover
        with Image.open(path) as result:
            assert result.getpixel((1, 1)) == (80, 0, 0, 255)
    assert capsys.readouterr().out.count("using u2net") == 2 * fallback


def test_trellis_cuda_uncut_edge_requires_conscious_override(tmp_path):
    from PIL import Image

    from scripts import trellis_cuda_generate as worker

    source = tmp_path / "source.png"
    partial = Image.new("RGBA", (8, 8), "white")
    partial.paste((0, 0, 0, 0), (2, 2, 6, 6))
    partial.save(source)
    out = tmp_path / "mesh.glb"
    with pytest.raises(SystemExit, match="border"):
        worker.prepare_image(source, out, force_matte=False, allow_uncut=False)
    assert worker.prepare_image(source, out, force_matte=False, allow_uncut=True) == (source, None)


@pytest.mark.parametrize("export_glb", [False, True])
def test_decode_repair_keeps_attributes_and_exports_optional_glb(monkeypatch, tmp_path, export_glb):
    import torch
    import trimesh

    from scripts import repair_decode

    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source, dest = tmp_path / "distorted.pt", tmp_path / "repaired.pt"
    glb = tmp_path / "preview" / "repaired.glb"
    torch.save({"vertices": torch.tensor(triangle), "faces": torch.tensor([[0, 1, 2]]), "seed": 7}, source)
    args = ["repair_decode", str(source), str(dest)]
    if export_glb:
        args.extend(["--export-glb", str(glb)])
    monkeypatch.setattr(sys, "argv", args)
    assert repair_decode.main() == 0
    repaired = torch.load(dest, weights_only=False)
    assert repaired["seed"] == 7 and repaired["faces"].shape == (1, 3)
    np.testing.assert_allclose(np.sort(repaired["vertices"].numpy(), axis=0), np.sort(triangle, axis=0))
    assert glb.exists() is export_glb
    if export_glb:
        assert len(trimesh.load(glb, force="mesh").faces) == 1


@pytest.mark.parametrize("fragment", [None, False, True])
def test_highpoly_cli_warns_only_when_decimation_shatters_surface(monkeypatch, tmp_path, capsys, fragment):
    import torch
    import trimesh

    from scripts import export_decode_highpoly

    mesh = trimesh.creation.box()
    source, dest = tmp_path / "decode.pt", tmp_path / "source.ply"
    torch.save({"vertices": torch.tensor(mesh.vertices), "faces": torch.tensor(mesh.faces)}, source)

    def simplify(vertices, faces, *, target_count):
        assert target_count == 3 and faces.shape == (12, 3)
        if fragment:
            triangles = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            return np.concatenate([triangles + [0, 0, z] for z in (0, 2, 4)]), np.arange(9).reshape(3, 3)
        return vertices, faces[:3]

    monkeypatch.setitem(sys.modules, "fast_simplification", SimpleNamespace(simplify=simplify))
    args = ["export_decode_highpoly", str(source), str(dest), "--no-repair"]
    if fragment is not None:
        args.extend(["--target-faces", "3"])
    monkeypatch.setattr(sys, "argv", args)
    assert export_decode_highpoly.main() == 0
    assert len(trimesh.load(dest, force="mesh").faces) == (12 if fragment is None else 3)
    assert ("WARNING: decimation multiplied" in capsys.readouterr().out) is bool(fragment)
    assert export_decode_highpoly.component_count(mesh.vertices, np.empty((0, 3), dtype=int)) == 0


def test_hero_preserves_legacy_notes_and_refuses_to_finish_without_decode(tmp_path, monkeypatch):
    from scripts import hero_route as hero

    out = tmp_path / "run"
    out.mkdir()
    (out / "unsure.json").write_text('["original note"]')
    operation = hero.Run(out)
    try:
        assert operation.notes == {"earlier": ["original note"]}
    finally:
        operation.log.close()

    from PIL import Image

    picture = tmp_path / "image.png"
    Image.new("RGB", (3, 3)).save(picture)
    (out / "picture_cutout.png").write_bytes(picture.read_bytes())
    (out / "body_raw.glb").touch()
    import pixal3d_generate

    monkeypatch.setattr(pixal3d_generate, "dump_supported", lambda: True)
    # Existing artifacts bypass the preflight dependencies and generation.
    monkeypatch.setattr(hero, "needed_parts", dict)
    monkeypatch.setattr(sys, "argv", ["hero_route", str(picture), "--out", str(out)])
    with pytest.raises(SystemExit, match="no clean-up data"):
        hero.main()


@pytest.mark.parametrize("failure", ["vision", "right", "left", None])
def test_side_views_reports_engine_failures_and_keeps_drawn_view_notes(tmp_path, monkeypatch, capsys, failure):
    import json

    from PIL import Image

    from scripts import make_side_views as views

    source, right, left = [tmp_path / name for name in ("front.png", "right.png", "left.png")]
    front = Image.new("RGBA", (8, 8))
    front.paste((40, 30, 20, 255), (2, 2, 6, 6))
    front.save(source)
    engine = SimpleNamespace(vision_file=True, label="Fixture")
    monkeypatch.setattr(views.image_api, "pick_engine", lambda name, *, edit: engine)
    monkeypatch.setattr(views.image_api, "resolve_vision",
                        lambda _engine: None if failure == "vision" else tmp_path / "vision")
    monkeypatch.setattr(views.image_api, "resolve_weights", lambda _engine: {})
    monkeypatch.setattr(views.image_api, "clean_settings", lambda settings, *, engine: settings)
    monkeypatch.setattr(views.image_api, "provenance", lambda *args, **kwargs: {"fixture": True})
    monkeypatch.setattr(views, "command_for",
                        lambda reference, pic, seed, weights, vision, engine: ["fixture", str(reference), str(pic)])
    monkeypatch.setattr(views.leg_check, "check_picture", lambda pic, view: ({"open_under_body": .8}, []))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if (failure == "right" and len(calls) == 1) or (failure == "left" and len(calls) == 2):
            return SimpleNamespace(returncode=1, stdout="", stderr="fixture engine failed")
        Image.new("RGBA", (8, 8), (80, 30, 20, 255)).save(command[2])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(views.subprocess, "run", run)
    result = views.main([str(source), str(right), str(left), "--draw-left", "--four-legged", "--tries", "1"])
    assert result == int(failure is not None)
    assert len(calls) == {"vision": 0, "right": 1, "left": 2, None: 2}[failure]
    if failure == "vision":
        assert "needs its vision file" in capsys.readouterr().out
    if failure in {"right", "left"}:
        assert "fixture engine failed" in capsys.readouterr().out
        assert not left.exists()
    if failure is None:
        record = json.loads(left.with_suffix(".provenance.json").read_text())
        assert record["legs"] == {"open_under_body": .8, "problems": [], "pictures_drawn": 1}


def test_weight_audit_combines_selected_rules_against_local_empty_roots(monkeypatch, tmp_path):
    roots = (tmp_path / "empty",)
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(audit.find_superseded_checkpoints, "__defaults__", (roots,))
    monkeypatch.setattr(audit.find_cached_duplicates, "__defaults__", (hub, roots))
    assert audit.collect(("ckpt", "cached"), hub) == []


def test_trellis_decode_cache_names_and_keeps_valid_cpu_tensors(tmp_path):
    import torch

    from scripts import trellis_space_generate as worker

    vertices = torch.tensor([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    faces = torch.tensor([[0, 1, 2]])
    mesh = SimpleNamespace(vertices=vertices, faces=faces, attrs=torch.ones((3, 1)),
                           coords=torch.zeros((3, 3)))
    pipeline = SimpleNamespace(decode_latent=lambda *_args: [mesh], pbr_attr_layout={"base_color": [0]})
    output = tmp_path / "fixture.glb"
    bundle, _ = worker._decode_and_cache(
        pipeline, object(), object(), 512, output,
        save_decode=True, decode_path=None, pipeline_type="512", seed=15, origin={"source": "fixture.png"},
    )
    cache = torch.load(tmp_path / "fixture_decode.pt", map_location="cpu", weights_only=False)
    assert cache["seed"] == bundle["seed"] == 15
    assert cache["source"] == "fixture.png"
    assert torch.equal(cache["faces"], faces)
