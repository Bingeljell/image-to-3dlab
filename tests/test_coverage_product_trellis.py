"""Test TRELLIS job preparation and post-processing without the vendor runtime."""

import os
import struct
import subprocess
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from image_to_3dlab import matte
from image_to_3dlab import trellis_backend as t


@pytest.fixture
def trellis_checkout(tmp_path):
    repo = tmp_path / "checkout"
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (repo / "generate.py").touch()
    pipeline = repo / "TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("pipeline.rembg_model = None\n")
    return repo, pipeline


@pytest.fixture
def local_trellis_weights(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    snapshot = hub / "models--microsoft--TRELLIS.2-4B/snapshots/local"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return snapshot


def make_image(path, mode):
    Image.new(mode, (2, 2), "red" if mode == "RGB" else (255, 0, 0, 100)).save(path)
    return path


@pytest.mark.parametrize("mode", ["full", "minimal", "crash", "no-result"])
def test_trellis_worker_inputs_and_result(trellis_checkout, local_trellis_weights, monkeypatch, tmp_path, mode):
    repo, _ = trellis_checkout
    first = make_image(tmp_path / "front.png", "RGBA")
    second = make_image(tmp_path / "back.png", "RGBA")
    directory = tmp_path / "output"
    directory.mkdir()
    stale = directory / "front_trellis2_basecolor.png"
    stale.write_bytes(b"old")
    full = mode == "full"
    options = t.TrellisOptions(repo, seed=9, pre_simplify_cap=100 if full else None,
                               normalize_material=full, fix_winding=full, material_mode="matte")
    calls = []

    def worker(argv, *, cwd, env, check):
        calls.append(argv)
        assert not stale.exists()
        assert cwd == repo and check
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert env.get("I2L_PRE_CAP") == ("100" if full else None)
        if env.get("DEVELOPER_DIR"):
            assert env["DEVELOPER_DIR"] == "/Applications/Xcode.app/Contents/Developer"
        prepared = [directory / "front_trellis_input_view0.png", directory / "back_trellis_input_view1.png"] if full else [
            directory / "front_trellis_input.png",
        ]
        assert argv[2:2 + len(prepared)] == list(map(str, prepared))
        for path in prepared:
            with Image.open(path) as image:
                assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 100
        assert argv[argv.index("--seed") + 1] == "9"
        if mode == "crash":
            raise subprocess.CalledProcessError(17, argv)
        if mode != "no-result":
            trimesh.creation.box().export(str(Path(argv[argv.index("--output") + 1]).with_suffix(".glb")))
        if full:
            stale.write_bytes(b"new")

    monkeypatch.delenv("I2L_PRE_CAP", raising=False)
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(t.subprocess, "run", worker)
    inputs = [first, second] if full else first
    if mode in ("crash", "no-result"):
        with pytest.raises(RuntimeError, match="exit code 17" if mode == "crash" else "did not create"):
            t.generate_trellis(inputs, directory, options)
    else:
        result = t.generate_trellis(inputs, directory, options)
        assert result.asset == directory / "front_trellis2.glb"
        assert result.texture_backend == ("kdtree-cpu" if full else "metal-o-voxel")
        assert result.material_normalized is full
        assert result.material_mode == ("matte" if full else None)
        assert result.winding_repaired is full
        assert trimesh.load(str(result.asset), force="mesh").volume == pytest.approx(1)
    assert len(calls) == 1


@pytest.mark.parametrize(("fault", "error", "message"), [
    ("checkout", RuntimeError, "environment not found"),
    ("patch", RuntimeError, "BRIA-disable patch is missing"),
    ("views", ValueError, "at least one input view"),
])
def test_trellis_preflight_never_starts_worker(trellis_checkout, local_trellis_weights, monkeypatch, tmp_path,
                                              fault, error, message):
    repo, pipeline = trellis_checkout
    if fault == "checkout":
        (repo / "generate.py").unlink()
    if fault == "patch":
        pipeline.write_text("pipeline.rembg_model = object()\n")
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **kw: pytest.fail("worker launched before preflight"))
    with pytest.raises(error, match=message):
        t.generate_trellis([], tmp_path / "output", t.TrellisOptions(repo))
    assert os.environ.get("HF_HUB_OFFLINE") == ("1" if fault == "views" else None)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("transparent", [False, True])
def test_trellis_removes_background_or_refuses_opaque_result(tmp_path, monkeypatch, transparent):
    image = make_image(tmp_path / "picture.png", "RGB")
    output = tmp_path / "rgba.png"

    def remove(source):
        assert source.mode == "RGBA"
        source.putalpha(100 if transparent else 255)
        return source, None

    monkeypatch.setattr(matte, "cut_out", remove)
    if transparent:
        assert t._prepare_rgba(image, output) == output
        with Image.open(output) as saved:
            assert saved.getpixel((0, 0))[3] == 100
    else:
        with pytest.raises(RuntimeError, match="transparent foreground"):
            t._prepare_rgba(image, output)
        assert not output.exists()


@pytest.mark.parametrize(("magic", "message"), [(0, "not a binary glTF"), (0x46546C67, "no glTF JSON")])
def test_glb_reader_rejects_invalid_container(tmp_path, magic, message):
    path = tmp_path / "bad.glb"
    path.write_bytes(struct.pack("<III", magic, 2, 12))
    with pytest.raises(RuntimeError, match=message):
        t.read_glb(path)


def test_failed_winding_repair_leaves_invalid_file_available(tmp_path, capsys):
    mesh = tmp_path / "broken.glb"
    original = b"not a mesh"
    mesh.write_bytes(original)
    assert t._repair_winding(mesh) == (False, False)
    assert mesh.read_bytes() == original
    assert "winding repair skipped" in capsys.readouterr().out


def test_trellis_missing_weights_cannot_start_worker(trellis_checkout, monkeypatch, tmp_path):
    repo, _ = trellis_checkout
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr(t.subprocess, "run", lambda *a, **kw: pytest.fail("worker should not be called"))
    with pytest.raises(SystemExit, match=r"TRELLIS\.2.*16\.5 GB"):
        t.generate_trellis(tmp_path / "in.png", tmp_path / "output", t.TrellisOptions(repo))
    assert not (tmp_path / "output").exists()
