"""Exercise TRELLIS export helpers through the installed vendor code."""

import importlib
import runpy
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import patch_ovoxel_remesh_checkpoints as checkpoints
from scripts import patch_trellis_dump_decode as decode
from scripts import patch_trellis_highpoly as highpoly
from scripts import patch_trellis_quality as quality

GENERATOR = '''import argparse

def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--texture-size", type=int, default=1024,
        choices=[512, 1024, 2048],
    )
    parser.add_argument(
        "--no-texture", action="store_true",
    )
    return parser.parse_args(argv)

def run(outputs, args, o_voxel):
    mesh_out = outputs[0] if isinstance(outputs, list) else outputs
    verts, faces = mesh_out.vertices.numpy(), mesh_out.faces.numpy()
    print(f"\\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
    target_faces, tex_size = 15, args.texture_size
    if not args.no_texture:
        if mesh_out is not None:
            if verts.shape[0]:
                glb = o_voxel.postprocess.to_glb(
                    decimation_target=target_faces,
                    texture_size=tex_size,
                    verbose=True,
                )
                return glb
    return None
'''


@pytest.fixture
def mesh():
    torch = pytest.importorskip("torch")
    return SimpleNamespace(
        vertices=torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.float64),
        faces=torch.tensor([[0, 2, 1]]), attrs=torch.tensor([[0.1, 0.2, 0.3]]),
        coords=torch.tensor([[0, 0, 0]]), layout={"color": [0, 1, 2]}, voxel_size=0.125,
    )


@pytest.mark.parametrize("packing", ["supported", "silent", "warn"])
def test_generator_dispatches_exports_and_optional_packing(tmp_path, monkeypatch, capsys, mesh, packing):
    torch = pytest.importorskip("torch")
    from scripts import trellis_export_compat as exports

    generator = tmp_path / "generate.py"
    generator.write_text(GENERATOR)
    highpoly.patch(generator)
    quality.patch(generator)
    monkeypatch.setattr(sys, "argv", [decode.__file__, "--path", str(generator)])
    assert decode.main() == 0
    helper = tmp_path / "_i2l_trellis_export.py"
    assert helper.read_bytes() == Path(exports.__file__).read_bytes()
    initial = generator.read_bytes(), helper.read_bytes()
    assert decode.main() == 0
    highpoly.patch(generator)
    quality.patch(generator)
    assert (generator.read_bytes(), helper.read_bytes()) == initial

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "_i2l_trellis_export", raising=False)
    namespace = runpy.run_path(str(generator))
    ply, cache = tmp_path / "mesh/raw.ply", tmp_path / "decoded.pt"
    args = ["--dump-highpoly", str(ply), "--dump-decode", str(cache), "--texture-size", "4096",
            "--remesh", "--uv-cone-degrees", "45"]
    if packing == "silent":
        args.append("--no-uv-brute-force-packing")
    calls = []

    def record(options):
        calls.append(options)
        assert ply.is_file() and cache.is_file()
        return "baked fixture"

    def supported(*, xatlas_pack_charts_kwargs, **options):
        assert xatlas_pack_charts_kwargs == {"brute_force": True}
        return record(options)

    oven = SimpleNamespace(postprocess=SimpleNamespace(to_glb=supported if packing == "supported" else record))
    if packing != "supported":
        def legacy(**options):
            assert "xatlas_pack_charts_kwargs" not in options
            return record(options)
        oven.postprocess.to_glb = legacy
    try:
        assert namespace["run"]([mesh], namespace["parse"](args), oven) == "baked fixture"
        assert namespace["run"](mesh, namespace["parse"](["--no-texture"]), oven) is None
    finally:
        sys.modules.pop("_i2l_trellis_export", None)
    options, = calls
    assert options["decimation_target"] == 15 and options["texture_size"] == 4096
    assert options["remesh"] is True
    assert options["mesh_cluster_threshold_cone_half_angle_rad"] == pytest.approx(np.pi / 4)
    assert ("cannot forward packing options" in capsys.readouterr().out) is (packing == "warn")
    raw = ply.read_bytes()
    header, body = raw.split(b"end_header\n")
    assert header.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in header and b"element face 1\n" in header
    np.testing.assert_array_equal(np.frombuffer(body[:36], dtype="<f4").reshape(-1, 3), mesh.vertices.numpy())
    assert struct.unpack("<B3i", body[36:]) == (3, 0, 2, 1)
    payload = torch.load(cache, weights_only=True)
    assert set(payload) == {"vertices", "faces", "attrs", "coords", "layout", "voxel_size"}
    for field in ["vertices", "faces", "attrs", "coords"]:
        torch.testing.assert_close(payload[field], getattr(mesh, field))
    assert payload["layout"] == mesh.layout and payload["voxel_size"] == 0.125

    monkeypatch.setattr(sys, "argv", [decode.__file__, "--path", str(generator), "--revert"])
    assert decode.main() == 0
    assert decode.state(generator.read_text()) == "absent"
    assert helper.is_file() and "--dump-highpoly" in generator.read_text()


def test_checkpoint_helper_reads_mesh_only_when_enabled(tmp_path, monkeypatch, mesh):
    torch = pytest.importorskip("torch")
    from scripts.trellis_export_compat import save_checkpoint

    monkeypatch.delenv("I2L_CHECKPOINT_PREFIX", raising=False)

    def read():
        pytest.fail("read geometry with checkpoints disabled")

    save_checkpoint("unused", mesh=SimpleNamespace(read=read))
    prefix = tmp_path / "checkpoint/sample"
    monkeypatch.setenv("I2L_CHECKPOINT_PREFIX", str(prefix))
    with pytest.raises(RuntimeError, match="missing has no geometry"):
        save_checkpoint("missing", vertices=mesh.vertices)
    vertices = torch.arange(18, dtype=torch.float32, requires_grad=True).reshape(3, 6)[:, ::2]
    save_checkpoint("actual", vertices=vertices, faces=mesh.faces)
    payload = torch.load(tmp_path / "checkpoint/sample_actual.pt", weights_only=True)
    torch.testing.assert_close(payload["vertices"], vertices)
    assert payload["vertices"].is_contiguous() and not payload["vertices"].requires_grad
    assert payload["vertices"].device.type == "cpu"


def test_checkpoint_patch_installs_helper_and_dispatches_each_stage(tmp_path, monkeypatch, mesh):
    torch = pytest.importorskip("torch")
    from scripts import trellis_export_compat as exports

    package = tmp_path / "fixture_voxel"
    package.mkdir()
    (package / "__init__.py").touch()
    target = package / "postprocess.py"
    target.write_text(
        """# Mesh processing — cumesh auto-selects Metal/CUDA
def to_glb(mesh, out_vertices, out_faces, out_vmaps, remesh, device="cpu"):
    if remesh:
        if False:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        if False:
            print(f"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        if False:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    out_vmaps = out_vmaps.to(device)
    return out_vmaps
"""
    )
    monkeypatch.setattr(sys, "argv", [checkpoints.__file__, "--target", str(target)])
    assert checkpoints.main() == 0
    helper = target.with_name("_i2l_trellis_export.py")
    assert helper.read_bytes() == Path(exports.__file__).read_bytes()
    before = target.read_bytes()
    assert checkpoints.main() == 0 and target.read_bytes() == before
    monkeypatch.setenv("I2L_CHECKPOINT_PREFIX", str(tmp_path / "points/model"))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        vendor = importlib.import_module("fixture_voxel.postprocess")
        vendor.to_glb(SimpleNamespace(read=lambda: (mesh.vertices, mesh.faces)),
                      mesh.vertices + 2, mesh.faces, torch.zeros(1), True)
    finally:
        for module in tuple(sys.modules):
            if module == "fixture_voxel" or module.startswith("fixture_voxel."):
                sys.modules.pop(module)
    for stage in ("after_remesh", "after_cleanup", "after_simplify", "after_uv"):
        payload = torch.load(tmp_path / f"points/model_{stage}.pt", weights_only=True)
        torch.testing.assert_close(payload["faces"], mesh.faces)
        torch.testing.assert_close(payload["vertices"], mesh.vertices + 2 if stage == "after_uv"
                                   else mesh.vertices)


def test_decoded_helper_writes_a_cache_that_rebake_understands(tmp_path, mesh):
    torch = pytest.importorskip("torch")
    from scripts import trellis_rebake as rebake
    from scripts.trellis_export_compat import save_decoded_mesh

    destination = tmp_path / "decode.pt"
    save_decoded_mesh(destination, mesh)
    copied = rebake.load_payload(destination, torch)
    assert copied["layout"] == mesh.layout and copied["voxel_size"] == mesh.voxel_size
    for field in ("vertices", "faces", "attrs", "coords"):
        assert copied[field].device.type == "cpu"
        torch.testing.assert_close(copied[field], getattr(mesh, field))


@pytest.mark.parametrize("brute_force", [False, True])
def test_legacy_baker_is_called_without_packing_arguments(brute_force, capsys):
    from scripts.trellis_export_compat import packing_kwargs

    def legacy_bake(*, texture_size):
        return texture_size

    assert legacy_bake(texture_size=1024, **packing_kwargs(legacy_bake, brute_force)) == 1024
    assert ("cannot forward packing options" in capsys.readouterr().out) is brute_force


def test_generator_patch_rerun_migrates_earlier_inline_exports(tmp_path, monkeypatch, capsys):
    target = tmp_path / "generate.py"
    target.write_text(GENERATOR)
    highpoly.patch(target)
    quality.patch(target)
    args = [decode.__file__, "--path", str(target)]
    monkeypatch.setattr(sys, "argv", args)
    assert decode.main() == 0
    helper = target.with_name(decode.HELPER_NAME)
    current = target.read_text()
    legacy = (
        current.replace(highpoly.REPLACEMENT, highpoly.LEGACY_REPLACEMENT)
        .replace(quality.PACK_PREAMBLE, quality.LEGACY_PACK_PREAMBLE)
        .replace(decode.ADDED_DUMP, decode.LEGACY_DUMP)
    )
    target.write_text(legacy)
    helper.unlink()
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    assert decode.main() == 0 and capsys.readouterr().out.splitlines()[-1] == "APPLIED"

    monkeypatch.setattr(sys, "argv", args)
    assert decode.main() == 0
    highpoly.patch(target)
    quality.patch(target)
    assert target.read_text() == current
    assert helper.read_bytes() == decode.HELPER_SOURCE.read_bytes()

    helper.write_text("# stale helper")
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    assert decode.main() == 2
    assert "missing or outdated helper" in capsys.readouterr().out


def test_checkpoint_patch_keeps_legacy_check_and_updates_old_install(tmp_path, monkeypatch):
    target = tmp_path / "postprocess.py"
    calls = "".join(anchor + addition for anchor, addition in checkpoints.CALLS)
    target.write_text(checkpoints.LEGACY_HELPER + checkpoints.HELPER_ANCHOR + calls)
    args = [checkpoints.__file__, "--target", str(target)]
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    assert checkpoints.main() == 0
    monkeypatch.setattr(sys, "argv", args)
    assert checkpoints.main() == 0
    new = target.read_text()
    assert checkpoints.LEGACY_HELPER not in new and new.count(checkpoints.MARKER) == 1
    assert new.endswith(calls)
    assert target.with_name(checkpoints.HELPER_NAME).read_bytes() == checkpoints.HELPER_SOURCE.read_bytes()


def test_export_helper_saves_geometry_from_backend_mesh(tmp_path, monkeypatch, mesh):
    torch = pytest.importorskip("torch")
    from scripts.trellis_export_compat import save_checkpoint

    reads = []

    def read():
        reads.append(True)
        return mesh.vertices + 5, mesh.faces

    monkeypatch.setenv("I2L_CHECKPOINT_PREFIX", str(tmp_path / "probe"))
    save_checkpoint("placed", mesh=SimpleNamespace(read=read))
    checkpoint = torch.load(tmp_path / "probe_placed.pt", weights_only=True)
    assert reads == [True]
    torch.testing.assert_close(checkpoint["vertices"], mesh.vertices + 5)


@pytest.mark.parametrize(
    ("edited", "message"),
    [
        (lambda s: s.replace("        choices=[512, 1024, 2048]", "            choices=[512, 1024, 2048]"),
         "texture-size option not in the expected form"),
        (lambda s: s.replace('"--no-texture", action="store_true"', '"--other", action="store_true"'),
         "CLI option anchor"),
        (lambda s: s.replace("decimation_target=target_faces", "target=target_faces"),
         "expected to_glb call"),
        (lambda s: s.replace("glb = o_voxel.postprocess.to_glb(", "glb = renamed("),
         "expected to_glb call"),
    ],
)
def test_quality_patch_refuses_unrecognized_upstream_without_writing(tmp_path, edited, message):
    target = tmp_path / "generate.py"
    source = edited(GENERATOR)
    target.write_text(source)
    with pytest.raises(RuntimeError, match=message):
        quality.patch(target)
    assert target.read_text() == source
    assert not target.with_name(quality.HELPER_NAME).exists()


def test_highpoly_patch_refuses_moved_mesh_report_and_reconciles_existing_helper(tmp_path):
    target = tmp_path / "generate.py"
    source = GENERATOR.replace(" triangles", " faces")
    target.write_text(source)
    with pytest.raises(RuntimeError, match="mesh-report line"):
        highpoly.patch(target)
    assert target.read_text() == source

    target.write_text(GENERATOR)
    highpoly.patch(target)
    result = target.read_bytes()
    helper = target.with_name(highpoly.HELPER_NAME)
    helper.write_text("# stale helper")
    highpoly.patch(target)
    assert target.read_bytes() == result
    assert helper.read_bytes() == highpoly.HELPER_SOURCE.read_bytes()


def test_checkpoint_patch_repairs_damaged_helper_on_current_source(tmp_path, monkeypatch, capsys):
    target = tmp_path / "postprocess.py"
    target.write_text(checkpoints.HELPER_ANCHOR
                      + "".join(anchor for anchor, _ in checkpoints.CALLS))
    args = [checkpoints.__file__, "--target", str(target)]
    monkeypatch.setattr(sys, "argv", args)
    assert checkpoints.main() == 0
    source = target.read_bytes()
    helper = target.with_name(checkpoints.HELPER_NAME)
    helper.unlink()
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    assert checkpoints.main() == 2
    assert f"ABSENT {target}" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", args)
    assert checkpoints.main() == 0 and target.read_bytes() == source
    assert helper.read_bytes() == checkpoints.HELPER_SOURCE.read_bytes()


def test_export_patch_commands_use_configured_generator(tmp_path, monkeypatch, capsys):
    target = tmp_path / "generate.py"
    target.write_text(GENERATOR)
    monkeypatch.setattr(highpoly, "GENERATOR", target)
    monkeypatch.setattr(quality, "GENERATOR", target)
    highpoly.main()
    quality.main()
    output = capsys.readouterr().out
    assert "--dump-highpoly" in output and "unrestricted texture size" in output
    assert "write_highpoly(args.dump_highpoly" in target.read_text()
    assert "packing_kwargs(o_voxel.postprocess.to_glb" in target.read_text()


def test_quality_patch_requires_packing_anchor_even_in_partial_checkout(tmp_path):
    target = tmp_path / "generate.py"
    source = GENERATOR.replace(quality.OPTION_ANCHOR, quality.OPTIONS + quality.OPTION_ANCHOR)
    source = source.replace('"--no-texture"', '"--other"')
    target.write_text(source)
    with pytest.raises(RuntimeError, match="expected CLI option anchor"):
        quality.patch(target)
    assert target.read_text() == source


def test_decode_command_handles_missing_generator_and_repairs_helper(tmp_path, monkeypatch, capsys):
    target = tmp_path / "generate.py"
    monkeypatch.setattr(sys, "argv", [decode.__file__, "--path", str(target)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(decode.__file__, run_name="__main__")
    assert exc.value.code == 2 and "not found:" in capsys.readouterr().err
    generator = GENERATOR.replace("    return parser.parse_args(argv)",
                                  decode.ANCHOR_ARG + "    return parser.parse_args(argv)")
    target.write_text(decode.apply(generator))
    before = target.read_bytes()
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(decode.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert target.read_bytes() == before
    assert target.with_name(decode.HELPER_NAME).read_bytes() == decode.HELPER_SOURCE.read_bytes()


def test_checkpoint_targets_can_share_an_existing_helper(tmp_path, monkeypatch):
    first = tmp_path / "postprocess.py"
    second = tmp_path / "alternate.py"
    original = checkpoints.HELPER_ANCHOR + "".join(anchor for anchor, _ in checkpoints.CALLS)
    first.write_text(original)
    second.write_text(original)
    argv = [checkpoints.__file__, "--target", str(first), "--target", str(second)]
    monkeypatch.setattr(sys, "argv", argv)
    assert checkpoints.main() == 0
    assert first.read_bytes() == second.read_bytes()
    monkeypatch.setattr(sys, "argv", [*argv, "--check"])
    assert checkpoints.main() == 0
