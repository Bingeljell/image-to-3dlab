"""Run the patch commands against temporary vendor trees."""

import runpy
import sys

import patch_trellis_enable_cleanup as cleanup
import patch_trellis_face_cap as cap
import patch_trellis_mlx_attention as mlx
import pytest
from test_patch_trellis_mlx_attention import CONFIG_SOURCE, FULL_ATTN_SOURCE


def cli(module, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__file__, *args])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(module.__file__, run_name="__main__")
    return exit.value.code


def test_cleanup_cli_roundtrip_enables_original_methods_then_restores_stubs(tmp_path, monkeypatch, capsys):
    target = tmp_path / "mesh.py"
    source = "import torch\nclass Mesh:\n" + "".join(
        signature + stub + f"        return {index}\n"
        for index, (signature, stub) in enumerate(cleanup.STUBS, 1)
    )
    target.write_text(source)
    args = ["--path", str(target)]
    assert cli(cleanup, [*args, "--check"], monkeypatch) == 0
    assert "STUBBED" in capsys.readouterr().out
    assert cli(cleanup, args, monkeypatch) == 0
    patched = target.read_text()
    assert patched.count(cleanup.MARKER) == 3
    mesh = runpy.run_path(str(target))["Mesh"]()
    assert (mesh.fill_holes(), mesh.remove_faces(None), mesh.simplify()) == (1, 2, 3)
    assert cli(cleanup, [*args, "--check"], monkeypatch) == 0
    assert "ENABLED" in capsys.readouterr().out
    assert cli(cleanup, args, monkeypatch) == 0
    assert target.read_text() == patched
    assert "no change" in capsys.readouterr().out
    assert cli(cleanup, [*args, "--revert"], monkeypatch) == 0
    assert target.read_text() == source
    assert cli(cleanup, [*args, "--revert"], monkeypatch) == 0
    assert "already stubbed" in capsys.readouterr().out


def test_cleanup_cli_explains_missing_file(tmp_path, monkeypatch, capsys):
    assert cli(cleanup, ["--path", str(tmp_path / "missing.py")], monkeypatch) == 2
    assert "not found" in capsys.readouterr().err


def test_face_cap_cli_retargets_real_patched_module(tmp_path, monkeypatch, capsys):
    target = tmp_path / "generate.py"
    target.write_text("def limit(args, faces_np):\n    " + cap.ORIGINAL + "\n    return target_faces\n")
    args = ["--target", str(target), "--ceiling", "42"]
    assert cli(cap, [*args, "--check"], monkeypatch) == 0
    assert "NOT PATCHED" in capsys.readouterr().out
    assert cli(cap, args, monkeypatch) == 0
    assert "200,000 -> 42" in capsys.readouterr().out
    from types import SimpleNamespace

    limit = runpy.run_path(str(target))["limit"]
    assert limit(SimpleNamespace(bake_target_faces=60), [0] * 100) == 42
    assert limit(SimpleNamespace(bake_target_faces=60), [0] * 10) == 10
    assert cli(cap, [*args, "--check"], monkeypatch) == 0
    assert "PATCHED (cap now 42)" in capsys.readouterr().out
    assert cli(cap, args, monkeypatch) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert cli(cap, ["--target", str(target), "--ceiling", "62"], monkeypatch) == 0
    assert runpy.run_path(str(target))["limit"](SimpleNamespace(bake_target_faces=100), [0] * 100) == 62


def test_face_cap_cli_explains_missing_checkout(tmp_path, monkeypatch):
    assert "does not exist" in cli(cap, ["--target", str(tmp_path / "missing")], monkeypatch)


def test_mlx_cli_reports_changes_and_applies_without_loading_vendor(tmp_path, monkeypatch, capsys):
    paths = {}
    for (relative, _), source in zip(mlx.TARGETS, (CONFIG_SOURCE, FULL_ATTN_SOURCE), strict=True):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        paths[target] = source
    args = ["--root", str(tmp_path)]
    assert cli(mlx, [*args, "--check"], monkeypatch) == 0
    assert "2 file(s) would change" in capsys.readouterr().out
    assert {path: path.read_text() for path in paths} == paths
    assert cli(mlx, args, monkeypatch) == 0
    assert "2 file(s) changed" in capsys.readouterr().out
    patched = {path: path.read_text() for path in paths}
    assert all("'mlx'" in text for text in patched.values())
    assert cli(mlx, args, monkeypatch) == 0
    assert "0 file(s) changed" in capsys.readouterr().out
    assert {path: path.read_text() for path in paths} == patched
    target = next(iter(paths))
    target.unlink()
    assert str(target) in cli(mlx, args, monkeypatch)


def test_rocm_cli_check_reports_absent_extension_without_modifying_checkout(tmp_path, monkeypatch, capsys):
    import patch_trellis_rocm as rocm

    assert cli(rocm, ["--root", str(tmp_path), "--check"], monkeypatch) == 1
    message = capsys.readouterr().out
    assert "MISSING" in message and "not all patches applied" in message
    assert list(tmp_path.iterdir()) == []


def test_pixal_shape_patch_preserves_existing_cstdlib_and_refuses_missing_include_anchor():
    import patch_pixal3d_shape_reuse as shape
    from test_patch_pixal3d_shape_reuse import ORIGINAL

    source = shape.INCLUDE_ADDITION + ORIGINAL
    assert shape.apply(source).count("#include <cstdlib>") == 1
    with pytest.raises(SystemExit, match="include anchor not found"):
        shape.apply(ORIGINAL.replace(shape.INCLUDE_ANCHOR, ""))


def test_sf3d_metal_baker_patch_cli_reports_and_reapplies_on_new_checkout(tmp_path, monkeypatch, capsys):
    import patch_sf3d_mps_queue as mps

    target = tmp_path / mps.BAKER_FILE
    target.parent.mkdir(parents=True)
    args = ["--root", str(tmp_path)]
    assert cli(mps, [*args, "--check"], monkeypatch) == 1
    assert "not patched" in capsys.readouterr().out
    assert "is missing" in cli(mps, args, monkeypatch)
    target.write_text(mps.NEEDLE * mps.EXPECTED)
    assert cli(mps, args, monkeypatch) == 0
    assert mps.PROOF in target.read_text()
    assert "patched the Metal baker" in capsys.readouterr().out
    before = target.read_bytes()
    assert cli(mps, args, monkeypatch) == 0
    assert target.read_bytes() == before
    assert "already patched" in capsys.readouterr().out


def test_sf3d_linux_patch_cli_detects_missing_and_applied_files(tmp_path, monkeypatch, capsys):
    import patch_sf3d_linux as linux

    args = ["--root", str(tmp_path)]
    assert cli(linux, [*args, "--check"], monkeypatch) == 1
    assert "not patched" in capsys.readouterr().out
    for spec in linux.PATCHES:
        file = tmp_path / spec.file
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a") as handle:
            handle.write(spec.needle * spec.expected)
    assert cli(linux, args, monkeypatch) == 0
    assert cli(linux, [*args, "--check"], monkeypatch) == 0
    assert "patched" in capsys.readouterr().out
    before = {spec.file: (tmp_path / spec.file).read_bytes() for spec in linux.PATCHES}
    assert cli(linux, args, monkeypatch) == 0
    assert "already patched" in capsys.readouterr().out
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())


def test_cleanup_check_refuses_changed_mesh_signature(tmp_path, monkeypatch):
    target = tmp_path / "mesh.py"
    target.write_text("class NewMesh: pass\n")
    monkeypatch.setattr(sys, "argv", ["cleanup", "--path", str(target), "--check"])
    with pytest.raises(RuntimeError, match="anchor missing from the port"):
        cleanup.main()
    assert target.read_text() == "class NewMesh: pass\n"


def test_face_cap_refuses_moved_anchor_without_rewriting(tmp_path, monkeypatch):
    target = tmp_path / "generate.py"
    target.write_text("target_faces = limits.pre_simplify\n")
    assert "anchor not found" in cli(cap, ["--target", str(target)], monkeypatch)
    assert target.read_text() == "target_faces = limits.pre_simplify\n"
