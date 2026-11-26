"""Exercise patch commands against temporary upstream fixtures."""

import importlib
import runpy
import sys

import pytest


@pytest.mark.parametrize(
    ("name", "has_check"),
    [
        ("dump_continue", True),
        ("header_read", True),
        ("low_vram", True),
        ("model_subset", True),
        ("msvc", False),
        ("rembg", True),
        ("shape_reuse", True),
        ("steps", True),
    ],
)
def test_pixal3d_patch_command(name, has_check, tmp_path, monkeypatch, capsys):
    fixture = importlib.import_module(f"test_patch_pixal3d_{name}")
    target = tmp_path / fixture.patch.DEFAULT_TARGET.name
    script = fixture.SCRIPT

    def run(*flags):
        monkeypatch.setattr(sys, "argv", [str(script), str(target), *flags])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(script), run_name="__main__")
        return exc.value.code

    assert "not found:" in run()
    target.write_text(fixture.ORIGINAL, encoding="utf-8")
    if has_check:
        assert run("--check") == 1
        assert "NOT PATCHED" in capsys.readouterr().out
        assert target.read_text(encoding="utf-8") == fixture.ORIGINAL
    assert run() == 0
    patched = target.read_text(encoding="utf-8")
    assert patched != fixture.ORIGINAL
    assert fixture.patch.is_patched(patched)
    if has_check:
        assert run("--check") == 0
    capsys.readouterr()
    assert run() == 0
    assert "already patched" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8") == patched


def test_low_vram_patch_refreshes_existing_inline_hook(tmp_path, monkeypatch):
    from test_patch_pixal3d_low_vram import ORIGINAL, patch

    target = tmp_path / "generate_mps.py"
    target.write_text(ORIGINAL.replace(patch.ANCHOR, patch.LEGACY), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [patch.__file__, str(target)])
    assert patch.main() == 0
    assert "low_vram_for_run(args)" in target.read_text(encoding="utf-8")
    assert patch.LEGACY not in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["opaque_material", "remesh_checkpoints", "skip_spurious_fill"])
def test_ovoxel_patch_tracks_each_target(name, tmp_path, monkeypatch, capsys):
    patch = importlib.import_module(f"scripts.patch_ovoxel_{name}")
    if name == "opaque_material":
        original = patch.OLD_ALPHA + "        doubleSided=True,\n"
        expected = ["alpha_mode = 'OPAQUE'", "doubleSided=False"]
    elif name == "remesh_checkpoints":
        original = patch.HELPER_ANCHOR + "".join(anchor for anchor, _ in patch.CALLS)
        expected = [patch.MARKER, '("after_uv", vertices=out_vertices']
    else:
        original = patch.ANCHOR + patch.CHECKPOINT_ANCHOR
        expected = ["remove_duplicate_faces()", '("after_simplify_cleanup", mesh=mesh)']

    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.patch("changed upstream")

    target = tmp_path / "postprocess.py"
    target.write_text(original)
    missing = tmp_path / "missing.py"

    def run(*flags):
        monkeypatch.setattr(sys, "argv", [patch.__file__, "--target", str(target), *flags])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(patch.__file__, run_name="__main__")
        return exc.value.code, capsys.readouterr().out.splitlines()

    assert run("--check", "--target", str(missing)) == (
        2, [f"ABSENT {target}", f"MISSING {missing}"]
    )
    assert target.read_text() == original
    assert run() == (0, [f"PATCHED {target}"])
    patched = target.read_text()
    assert all(fragment in patched for fragment in expected)
    assert run("--check") == (0, [f"APPLIED {target}"])
    assert run() == (0, [f"APPLIED {target}"])
    assert target.read_text() == patched


def test_skip_fill_supports_diagnostic_variant():
    from scripts import patch_ovoxel_skip_spurious_fill as patch

    original = patch.ANCHOR + patch.DIAG_ANCHOR
    patched = patch.patch(original)
    assert patch.DIAG_ANCHOR + patch.POST_CLEANUP in patched
    assert "mesh.fill_holes(" not in patched
    assert patch.patch(patched) == patched
    with pytest.raises(RuntimeError, match="post-simplify anchor missing"):
        patch.patch(patch.ANCHOR)


def test_remesh_checkpoints_refuses_missing_call_site():
    from scripts import patch_ovoxel_remesh_checkpoints as patch

    with pytest.raises(RuntimeError, match="postprocess anchor missing"):
        patch.patch(patch.HELPER_ANCHOR)


def test_trellis_background_model_patch_disables_both_loaders(tmp_path, monkeypatch, capsys):
    from scripts import patch_trellis_no_bria as patch

    checkout = tmp_path / "vendor/trellis-mac"
    pipelines = checkout / "TRELLIS.2/trellis2/pipelines"
    pipelines.mkdir(parents=True)
    monkeypatch.setattr(patch, "ROOT", tmp_path)
    monkeypatch.setattr(patch, "PIPELINES", pipelines)
    loaders = [
        pipelines / "trellis2_image_to_3d.py",
        pipelines / "trellis2_texturing.py",
    ]
    constructor = (
        "        pipeline.rembg_model = getattr(rembg, "
        "args['rembg_model']['name'])(**args['rembg_model']['args'])"
    )
    for loader in loaders:
        loader.write_text(constructor)
    generate = checkout / "generate.py"
    generate.write_text(
        '    parser.add_argument(\n        "--no-texture", action="store_true",\n'
        "            target_faces = min(200000, len(faces))\n"
        "                target_faces = min(200000, len(faces_np))\n"
    )
    patch.main()
    assert "BRIA will not load" in capsys.readouterr().out
    for loader in loaders:
        text = loader.read_text()
        assert "pipeline.rembg_model = None" in text
        assert "getattr(rembg" not in text
    patched = generate.read_text()
    assert '"--bake-target-faces", type=int, default=50000' in patched
    assert "min(args.bake_target_faces, len(faces))" in patched
    assert "min(args.bake_target_faces, 200000, len(faces_np))" in patched
    patch.main()
    assert generate.read_text() == patched
    assert all(loader.read_text().count("pipeline.rembg_model = None") == 1 for loader in loaders)


@pytest.mark.parametrize(
    ("fixture", "error"),
    [
        ("", "CLI option anchor"),
        ('"--bake-target-faces"', "bake budget hook"),
        (
            '"--bake-target-faces"\n            target_faces = min(200000, len(faces))',
            "Metal bake budget hook",
        ),
    ],
)
def test_trellis_bake_budget_refuses_changed_upstream(tmp_path, fixture, error):
    from scripts import patch_trellis_no_bria as patch

    target = tmp_path / "generate.py"
    target.write_text(fixture)
    with pytest.raises(RuntimeError, match=error):
        patch.patch_cpu_bake_budget(target)
    assert target.read_text() == fixture


def test_trellis_background_model_patch_refuses_changed_upstream(tmp_path):
    from scripts import patch_trellis_no_bria as patch

    target = tmp_path / "pipeline.py"
    target.write_text("pipeline.rembg_model = changed_constructor()")
    with pytest.raises(RuntimeError, match="background-model hook"):
        patch.patch(target)
    assert target.read_text() == "pipeline.rembg_model = changed_constructor()"


def test_metal_hashmap_guards_both_lookup_widths_and_reverts(tmp_path, monkeypatch, capsys):
    from scripts import patch_metal_hashmap_miss as patch

    target = tmp_path / "remesh.metal"
    prefix = "// unrelated kernel\n"
    original = prefix + "".join(
        f"{signature}...) {{\n{body}}}\n" for signature, body, _ in patch.PATCHES
    )

    def run(*flags):
        monkeypatch.setattr(sys, "argv", [patch.__file__, "--path", str(target), *flags])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(patch.__file__, run_name="__main__")
        return exc.value.code, capsys.readouterr()

    code, captured = run("--check")
    assert code == 2
    assert "not found" in captured.err
    target.write_text(original)
    code, captured = run("--check")
    assert code == 0 and captured.out == "ABSENT\n"
    assert target.read_text() == original
    code, captured = run()
    assert code == 0
    assert "now rebuild" in captured.out
    patched = target.read_text()
    assert patched.startswith(prefix)
    guard = "if (idx == 0xFFFFFFFFu) return 1.0e30f;\n    return udf[idx];"
    assert patched.count(guard) == 2
    code, captured = run("--check")
    assert code == 0 and captured.out == "APPLIED\n"
    code, captured = run()
    assert code == 0 and "already applied" in captured.out
    assert target.read_text() == patched
    code, captured = run("--revert")
    assert code == 0 and "reverted" in captured.out
    assert target.read_text() == original
    assert run("--revert")[0] == 0
    assert target.read_text() == original


def test_metal_hashmap_refuses_unknown_or_half_patched_shader():
    from scripts import patch_metal_hashmap_miss as patch

    first, second = patch.PATCHES
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.state("")
    with pytest.raises(RuntimeError, match="does not match"):
        patch.state(first[0] + second[0])
    with pytest.raises(RuntimeError, match="half-patched"):
        patch.state(first[0] + first[2] + second[0] + second[1])


def test_mtlbvh_patch_updates_shader_and_native_source_together(tmp_path, monkeypatch, capsys):
    from scripts import patch_mtlbvh_production_traversal as patch

    metal = tmp_path / "bvh.metal"
    native = tmp_path / "metal_bvh.mm"
    monkeypatch.setattr(patch, "METAL", metal)
    monkeypatch.setattr(patch, "NATIVE", native)
    original = (
        "// ─── Fixed-size stack ────────────────────────────────────────────────────────\n"
        "struct FixedStack24 {\n"
        "    int elems[24];\n"
        "        if (count < 24)\n"
        "// ─── BVH traversal ──────────────────────────────────────────────────────────\n\n"
        "// Stack-based closest triangle (with sorting network)\n"
        "FixedStack24 stack;\n"
        + patch.UNSIGNED_OLD
    )

    def run(*flags):
        monkeypatch.setattr(sys, "argv", [patch.__file__, *flags])
        return patch.main()

    assert run("--check") == 2
    metal.write_text(original)
    assert run("--check") == 2
    native.write_text(patch.CPU_OLD)
    capsys.readouterr()
    assert run("--check") == 0
    assert capsys.readouterr().out == "ABSENT\n"
    assert run() == 0
    after = metal.read_text()
    assert "int elems[48]" in after
    assert "node.escape_idx" in after
    assert patch.UNSIGNED_NEW in after
    assert "@autoreleasepool" in native.read_text()
    capsys.readouterr()
    assert run("--check") == 0
    assert capsys.readouterr().out == "APPLIED\n"
    assert run() == 0
    assert metal.read_text() == after
    assert run("--revert") == 0
    assert metal.read_text() == original
    assert native.read_text() == patch.CPU_OLD
    assert run("--revert") == 0

    metal.write_text(after)
    with pytest.raises(RuntimeError, match="partially patched"):
        run()
    assert metal.read_text() == after
    assert native.read_text() == patch.CPU_OLD


def test_mtlbvh_patch_refuses_missing_anchors_and_modified_patch():
    from scripts import patch_mtlbvh_production_traversal as patch

    with pytest.raises(RuntimeError, match="anchors changed"):
        patch.apply_metal(patch.UNSIGNED_OLD)
    with pytest.raises(RuntimeError, match="patch differs"):
        patch.revert_metal("closest_triangle_stackless")
