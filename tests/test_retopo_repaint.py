"""Tests for the retopologise → repaint → compress chain runner.

The whole point of the script is that every asset gets identical treatment, so the things
worth testing are the ones that would silently differ per asset: the stage list, and the
positional argument order handed to `blender_retopo_bake.py`. That script takes nine
positional arguments after `--`, and swapping two of them produces a differently-tuned
asset rather than an error — exactly the kind of failure a comparison would hide.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retopo_repaint.py"


def _load():
    spec = importlib.util.spec_from_file_location("retopo_repaint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load()


def test_the_full_chain_runs_four_stages_in_order():
    assert rr.stage_plan(skip_paint=False, skip_compress=False) == [
        "retopologise", "repaint", "bake", "compress",
    ]


def test_the_bake_comes_after_the_repaint_because_the_repaint_re_unwraps():
    stages = rr.stage_plan(skip_paint=False, skip_compress=False)
    assert stages.index("bake") == stages.index("repaint") + 1


def test_skipping_the_bake_drops_only_the_bake():
    assert rr.stage_plan(skip_paint=False, skip_compress=False, skip_bake=True) == [
        "retopologise", "repaint", "compress",
    ]


def test_skipping_paint_keeps_the_transferred_texture_path():
    assert rr.stage_plan(skip_paint=True, skip_compress=False) == [
        "retopologise", "bake", "compress",
    ]


def test_skipping_compress_leaves_the_asset_uncompressed():
    assert rr.stage_plan(skip_paint=False, skip_compress=True) == [
        "retopologise", "repaint", "bake",
    ]


def test_retopology_is_never_skipped_for_a_game_ready_finish():
    """It is what makes the repaint affordable; only full detail leaves it out."""
    assert rr.stage_plan(skip_paint=True, skip_compress=True, skip_bake=True) == [
        "retopologise"]


def test_full_detail_keeps_the_mesh_and_only_pixel_matches_and_squeezes():
    """Nothing simplified means nothing to repaint or bake onto, whatever else was asked."""
    assert rr.stage_plan(skip_paint=False, skip_compress=False, photo=True, keep_mesh=True) == ["photo", "compress"]
    assert rr.stage_plan(skip_paint=True, skip_compress=False, keep_mesh=True) == ["compress"]


def test_full_detail_is_a_flag_on_the_command_line():
    args = rr.build_parser().parse_args(["a.glb", "a.png", "out.glb", "--keep-mesh"])
    assert args.keep_mesh is True
    assert rr.build_parser().parse_args(["a.glb", "a.png", "out.glb"]).keep_mesh is False


def test_the_blender_command_passes_arguments_in_the_documented_order():
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), faces=40000, atlas=2048, angle=89.0,
        voxel=0.004, metallic=0.648, roughness=0.686, ior=1.4,
        blender=Path("/bin/blender"),
    )
    separator = command.index("--")
    positional = command[separator + 1:]
    assert positional == [
        "in.glb", "out.glb", "40000", "2048", "89.0", "0.004", "0.648", "0.686", "1.4",
    ]


def test_the_blender_command_runs_headless_with_the_retopo_script():
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), 40000, 2048, 89.0, 0.004, 0.25, 0.65, 1.45,
    )
    assert "--background" in command
    assert command[command.index("--python") + 1].endswith("blender_retopo_bake.py")
    # Headless, not the live socket: a Cycles bake through the GUI blocks the viewport.
    assert "--python-console" not in command


def test_a_zero_voxel_fraction_is_passed_through_not_dropped():
    """0 means 'skip the remesh' downstream; it must survive as a value."""
    command = rr.retopo_command(
        Path("in.glb"), Path("out.glb"), 40000, 2048, 89.0, 0.0, 0.25, 0.65, 1.45,
    )
    assert command[command.index("--") + 6] == "0.0"


def test_a_stage_is_only_reused_when_a_resume_asks_for_it(tmp_path):
    artifact = tmp_path / "result_retopo.glb"
    artifact.write_bytes(b"mesh")
    assert rr.reuse(artifact, resume=True) is True
    assert rr.reuse(artifact, resume=False) is False


def test_a_missing_or_truncated_artifact_is_never_reused(tmp_path):
    # A zero-byte file is a stage that died mid-write. Reusing it would hand the next
    # stage a truncated GLB instead of re-running the stage that actually failed.
    assert rr.reuse(tmp_path / "absent.glb", resume=True) is False
    empty = tmp_path / "result_painted.glb"
    empty.write_bytes(b"")
    assert rr.reuse(empty, resume=True) is False


def test_resume_is_opt_in_on_the_real_parser():
    parsed = rr.build_parser().parse_args(["a.glb", "a.png", "out.glb"])
    assert parsed.resume is False
    assert rr.build_parser().parse_args(
        ["a.glb", "a.png", "out.glb", "--resume"]
    ).resume is True


def test_the_bake_command_bakes_the_original_onto_the_current_mesh():
    command = rr.bake_command(Path("gen.glb"), Path("painted.glb"), Path("baked.glb"), 2048,
                              blender=Path("/bl"))
    assert command[:6] == ["/bl", "--background", "--python-exit-code", "1", "--python",
                           command[5]]
    assert command[5].endswith("blender_bake_detail.py")
    after = command[command.index("--") + 1:]
    assert after == ["gen.glb", "painted.glb", "baked.glb", "2048"]


@pytest.mark.parametrize("build", [
    lambda: rr.retopo_command(Path("in.glb"), Path("out.glb"), 40000, 2048, 89.0, 0.004,
                              0.25, 0.65, 1.45),
    lambda: rr.bake_command(Path("gen.glb"), Path("painted.glb"), Path("baked.glb"), 2048),
])
def test_a_crash_in_a_blender_stage_fails_the_stage(build):
    """Blender exits 0 when a script raises, unless this comes before ``--python``."""
    command = build()
    flag = command.index("--python-exit-code")
    assert command[flag + 1] == "1"
    assert flag < command.index("--python")


def test_a_stage_line_still_names_the_script_it_runs(capsys, tmp_path):
    """The progress line printed the first four words, which were the script path until
    --python-exit-code 1 took two of them."""
    command = rr.bake_command(Path("gen.glb"), Path("painted.glb"), Path("baked.glb"), 2048,
                              blender=Path("/usr/bin/true"))
    assert rr.shown(command).endswith("blender_bake_detail.py")
    rr._run(command, tmp_path / "bake.log", "bake")
    line = capsys.readouterr().out
    assert line.startswith("[bake] /usr/bin/true --background --python-exit-code 1 --python ")
    assert "blender_bake_detail.py ..." in line
    assert rr.shown(["gltfpack", "-i", "a.glb", "-o", "b.glb", "-cc"]) == "gltfpack -i a.glb -o"


def test_skip_bake_is_on_the_real_parser_and_off_by_default():
    parser = rr.build_parser()
    assert parser.parse_args(["a.glb", "b.png", "c.glb"]).skip_bake is False
    assert parser.parse_args(["a.glb", "b.png", "c.glb", "--skip-bake"]).skip_bake is True


def test_the_photo_stage_runs_after_the_repaint_and_before_the_bake():
    assert rr.stage_plan(skip_paint=False, skip_compress=False, photo=True) == [
        "retopologise", "repaint", "photo", "bake", "compress",
    ]
    assert "photo" not in rr.stage_plan(skip_paint=False, skip_compress=False)


def test_the_photo_stage_follows_retopology_when_there_is_no_repaint():
    assert rr.stage_plan(skip_paint=True, skip_compress=False, photo=True) == [
        "retopologise", "photo", "bake", "compress",
    ]


def test_views_default_to_off():
    assert rr.build_parser().parse_args(["a.glb", "b.png", "c.glb"]).views is None


def test_without_a_steps_folder_in_between_files_keep_their_old_names(tmp_path):
    paths = rr.step_paths(tmp_path / "out.glb")
    assert paths["retopo"] == tmp_path / "out_retopo.glb"
    assert paths["painted"] == tmp_path / "out_painted.glb"
    assert paths["photo_weights"] == tmp_path / "out_photo_weights.png"
    assert paths["bake_log"] == tmp_path / "out_bake.log"


def test_a_steps_folder_numbers_the_in_between_files_by_stage(tmp_path):
    steps = tmp_path / "steps"
    paths = rr.step_paths(tmp_path / "vanguard_5k.glb", steps)
    assert [paths[k].name for k in ("retopo", "painted", "photo", "baked")] == [
        "1_retopo.glb", "2_painted.glb", "3_photo.glb", "4_baked.glb"]
    assert all(path.parent == steps for path in paths.values())


def test_the_steps_flag_is_offered_and_off_by_default():
    assert rr.build_parser().parse_args(["a.glb", "a.png", "b.glb"]).steps_dir is None
    parsed = rr.build_parser().parse_args(["a.glb", "a.png", "b.glb", "--steps-dir", "s"])
    assert parsed.steps_dir == Path("s")


def test_preflight_refuses_a_missing_blender_before_any_work(tmp_path):
    blender, problem = rr.preflight(None, skip_paint=True, platform="nvidia", find=lambda: None)
    assert blender is None and "Blender" in problem


def test_preflight_refuses_the_repaint_off_apple_silicon(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    _, problem = rr.preflight(exe, skip_paint=False, platform="nvidia")
    assert "--skip-paint" in problem


def test_preflight_passes_on_nvidia_without_the_repaint_and_on_a_mac_with_it(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    assert rr.preflight(exe, skip_paint=True, platform="nvidia") == (exe, None)
    assert rr.preflight(exe, skip_paint=False, platform="apple-silicon") == (exe, None)


def test_preflight_finds_blender_when_none_is_given(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    assert rr.preflight(None, skip_paint=True, platform="nvidia", find=lambda: exe) == (exe, None)


def test_a_run_with_no_repaint_and_no_photo_needs_only_the_mesh():
    parser = rr.build_parser()
    alone = parser.parse_args(["a.glb", "missing.png", "out.glb", "--skip-paint"])
    assert rr.required_inputs(alone) == [Path("a.glb")]
    with_views = parser.parse_args(["a.glb", "a.png", "out.glb", "--skip-paint", "--views", "v"])
    assert rr.required_inputs(with_views) == [Path("a.glb"), Path("a.png")]
    repaint = parser.parse_args(["a.glb", "a.png", "out.glb"])
    assert rr.required_inputs(repaint) == [Path("a.glb"), Path("a.png")]


def test_edge_fade_is_on_unless_the_run_turns_it_off():
    assert rr.build_parser().parse_args(["a.glb", "a.png", "out.glb"]).no_edge_fade is False
    assert rr.build_parser().parse_args(["a.glb", "a.png", "out.glb", "--no-edge-fade"]).no_edge_fade is True
    assert rr.photo_settings().inner_edge > 0
    off = rr.photo_settings(edge_fade=False)
    assert off.inner_edge == 0.0 and off.cutout_edge > 0      # only the inner-edge fade goes


def test_nudge_is_on_unless_the_run_turns_it_off():
    assert rr.build_parser().parse_args(["a.glb", "a.png", "out.glb"]).no_nudge is False
    assert rr.build_parser().parse_args(["a.glb", "a.png", "out.glb", "--no-nudge"]).no_nudge is True
    assert rr.photo_settings().align is True
    off = rr.photo_settings(nudge=False)
    assert off.align is False and off.inner_edge > 0          # edge fade untouched
    both = rr.photo_settings(edge_fade=False, nudge=False)
    assert both.align is False and both.inner_edge == 0.0


# --- The Pixal3D route: Pixal3D's own clean-up instead of the classic retopology --------------


@pytest.mark.parametrize("dump, keep_mesh, available, expected", [
    (Path("run.post.bin"), False, True, True),
    (None, False, True, False),                  # no dump: an older model, or another engine
    (Path("run.post.bin"), False, False, False),  # no post-replay: upstream's prebuilt download
    (Path("run.post.bin"), True, True, False),    # full detail keeps the generator's mesh
])
def test_the_pixal3d_route_needs_a_dump_and_post_replay(dump, keep_mesh, available, expected):
    assert rr.finish_route(dump, keep_mesh, available) is expected


def test_the_dump_is_a_required_input_when_given(tmp_path):
    args = rr.build_parser().parse_args(
        ["in.glb", "in.png", "out.glb", "--skip-paint", "--native-dump", "run.post.bin"])
    assert Path("run.post.bin") in rr.required_inputs(args)


def test_without_a_dump_the_inputs_are_unchanged():
    args = rr.build_parser().parse_args(["in.glb", "in.png", "out.glb", "--skip-paint"])
    assert rr.required_inputs(args) == [Path("in.glb")]


def test_a_refused_clean_up_falls_back_to_the_classic_route(tmp_path, monkeypatch, capsys):
    """A frame check that objects must not fail the Finish: it takes the classic route."""
    calls = []

    def refuse(*a, **k):
        raise rr.pixal3d_native.FrameMismatch("the clean-up's model sits 3.0% away")

    monkeypatch.setattr(rr.pixal3d_native, "native_retopo", refuse)
    assert rr.native_or_classic(lambda: calls.append("classic"), tmp_path / "run.post.bin",
                                tmp_path / "raw.glb", tmp_path / "out.glb", 20000, 2048,
                                tmp_path / "log") is None
    assert calls == ["classic"]
    assert "classic route" in capsys.readouterr().out


def test_the_bake_command_passes_the_fade_mask_after_the_size():
    command = rr.bake_command(Path("gen.glb"), Path("painted.glb"), Path("baked.glb"), 2048,
                              blender=Path("/bl"), fade=Path("fade.png"))
    assert command[command.index("--") + 1:] == [
        "gen.glb", "painted.glb", "baked.glb", "2048", "fade.png"]


def test_the_fade_mask_has_its_own_step_file(tmp_path):
    assert rr.step_paths(tmp_path / "out.glb")["photo_fade"] == tmp_path / "out_photo_fade.png"
    assert rr.step_paths(tmp_path / "o.glb", tmp_path / "s")["photo_fade"].name == "3_photo_fade.png"


def test_the_fade_mask_is_the_trust_map_at_the_bake_size_with_soft_edges(tmp_path):
    import numpy as np
    from PIL import Image

    trust = np.zeros((64, 64), dtype=np.uint8)
    trust[:, 32:] = 255  # right half painted from the picture
    Image.fromarray(trust).save(tmp_path / "w.png")
    rr.normal_fade_mask(tmp_path / "w.png", 128, tmp_path / "fade.png")
    mask = np.asarray(Image.open(tmp_path / "fade.png"))
    assert mask.shape == (128, 128)
    assert mask[64, 5] == 0 and mask[64, 122] == 255  # far from the edge: unchanged
    assert 0 < mask[64, 64] < 255                       # at the edge: blended, no seam
