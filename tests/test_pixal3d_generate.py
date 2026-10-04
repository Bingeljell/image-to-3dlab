"""Tests for the Pixal3D wrapper.

Two things decide whether a run is even comparable to our others: whether the image goes in
pre-matted (which skips BiRefNet and keeps the cutout identical to the TRELLIS runs), and
whether the gauge camera is passed at all — Pixal3D conditions on pixel-aligned features
projected through it, so a missing FOV is not a cosmetic difference.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import glb_turn
from tests.test_glb_turn import make_glb, read

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pixal3d_generate.py"


def _load():
    spec = importlib.util.spec_from_file_location("pixal3d_generate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


px = _load()


def test_a_matted_image_uses_the_single_view_path():
    command = px.build_command(
        Path("fox.png"), Path("out.glb"), res=1024, seed=42, fov=px.DEFAULT_FOV,
        models=Path("/m"), cli=Path("/bin/trellis-cli"), matted=True,
    )
    assert "--sv-image" in command
    assert command[command.index("--sv-image") + 1] == "fox.png"
    assert "--bg-removal" not in command


def test_an_unmatted_image_is_handed_to_birefnet():
    # Corrected 2026-09-22. This asserted `"--sv-image" not in command`, pinning a command
    # line the CLI refuses outright: `--pixal3d-weights` requires `--sv-image` or `--views`.
    # It passed because `has_alpha` never returned False, so the branch it describes had
    # never once been executed. A test can only pin behaviour something actually runs.
    command = px.build_command(
        Path("fox.jpg"), Path("out.glb"), res=1024, seed=42, fov=px.DEFAULT_FOV,
        models=Path("/m"), cli=Path("/bin/trellis-cli"), matted=False,
    )
    assert "--sv-image" in command
    assert command[command.index("--bg-removal") + 1] == "birefnet"


def test_the_gauge_camera_and_weight_family_are_always_passed():
    command = px.build_command(
        Path("fox.png"), Path("out.glb"), 1024, 42, px.DEFAULT_FOV,
        Path("/m"), Path("/bin/trellis-cli"), True,
    )
    assert command[command.index("--fov") + 1] == str(px.DEFAULT_FOV)
    assert command[command.index("--pixal3d-weights") + 1] == "sv"
    assert command[-1] == "out.glb"  # output is positional and last


def test_the_default_fov_is_twenty_degrees():
    import math

    assert math.degrees(px.DEFAULT_FOV) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("[2/6] SS proj conditioning + flow", ("ss", 33)),
        ("[3/6] shape SLAT flow (LR 512 -> upsample -> HR 1024 cascade)", ("shape", 50)),
        ("[6/6] write out.glb", ("write", 100)),
    ],
)
def test_stage_banners_are_recognised(line, expected):
    stage, percent = px.stage_from_banner(line)
    assert stage == expected[0]
    assert percent == min(99, expected[1])


def test_the_bar_never_reaches_a_hundred_before_the_run_ends():
    _stage, percent = px.stage_from_banner("[6/6] write out.glb")
    assert percent == 99


def test_ordinary_output_is_not_a_stage():
    assert px.stage_from_banner("done in 349.9s -> out.glb") is None
    assert px.stage_from_banner("[cond] flow (load + runner + sampler) (77.6s)") is None
    assert px.stage_from_banner("ggml_metal_library_compile_pipeline: loaded kernel") is None


def test_stage_zero_is_not_a_stage():
    """`[0/6]` is the banner echoing its input, before any work happens."""
    assert px.stage_from_banner("[0/6] Pixal3D single view: fox.png") is None


def test_readiness_reports_what_is_missing(tmp_path):
    state = px.readiness(cli=tmp_path / "absent", models=tmp_path / "nope")
    assert state["ready"] is False
    assert state["cli_built"] is False
    assert state["weights_present"] == 0


def test_readiness_needs_the_whole_weight_set(tmp_path):
    cli = tmp_path / "trellis-cli"
    cli.write_text("#!/bin/sh\n")
    models = tmp_path / "models"
    models.mkdir()
    for index in range(8):
        (models / f"part{index}.gguf").write_bytes(b"x")

    assert px.readiness(cli, models)["ready"] is False  # 8 of 9

    (models / "part8.gguf").write_bytes(b"x")
    assert px.readiness(cli, models)["ready"] is True


def test_guidance_strength_is_always_passed_and_defaults_to_ten():
    """The CLI's own default of 7.5 dropped the warrior girl's sword blade entirely.

    Leaving `--gss` off the command line is therefore not a neutral choice, so the wrapper
    states it on every run.
    """
    command = px.build_command(
        Path("fox.png"), Path("out.glb"), 1024, 42, px.DEFAULT_FOV,
        Path("/m"), Path("/bin/trellis-cli"), True,
    )
    assert command[command.index("--gss") + 1] == "10.0"
    assert px.DEFAULT_GSS == 10.0


def test_shape_guidance_is_omitted_unless_asked_for():
    """`--gsh` has no tested value here, so an unset one must leave the runtime default."""
    without = px.build_command(
        Path("fox.png"), Path("out.glb"), 1024, 42, px.DEFAULT_FOV,
        Path("/m"), Path("/bin/trellis-cli"), True,
    )
    assert "--gsh" not in without

    with_gsh = px.build_command(
        Path("fox.png"), Path("out.glb"), 1024, 42, px.DEFAULT_FOV,
        Path("/m"), Path("/bin/trellis-cli"), True, gss=10.0, gsh=3.5,
    )
    assert with_gsh[with_gsh.index("--gsh") + 1] == "3.5"
    assert with_gsh[-1] == "out.glb"  # output stays positional and last


def test_paths_reach_the_cli_absolute(tmp_path, monkeypatch, capsys):
    """`trellis-cli` runs from its own tree, so a relative path resolves against the wrong
    directory and the run dies at once with "can't fopen". Caught for real on three assets.
    """
    image = tmp_path / "gnome.png"
    image.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    recorded = {}

    def fake_popen(command, **kwargs):
        recorded["command"] = command
        raise SystemExit(0)

    monkeypatch.setattr(px, "has_alpha", lambda _: True)
    monkeypatch.setattr(px, "readiness", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(px.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys, "argv", ["pixal3d_generate.py", "gnome.png", "out/gnome.glb"],
    )

    with pytest.raises(SystemExit):
        px.main()

    command = recorded["command"]
    assert command[command.index("--sv-image") + 1] == str(image.resolve())
    assert command[-1] == str((tmp_path / "out" / "gnome.glb").resolve())


# --- An alpha channel is not a cutout ----------------------------------------------------
#
# Found the hard way on 2026-09-22, in the promo reel for the release whose headline is the
# text-to-image step. Qwen-Image through stable-diffusion.cpp writes RGBA, but its alpha is
# noise in the 219-255 range with nothing actually transparent. `has_alpha` tested the file
# mode, so it answered "already matted", BiRefNet was skipped, and Pixal3D reconstructed the
# grey backdrop as geometry -- two enormous white sheets either side of a fox's head.
#
# The mode tells you a fourth channel exists. Only the contents tell you it means anything.


def _image(tmp_path, mode, alpha=None, name="i.png"):
    from PIL import Image
    im = Image.new(mode, (64, 64), (200, 120, 60) if mode == "RGB" else (200, 120, 60, 255))
    if alpha is not None:
        im.putalpha(Image.fromarray(alpha))
    path = tmp_path / name
    im.save(path)
    return path


def test_an_opaque_alpha_channel_does_not_count_as_matted(tmp_path):
    """The actual bug: RGBA, no transparency, so it must still be sent to BiRefNet."""
    import numpy as np
    noise = np.random.default_rng(0).integers(219, 256, (64, 64), dtype=np.uint8)
    assert px.has_alpha(_image(tmp_path, "RGBA", noise)) is False


def test_a_real_cutout_counts_as_matted(tmp_path):
    """A genuine matte leaves large regions fully transparent, corners included."""
    import numpy as np
    a = np.zeros((64, 64), dtype=np.uint8)
    a[16:48, 16:48] = 255            # subject in the middle, background cut away
    assert px.has_alpha(_image(tmp_path, "RGBA", a)) is True


def test_an_image_with_no_alpha_channel_is_not_matted(tmp_path):
    assert px.has_alpha(_image(tmp_path, "RGB")) is False


def test_a_barely_transparent_edge_is_not_mistaken_for_a_matte(tmp_path):
    """Antialiasing or a soft vignette is not a cutout, and must not skip BiRefNet."""
    import numpy as np
    a = np.full((64, 64), 255, dtype=np.uint8)
    a[0, 0] = 0                       # a single transparent pixel
    assert px.has_alpha(_image(tmp_path, "RGBA", a)) is False


def test_an_unmatted_image_still_uses_the_single_view_path():
    """`--pixal3d-weights` only works with `--sv-image`, matted or not.

    The old un-matted branch passed the image positionally, which the CLI rejects with
    "--pixal3d-weights requires --views DIR or --sv-image PATH". Nobody noticed because
    `has_alpha` never returned False until the alpha check was fixed on 2026-09-22 --
    so this branch had been dead and broken at the same time.
    """
    command = px.build_command(Path("i.png"), Path("o.glb"), 1024, 42, 0.349, matted=False)
    assert "--sv-image" in command
    # BiRefNet does the cutting, since we have not done it ourselves.
    assert command[command.index("--bg-removal") + 1] == "birefnet"
    # And the image is never a bare positional, which is what broke it.
    assert command[command.index("--sv-image") + 1] == "i.png"
    assert command[-1] == "o.glb"


def test_a_matted_image_is_not_sent_through_birefnet_again():
    command = px.build_command(Path("i.png"), Path("o.glb"), 1024, 42, 0.349, matted=True)
    assert "--sv-image" in command
    assert "--bg-removal" not in command


# --- Matting is ours to do, not the CLI's ------------------------------------------------
#
# Asking trellis-cli to matte (`--bg-removal birefnet`) was measured on 2026-09-22 and it
# changed nothing: the winged fox came back byte-identical in geometry, 934,330 faces and
# the same extents. Its own docs say "a pre-matted image keeps its alpha", so the junk
# Qwen alpha reads as pre-matted to it exactly as it did to us. We cut the image out
# ourselves and hand over a real matte, so nothing downstream has to guess.


def test_matting_writes_a_real_cutout_beside_the_source(tmp_path, monkeypatch):
    from PIL import Image
    import numpy as np

    source = tmp_path / "fox.png"
    Image.new("RGB", (32, 32), (200, 120, 60)).save(source)

    def fake_remove(image):
        out = image.convert("RGBA")
        a = np.zeros((32, 32), dtype=np.uint8)
        a[8:24, 8:24] = 255
        out.putalpha(Image.fromarray(a))
        return out, "birefnet-general-lite"

    monkeypatch.setattr(px, "_rembg_remove", fake_remove)
    matted, model = px.matte(source, tmp_path / "cut.png")
    assert model == "birefnet-general-lite"
    assert matted.is_file()
    # And the result must satisfy our own matte test, or we have solved nothing.
    assert px.has_alpha(matted) is True


def test_the_matte_is_named_so_it_is_obvious_it_is_not_the_original(tmp_path):
    assert "matted" in px.matte_path(tmp_path / "fox.png").name


def test_the_matted_path_reaches_the_cli_absolute(tmp_path, monkeypatch):
    """Matting must not undo the absolute-path rule: the cutout is what the CLI opens.

    The first version mattted before resolving, so an image given relatively reached
    trellis-cli relative and would have died with "can't fopen" -- the exact failure the
    resolve exists to prevent, reintroduced through a new branch.
    """
    image = tmp_path / "gnome.png"
    image.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    recorded = {}

    def fake_popen(command, **kwargs):
        recorded["command"] = command
        raise SystemExit(0)

    monkeypatch.setattr(px, "has_alpha", lambda _: False)          # forces the matte branch
    monkeypatch.setattr(px, "matte", lambda p, d=None: (px.matte_path(p), "u2net"))
    monkeypatch.setattr(px, "readiness", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(px.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", ["pixal3d_generate.py", "gnome.png", "out/gnome.glb"])
    with pytest.raises(SystemExit):
        px.main()

    passed = recorded["command"][recorded["command"].index("--sv-image") + 1]
    assert Path(passed).is_absolute(), passed
    assert "matted" in passed


# --- Provenance: every run leaves a record beside its GLB ---------------------------------
#
# Reported in #50: this route wrote no record at all, from the CLI or the viewer (which runs
# this same script), unlike the Hunyuan route and pipeline.py. The record sits at
# <output>.json, where the viewer already looks for a job's manifest.


def test_the_manifest_carries_the_licence_from_the_catalogue(tmp_path):
    from viewer.backend_catalog import resolve

    image = tmp_path / "orc.png"
    image.write_bytes(b"orc")
    output = tmp_path / "orc.glb"
    output.write_bytes(b"glb")
    record = px.manifest(image, output, res=1024, seed=42, fov=0.35, gss=10.0, gsh=None,
                         matted=True, matted_here=False, seconds=12.34)
    catalogue = resolve("pixal3d")
    assert record["backend"] == "pixal3d"
    assert record["license"] == {"name": catalogue.license_name, "url": catalogue.license_url}
    assert record["parameters"] == {"res": 1024, "seed": 42, "fov": 0.35, "gss": 10.0,
                                    "gsh": None, "matted": True, "steps": 12}
    assert len(record["input"]["sha256"]) == 64
    assert len(record["output"]["sha256"]) == 64
    assert record["timings_seconds"]["total"] == 12.3


def test_background_removal_is_named_only_when_this_run_did_it(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"a")
    output = tmp_path / "a.glb"
    output.write_bytes(b"b")
    ran = px.manifest(image, output, res=1024, seed=1, fov=0.35, gss=10.0, gsh=None,
                      matted=True, matted_here=True, seconds=1.0,
                      matte_model="birefnet-general-lite")
    given = px.manifest(image, output, res=1024, seed=1, fov=0.35, gss=10.0, gsh=None,
                        matted=True, matted_here=False, seconds=1.0)
    assert [c["component"] for c in ran["components"]] == ["rembg/birefnet-general-lite"]
    assert ran["components"][0]["license"] == "MIT (BiRefNet)"
    assert given["components"] == []


def test_a_successful_run_writes_its_manifest_beside_the_glb(tmp_path, monkeypatch):
    image = tmp_path / "gnome.png"
    image.write_bytes(b"png")
    output = tmp_path / "out" / "gnome.glb"

    views = output.with_suffix(".svviews")

    class FakeProcess:
        stdout = iter(["[1/3] structure\n"])

        def wait(self):
            # What trellis-cli leaves: a GLB facing -Z, and the camera it staged.
            output.write_bytes(make_glb([[0.1, 0.2, 0.3]]))
            views.mkdir()
            (views / "transforms.json").write_text(json.dumps(
                {"frames": [{"file_path": "input.png", "transform_matrix": np.eye(4).tolist()}]}))
            return 0

    monkeypatch.setattr(px, "has_alpha", lambda _: True)
    monkeypatch.setattr(px, "readiness", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(px.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(sys, "argv", ["pixal3d_generate.py", str(image), str(output)])

    assert px.main() == 0
    record = __import__("json").loads((tmp_path / "out" / "gnome.json").read_text())
    assert record["backend"] == "pixal3d"
    assert record["output"]["path"] == str(output.resolve())
    # Turned to face front, with its camera, before the record hashes it.
    assert glb_turn.glb_faces_front(output.read_bytes())
    assert np.allclose(read(output.read_bytes(), 0), [[-0.1, 0.2, -0.3]])
    camera = json.loads((views / "transforms.json").read_text())
    assert camera["frames"][0]["transform_matrix"][0][0] == -1.0
    assert record["output"]["sha256"] == px.sha256_file(output)


def test_a_run_without_a_saved_camera_still_turns_the_model(tmp_path):
    output = tmp_path / "o.glb"
    output.write_bytes(make_glb([[1.0, 0.0, 0.0]]))
    px.face_front(output)
    assert glb_turn.glb_faces_front(output.read_bytes())


# --- Fewer sampling steps ------------------------------------------------------------------
#
# `--steps` only works on a trellis-cli rebuilt after scripts/patch_pixal3d_steps.py. An
# unpatched build ignores the variable and runs 12 steps, so every check below exists to
# stop a slow run from being recorded as a fast one.


def _flow_source(tmp_path, patched):
    source = tmp_path / "flow_runner.cpp"
    source.write_text("// i2l_steps\n" if patched else "// upstream\n")
    return source


def test_default_steps_need_no_patch(tmp_path):
    missing = tmp_path / "nope.cpp"
    assert px.steps_problem(None, missing, missing) is None
    assert px.steps_problem(12, missing, missing) is None


def test_fewer_steps_on_an_unpatched_source_is_refused(tmp_path):
    source = _flow_source(tmp_path, patched=False)
    cli = tmp_path / "trellis-cli"
    cli.write_text("")
    assert "not patched" in px.steps_problem(8, source, cli)


def test_a_binary_older_than_the_patch_is_refused(tmp_path):
    import os
    source = _flow_source(tmp_path, patched=True)
    cli = tmp_path / "trellis-cli"
    cli.write_text("")
    os.utime(cli, (1, 1))
    assert "rebuild" in px.steps_problem(8, source, cli)


def test_a_rebuilt_patched_binary_is_accepted(tmp_path):
    import os
    source = _flow_source(tmp_path, patched=True)
    os.utime(source, (1, 1))
    cli = tmp_path / "trellis-cli"
    cli.write_text("")
    assert px.steps_problem(8, source, cli) is None


def test_out_of_range_steps_are_refused(tmp_path):
    assert "1 to 50" in px.steps_problem(0, tmp_path / "x", tmp_path / "y")


def test_the_env_carries_steps_only_when_asked():
    assert px.run_env(8, {"PATH": "/bin"}) == {"PATH": "/bin", "PIXAL3D_STEPS": "8"}
    assert px.run_env(None, {"PATH": "/bin"}) == {"PATH": "/bin"}


def test_a_stray_steps_variable_in_the_shell_is_dropped():
    """Otherwise a run labelled 12 steps in its manifest could quietly be an 8-step one."""
    assert px.run_env(None, {"PIXAL3D_STEPS": "4"}) == {}


def test_a_flow_starting_without_the_override_stops_the_run():
    progress = "      [flow] [....................]  0/12    0.0s  starting"
    assert px.steps_not_applied(progress, 8, seen_override=False)
    assert not px.steps_not_applied(progress, 8, seen_override=True)
    assert not px.steps_not_applied(progress, None, seen_override=False)
    assert not px.steps_not_applied("[2/6] SS proj conditioning", 8, seen_override=False)


def test_the_manifest_records_the_steps(tmp_path):
    image = tmp_path / "i.png"
    output = tmp_path / "o.glb"
    image.write_bytes(b"i")
    output.write_bytes(b"o")
    record = px.manifest(image, output, res=1024, seed=42, fov=0.349, gss=10.0, gsh=None,
                         matted=True, matted_here=False, seconds=1.0, steps=8)
    assert record["parameters"]["steps"] == 8


def _built(tmp_path, patched):
    import os
    source = _flow_source(tmp_path, patched=patched)
    os.utime(source, (1, 1))
    cli = tmp_path / "trellis-cli"
    cli.write_text("")
    return source, cli


def test_auto_runs_fast_steps_on_a_patched_build(tmp_path):
    source, cli = _built(tmp_path, patched=True)
    steps, note = px.resolve_steps(None, source, cli)
    assert steps == px.FAST_STEPS == 8
    assert "auto" in note


def test_auto_falls_back_to_twelve_on_an_unpatched_build(tmp_path):
    """Every install that predates the patch lands here; it must run, not refuse."""
    source, cli = _built(tmp_path, patched=False)
    steps, note = px.resolve_steps(None, source, cli)
    assert steps == 12
    assert "patch_pixal3d_steps.py" in note


def test_an_explicit_count_the_build_cannot_honour_is_refused(tmp_path):
    source, cli = _built(tmp_path, patched=False)
    with pytest.raises(SystemExit, match="not patched"):
        px.resolve_steps(8, source, cli)
    assert px.resolve_steps(12, source, cli)[0] == 12


def test_the_manifest_records_the_licence_class_props_inherit(tmp_path):
    """The Props tab copies output.classification; without it every prop said null."""
    from image_to_3dlab.provenance import LICENSES
    from viewer.backend_catalog import resolve

    image = tmp_path / "i.png"
    output = tmp_path / "o.glb"
    image.write_bytes(b"i")
    output.write_bytes(b"o")
    record = px.manifest(image, output, res=1024, seed=42, fov=0.349, gss=10.0, gsh=None,
                         matted=True, matted_here=False, seconds=1.0)
    profile = LICENSES["pixal3d"]
    assert record["output"]["classification"] == profile.classification == "commercial-conditional"
    catalogue = resolve("pixal3d")
    assert (profile.license_name, profile.license_url) == (catalogue.license_name,
                                                           catalogue.license_url)
