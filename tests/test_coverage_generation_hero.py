"""Exercise Hero route restart, provenance and preflight with local assets."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab.provenance import sha256_file
from scripts import hero_route as hero
from tests.test_hero import _skinned_glb


def _command(monkeypatch, picture, out, *args):
    monkeypatch.setattr(sys, "argv", [hero.__file__, str(picture), "--out", str(out), *args])


def _installed(monkeypatch, tmp_path):
    import pixal3d_generate

    binary = tmp_path / "trellis-cli"
    binary.write_bytes(b"test binary " + pixal3d_generate.DUMP_MARKER)
    monkeypatch.setattr(hero, "needed_parts", lambda: {"Pixal3D": binary})
    # Keep the shipped binary check while binding it to this fixture.
    monkeypatch.setattr(pixal3d_generate.dump_supported, "__defaults__", (binary,))


@pytest.mark.parametrize("record_suffix", [".json", ".provenance.json", None])
def test_hero_resume_keeps_outputs_and_records_source_licence(
    record_suffix, tmp_path, monkeypatch, capsys
):
    _installed(monkeypatch, tmp_path)
    picture = tmp_path / "front.png"
    back = tmp_path / "back.png"
    Image.new("RGB", (128, 128), (3, 7, 11)).save(picture)
    Image.new("RGB", (128, 128), (4, 8, 12)).save(back)

    out = tmp_path / "run"
    out.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=1)
    model = mesh.export(file_type="glb")
    for name in ("body", "head_raw", "head_clean", "head", "head_baked", "grafted", "joined", "hero"):
        (out / f"{name}.glb").write_bytes(model)
    joints = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    joints[:, 0] = mesh.vertices[:, 1] > 0.5
    weights = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
    weights[:, 0] = 1
    (out / "measure_rig.glb").write_bytes(_skinned_glb(mesh.vertices, joints, weights))

    cutout = Image.new("RGBA", (128, 128))
    cutout.paste((3, 7, 11, 255), (20, 12, 108, 116))
    for name in ("picture_cutout", "head_crop", "head_crop_cutout"):
        cutout.save(out / f"{name}.png")

    body = tmp_path / "source.glb"
    body.write_bytes(model)
    views = tmp_path / "source.svviews"
    views.mkdir()
    (views / "transforms.json").write_text("{}")
    dump = tmp_path / "source.post.bin"
    dump.write_bytes(b"small test dump")
    args = ["--back", str(back), "--body", str(body), "--body-views", str(views), "--body-dump", str(dump)]
    if record_suffix:
        source_record = body.with_suffix(record_suffix)
        source_record.write_text(json.dumps({
            "license": "test source licence", "components": [{"component": "local sample"}]
        }))
        args.extend(["--body-record", str(source_record)])
    (out / "head_raw.provenance.json").write_text(
        json.dumps({"components": [{"component": "head sample"}]})
    )
    result = tmp_path / "delivery/result.glb"
    args.extend(["--result", str(result)])
    _command(monkeypatch, picture, out, *args)

    shots = ("head_front", "head_three_quarter", "head_side", "head_back", "body_front", "body_three_quarter")
    processes = []

    class CheckProcess:
        stdout = io.StringIO("check complete\n")

        def wait(self):
            return 0

    def launch(argv, **kwargs):
        assert argv[1:] == [str(hero.BLENDER), "check", str(out / "hero.glb"), str(out / "check")]
        assert kwargs["encoding"] == "utf-8"
        processes.append(argv)
        Path(argv[-1]).mkdir()
        for name in shots:
            Image.new("RGB", (4, 3), (10, 20, 30)).save(out / "check" / f"{name}.png")
        return CheckProcess()

    monkeypatch.setattr(hero.subprocess, "Popen", launch)
    assert hero.main() == 0
    assert len(processes) == 1
    assert (out / "picture.png").read_bytes() == picture.read_bytes()
    assert (out / "back.png").read_bytes() == back.read_bytes()
    assert (out / "body_raw.svviews/transforms.json").is_file()
    assert (out / "body_raw.post.bin").read_bytes() == dump.read_bytes()
    assert Image.open(out / "check_sheet.jpg").size == (24, 3)

    record = json.loads(result.with_suffix(".provenance.json").read_text())
    assert record["route"] == "hero"
    assert record["input"]["back"] == str(back)
    assert record["output"] == {"path": str(result), "sha256": sha256_file(result)}
    assert record["parameters"]["head_bones"] == [1]
    component_names = {item["component"] for item in record["components"]}
    assert {"head sample", "Real-ESRGAN x4plus", "SkinTokens"} <= component_names
    if record_suffix:
        assert record["license"] == "test source licence"
        assert "local sample" in component_names
        assert str(out / f"body_raw{record_suffix}") in record["sub_runs"]
    else:
        assert record["license"] is None
        assert any("body_raw.glb has no run record" in note for note in record["unsure"])

    assert hero.main() == 0
    assert len(processes) == 1
    assert "check sheet: kept head_front.png" in capsys.readouterr().out


def test_hero_preflight_rejects_missing_tools_and_missing_body_cameras(tmp_path, monkeypatch, capsys):
    picture = tmp_path / "missing-picture.png"
    out = tmp_path / "run"
    _command(monkeypatch, picture, out)
    monkeypatch.setattr(hero, "needed_parts", lambda: {"Pixal3D": tmp_path / "missing-binary"})
    assert hero.main() == 2
    assert "not installed: Pixal3D" in capsys.readouterr().err
    assert not out.exists()

    _installed(monkeypatch, tmp_path)
    _command(monkeypatch, picture, out, "--body", str(tmp_path / "body.glb"))
    assert hero.main() == 2
    assert "body model needs its cameras" in capsys.readouterr().err
    assert not out.exists()


def test_hero_runs_fresh_steps_and_keeps_head_when_bake_refuses(tmp_path, monkeypatch, capsys):
    import autorig_api
    import retopo_repaint

    from image_to_3dlab import matte, pixal3d_native

    _installed(monkeypatch, tmp_path)
    original = tmp_path / "front.png"
    Image.new("RGB", (160, 160), (60, 30, 90)).save(original)
    out = tmp_path / "new"
    _command(monkeypatch, original, out, "--seed", "8")
    body = trimesh.creation.icosphere(subdivisions=3)
    head = trimesh.creation.icosphere(subdivisions=2)
    for mesh in (head, body):
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=(mesh.vertices[:, :2] + 1) / 2,
            material=trimesh.visual.material.PBRMaterial(
                metallicRoughnessTexture=Image.new("RGB", (8, 8), (0, 100, 200))
            ),
        )
    calls = []

    def cut_out(image):
        assert image.mode == "RGB"
        w, h = image.size
        cut = Image.new("RGBA", image.size)
        cut.paste((40, 50, 60, 255), (w // 8, h // 8, 7 * w // 8, 7 * h // 8))
        calls.append("matte")
        return cut, "test-remover"

    monkeypatch.setattr(matte, "cut_out", cut_out)
    monkeypatch.setattr(autorig_api, "ensure_port_patch", lambda: "test rig port")
    monkeypatch.setattr(autorig_api, "rig_ports", lambda: {"SKINTOKENS_PORT": "42222"})
    monkeypatch.setattr(autorig_api, "venv_python", lambda: hero.PYTHON)
    monkeypatch.setattr(autorig_api, "rig_commands",
                        lambda body_path, rig: [([hero.PYTHON, "test-rig", body_path, rig], out)])
    monkeypatch.setattr(retopo_repaint, "find_blender", lambda: tmp_path / "test-blender")
    monkeypatch.setattr(pixal3d_native.replay_command, "__defaults__", (tmp_path / "test-replay",))

    class Process:
        def __init__(self, code=0):
            self.stdout = io.StringIO("external fixture\n")
            self.returncode = code

        def wait(self):
            return self.returncode

    def launch(argv, cwd, env, **_kwargs):
        if argv[1] == "test-rig":
            assert cwd == out and env["SKINTOKENS_PORT"] == "42222"
            mesh = trimesh.load(argv[-2], force="mesh", process=False)
            joints = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
            joints[:, 0] = mesh.vertices[:, 1] > 0.56
            weights = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
            weights[:, 0] = 1
            Path(argv[-1]).write_bytes(_skinned_glb(mesh.vertices, joints, weights))
            calls.append("rig")
        elif str(hero.GENERATE) in argv:
            assert argv[-2:] == ["--seed", "8"]
            model = Path(argv[4])
            (body if model.stem == "body_raw" else head).export(model)
            model.with_suffix(".post.bin").touch()
            model.with_suffix(".svviews").mkdir()
            model.with_suffix(".provenance.json").write_text('{"license": "test licence"}')
            calls.append(model.stem)
        elif str(hero.FINISH) in argv:
            model = Path(argv[5])
            (body if model.stem == "body" else head).export(model)
            model.with_suffix(".retopo-repaint.json").write_text('{"retopology": {"route": "classic"}}')
            calls.append(model.stem)
        elif str(hero.UPSCALE) in argv:
            with Image.open(argv[2]) as image:
                image.resize((128, 128)).save(argv[3])
            calls.append("upscale")
        elif argv[0] == str(tmp_path / "test-replay"):
            head.export(next(name for name in argv if name.endswith(".replay.glb")))
            calls.append("replay")
        elif argv[0] == str(tmp_path / "test-blender"):
            calls.append("bake refuses")
            return Process(1)
        elif str(hero.BLENDER) in argv and "join" in argv:
            Path(argv[-1]).write_bytes(Path(argv[-2]).read_bytes())
            calls.append("join")
        elif str(hero.COMPRESS) in argv:
            Path(argv[3]).write_bytes(Path(argv[2]).read_bytes())
            calls.append("compress")
        elif str(hero.BLENDER) in argv and "check" in argv:
            target = Path(argv[-1])
            target.mkdir()
            for name in ("head_front", "head_three_quarter", "head_side",
                         "head_back", "body_front", "body_three_quarter"):
                Image.new("RGB", (4, 3)).save(target / f"{name}.png")
            calls.append("check")
        else:
            pytest.fail(f"Unexpected process {argv}")
        return Process()

    monkeypatch.setattr(hero.subprocess, "Popen", launch)

    assert hero.main() == 0
    assert calls == ["matte", "body_raw", "body", "rig", "upscale", "matte", "head_raw",
                     "replay", "head", "bake refuses", "join", "compress", "check"]
    assert (out / "head.glb").read_bytes() == (out / "head_baked.glb").read_bytes()
    assert (out / "graft.json").is_file()
    output = capsys.readouterr().out
    assert "test rig port" in output and "fell back to the classic clean-up" in output
    assert "bump-map bake refused" in output
    assert json.loads((out / "hero.provenance.json").read_text())["license"] == "test licence"
