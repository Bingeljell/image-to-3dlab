"""Verify route lock generation and input classification at external boundaries."""

import hashlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import classify_trellis_input as classifier
from scripts import lock_nvidia_routes as locks


@pytest.mark.parametrize("tag", [None, "cu128"])
def test_compile_lock_calls_uv_and_removes_the_device_specific_wheels(tmp_path, monkeypatch, capsys, tag):
    output = tmp_path / "locks" / "route.txt"
    monkeypatch.setattr(locks, "REPO", tmp_path)

    def compile_process(command, *, check):
        assert check
        assert command[:3] == ["uv", "pip", "compile"]
        assert command[command.index("--python-version") + 1] == "3.11"
        assert command[command.index("--python-platform") + 1] == "x86_64-manylinux_2_28"
        if tag is None:
            assert "--extra-index-url" not in command
        else:
            assert command[command.index("--extra-index-url") + 1].endswith(tag)
        assert Path(command[-3]).read_text() == "pillow\ntorch==2.9.1\n"
        Path(command[-1]).write_text("pillow==12.0\ntorch==2.9.1\n\ncuda-runtime==4\nnumpy==2.1\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", compile_process)
    locks.compile_lock(["pillow"], ["torch==2.9.1"], tag, output)
    assert output.read_text().splitlines()[2:] == ["pillow==12.0", "numpy==2.1"]
    assert f"({tag or 'PyPI'}); PyTorch itself is installed separately." in output.read_text()
    assert "(2 packages)" in capsys.readouterr().out


@pytest.mark.parametrize("routes,expected", [([], ["trellis", "hunyuan"]), (["trellis"], ["trellis"]),
                                              (["hunyuan"], ["hunyuan"])])
def test_lock_cli_selects_requested_routes_and_reads_pinned_requirements(monkeypatch, routes, expected):
    def open_requirements(url):
        assert url.endswith(f"{locks.hunyuan.COMMIT}/requirements.txt")
        return io.BytesIO(b"requests\n--index-url example.invalid\n")

    calls = []

    def compile_route(inputs, pins, tag, output):
        assert all(pin.startswith(("torch==", "torchvision==")) for pin in pins)
        assert tag.startswith("cu")
        calls.append(output.stem)
        if output == locks.hunyuan.PACKAGE_LOCK:
            assert "requests" in inputs
            assert all(not line.startswith("--index-url") for line in inputs)
            assert locks.hunyuan.SETUPTOOLS_PIN in inputs

    monkeypatch.setattr(locks.urllib.request, "urlopen", open_requirements)
    monkeypatch.setattr(locks, "compile_lock", compile_route)
    assert locks.main(routes) == 0
    outputs = {"trellis": locks.trellis.PACKAGE_LOCK.stem, "hunyuan": locks.hunyuan.PACKAGE_LOCK.stem}
    assert calls == [outputs[route] for route in expected]


def test_lock_cli_rejects_unknown_route_before_building(monkeypatch, capsys):
    monkeypatch.setattr(locks, "compile_lock", lambda *args: pytest.fail("compiled invalid route"))
    with pytest.raises(SystemExit) as exc:
        locks.main(["bad-route"])
    assert exc.value.code == 2
    assert "unknown route: bad-route" in capsys.readouterr().err


@pytest.mark.parametrize("allow_download", [False, True])
def test_classifier_cli_passes_pinned_model_contract_and_reports_real_scores(
    monkeypatch, tmp_path, capsys, allow_download,
):
    import torch

    source = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 4), (255, 0, 0, 0)).save(source)
    opts = {"revision": classifier.MODEL_REVISION, "local_files_only": not allow_download}

    class Processor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == classifier.MODEL_ID
            assert kwargs == {**opts, "use_fast": True}
            return cls()

        def __call__(self, *, text, images, return_tensors, padding):
            assert text == [*classifier.FLAT_PROMPTS, *classifier.DIMENSIONAL_PROMPTS]
            assert images.size == (4, 4)
            assert images.getpixel((3, 3)) == classifier.NEUTRAL_BACKGROUND
            assert return_tensors == "pt" and padding
            return {"pixel_values": torch.ones(1)}

    class Model:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == classifier.MODEL_ID and kwargs == opts
            return cls()

        def eval(self):
            return self

        def __call__(self, **inputs):
            assert torch.is_inference_mode_enabled()
            assert inputs["pixel_values"].item() == 1
            return SimpleNamespace(logits_per_image=torch.tensor([[5.] * 4 + [1.] * 4]))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor))
    argv = ["classify_trellis_input.py", str(source)]
    if allow_download:
        argv.append("--allow-download")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as status:
        runpy.run_path(classifier.__file__, run_name="__main__")
    assert status.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_revision"] == classifier.MODEL_REVISION
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["verdict"] == "likely_flat"
    assert report["advisory_only"] is True
    assert report["flat_risk"] == pytest.approx(0.982014)


def test_advisory_score_refuses_missing_prompt_group():
    with pytest.raises(ValueError, match="both prompt groups"):
        classifier.summarize_logits([], [1])
