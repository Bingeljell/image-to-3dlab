"""Exercise weight-download decisions using local files and an in-memory transport."""

import io
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from hunyuan_mlx import download_weights
from hunyuan_mlx.shape.scripts import dl_any, dl_modelscope


@pytest.fixture
def hub(monkeypatch, tmp_path):
    calls = []

    def fetch(repo, file, *, local_dir):
        calls.append((repo, file, local_dir))
        target = local_dir / file
        target.write_bytes(b"source")
        return str(target)

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=fetch, snapshot_download=snapshot,
    ))
    monkeypatch.setattr(download_weights, "SHAPE_WEIGHTS", tmp_path / "shape")
    monkeypatch.setattr(download_weights, "PAINT_WEIGHTS", tmp_path / "paint")
    return calls


def test_default_download_selects_only_the_loaders_inputs(hub, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["download_weights.py", "--yes"])
    assert download_weights.main() == 0
    assert "13.3 GB" in capsys.readouterr().out
    config, weights, paint = hub
    assert config[0] == weights[0] == "tencent/Hunyuan3D-2"
    assert config[1] == "hunyuan3d-dit-v2-0/config.yaml"
    assert weights[1] == "hunyuan3d-dit-v2-0/model.fp16.safetensors"
    assert config[2] == weights[2] == download_weights.SHAPE_WEIGHTS / "Hunyuan3D-2"
    assert paint[0] == download_weights.PAINT_HF_REPO
    assert paint[1]["revision"] == download_weights.PAINT_HF_REVISION
    assert paint[1]["allow_patterns"] == list(download_weights.PAINT_FILES)
    assert paint[1]["local_dir"] == download_weights.PAINT_WEIGHTS / "hunyuan3d-paintpbr-v2-1"


@pytest.mark.parametrize("preconverted", [False, True])
def test_all_routes_convert_only_when_needed(hub, monkeypatch, preconverted):
    monkeypatch.setattr(sys, "argv", ["download_weights.py", "--yes", "--all", "--skip-paint"])
    target = download_weights.SHAPE_WEIGHTS / "Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.safetensors"
    if preconverted:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous conversion")
    subprocess_calls = []

    def run(args, **kwargs):
        subprocess_calls.append(args)
        assert args[0] == sys.executable
        if args[1:] == ["-c", "import torch"]:
            assert kwargs == {"capture_output": True, "check": False}
            return subprocess.CompletedProcess(args, 0)
        assert args[1].endswith("/convert_v21_ckpt.py")
        assert args[2] == str(target.with_suffix(".ckpt"))
        assert args[3] == str(target)
        assert kwargs == {"check": True}
        target.write_bytes(b"converted")

    monkeypatch.setattr(subprocess, "run", run)
    assert download_weights.main() == 0
    fetched = [file for _repo, file, _local in hub]
    assert fetched == [
        "hunyuan3d-dit-v2-0/config.yaml", "hunyuan3d-dit-v2-0/model.fp16.safetensors",
        "hunyuan3d-dit-v2-0-turbo/config.yaml", "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors",
        "hunyuan3d-dit-v2-1/config.yaml", "hunyuan3d-dit-v2-1/model.fp16.ckpt",
    ]
    assert len(subprocess_calls) == (1 if preconverted else 2)
    assert not target.with_suffix(".ckpt").exists()
    assert target.read_bytes() == (b"previous conversion" if preconverted else b"converted")


def test_converter_failure_preserves_checkpoint_for_retry(hub, monkeypatch):
    def run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        download_weights.download_shape("2.1")
    source = list(download_weights.SHAPE_WEIGHTS.rglob("*.ckpt"))
    assert len(source) == 1
    assert source[0].read_bytes() == b"source"


def test_cleanup_preserves_source_if_paths_resolve_to_the_same_file(tmp_path):
    source = tmp_path / "checkpoint"
    source.write_bytes(b"data")
    assert not download_weights.discard_converted_checkpoint(source, source)
    assert source.read_bytes() == b"data"


class ModelScopeTransport:
    def __init__(self, responses, files=()):
        self.responses = iter(responses)
        self.files = files
        self.requests = []

    def open(self, request, *, timeout):
        if isinstance(request, str):
            assert request.endswith("Revision=master&Recursive=true")
            assert timeout == 60
            return io.BytesIO(json.dumps({"Data": {"Files": self.files}}).encode())
        self.requests.append(request)
        assert timeout == 120
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _response(body, *, status=None, code=None):
    stream = io.BytesIO(body)
    stream.status = status
    stream.code = code
    return stream


@pytest.fixture(params=[dl_any, dl_modelscope], ids=["generic", "mini"])
def modelscope(request, monkeypatch, tmp_path):
    module = request.param
    out = tmp_path / "nested" / "weights.bin"
    delays = []
    monkeypatch.setattr(module.time, "sleep", delays.append)

    def run(responses):
        transport = ModelScopeTransport(responses)
        if module is dl_any:
            monkeypatch.setattr(module, "_opener", lambda: transport)
            result = module.download("org/repo", "weights.bin", str(out), 7000)
        else:
            monkeypatch.setattr(module, "opener", transport)
            monkeypatch.setattr(module, "ROOT", str(out.parent))
            result = module.download("weights.bin", 7000)
        return result, transport.requests

    return run, out, delays


def test_modelscope_resumes_after_partial_response_and_retry(modelscope):
    run, out, delays = modelscope
    result, requests = run([
        _response(b"first", status=200),
        HTTPError(None, 503, "busy", None, None),
        OSError("disconnected"),
        _response(b"x" * 6995, code=206),
    ])
    assert result
    assert out.read_bytes() == b"first" + b"x" * 6995
    assert [req.get_header("Range") for req in requests] == [None, "bytes=5-", "bytes=5-", "bytes=5-"]
    assert delays == [4, 8]


def test_modelscope_replaces_partial_file_if_server_ignores_range(modelscope):
    run, out, delays = modelscope
    result, requests = run([_response(b"first", status=200), _response(b"x" * 7000, status=200)])
    assert result
    assert out.read_bytes() == b"x" * 7000
    assert requests[-1].get_header("Range") == "bytes=5-"
    assert not delays


def test_modelscope_exhausts_retries_on_incomplete_file(modelscope):
    run, out, delays = modelscope
    result, requests = run([HTTPError(None, 416, "range", None, None)] * 80)
    assert not result
    assert not out.exists()
    assert len(delays) == len(requests) in (60, 80)
    assert max(delays) == 20


def test_modelscope_accepts_file_completed_during_range_request(modelscope):
    run, out, delays = modelscope

    def responses():
        yield _response(b"first", status=200)
        out.write_bytes(b"x" * 7000)
        yield HTTPError(None, 416, "range", None, None)

    result, requests = run(responses())
    assert result
    assert requests[-1].get_header("Range") == "bytes=5-"
    assert not delays


@pytest.mark.parametrize("answer,expected", [("no", 1), (" YES ", 0), (None, 1)])
def test_generic_modelscope_names_size_and_requires_consent(tmp_path, monkeypatch, capsys, answer, expected):
    out = tmp_path / "model"
    transport = ModelScopeTransport(
        [_response(b"x" * 7000, status=200)],
        files=[{"Path": "ignored", "Size": 90}, {"Path": "file", "Size": "7000"}],
    )
    monkeypatch.setattr(dl_any, "_opener", lambda: transport)

    def consent():
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", consent)
    assert dl_any.main(["org/repo", "file", str(out)]) == expected
    assert out.exists() == (expected == 0)
    assert len(transport.requests) == (expected == 0)
    prompt = capsys.readouterr().out
    assert "ModelScope org/repo" in prompt
    assert "GB" in prompt
    assert "South Korea" in prompt


def test_modelscope_reports_missing_file_before_prompting(tmp_path, monkeypatch, capsys):
    transport = ModelScopeTransport([], files=[{"Path": "other", "Size": 9000}])
    monkeypatch.setattr(dl_any, "_opener", lambda: transport)
    assert dl_any.main(["org/repo", "missing", str(tmp_path / "file")]) == 1
    assert "could not find missing" in capsys.readouterr().out
    assert not transport.requests


def test_generic_modelscope_explicit_yes_skips_prompt(tmp_path, monkeypatch):
    out = tmp_path / "model"
    transport = ModelScopeTransport(
        [_response(b"x" * 7000, status=200)], files=[{"Path": "file", "Size": "7000"}],
    )
    monkeypatch.setattr(dl_any, "_opener", lambda: transport)
    monkeypatch.setattr("builtins.input", lambda: pytest.fail("prompted with --yes"))
    assert dl_any.main(["org/repo", "file", str(out), "--yes"]) == 0
    assert out.stat().st_size == 7000


@pytest.mark.parametrize("second_ok", [True, False])
def test_mini_cli_reports_partial_download_and_keeps_first_file(tmp_path, monkeypatch, second_ok):
    monkeypatch.setattr(dl_modelscope, "FILES", [("one.bin", 7000), ("two.bin", 7000)])
    monkeypatch.setattr(dl_modelscope, "ROOT", str(tmp_path))
    monkeypatch.setattr(dl_modelscope.time, "sleep", lambda _seconds: None)
    good = _response(b"y" * 7000, status=200)
    second = [good] if second_ok else [HTTPError(None, 503, "busy", None, None)] * 60
    transport = ModelScopeTransport([_response(b"x" * 7000, status=200), *second])
    monkeypatch.setattr(dl_modelscope, "opener", transport)
    assert dl_modelscope.main(["--yes"]) == (0 if second_ok else 1)
    assert (tmp_path / "one.bin").stat().st_size == 7000
    assert (tmp_path / "two.bin").is_file() == second_ok



def test_shape_only_command_reports_smaller_size_while_waiting_for_consent(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["download_weights.py", "--model", "2.0-turbo", "--skip-paint"])
    monkeypatch.setattr("builtins.input", lambda: "no")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit) as status:
        runpy.run_path(download_weights.__file__, run_name="__main__")
    assert status.value.code == 1
    prompt = capsys.readouterr().out
    assert "2.0-turbo" in prompt and "4.6 GB" in prompt and "Download cancelled" in prompt
    assert "8.7 GB" not in prompt and "13.3 GB" not in prompt


@pytest.mark.parametrize("module", [dl_any, dl_modelscope], ids=["generic", "mini"])
def test_modelscope_transfer_does_not_report_progress_for_each_block(module, tmp_path, capsys):
    path = tmp_path / "model.bin"
    data = b"x" * (1 << 20)
    with _response(data, status=200) as response:
        assert module.write_body(response, str(path), 0) == len(data)
    assert path.read_bytes() == data
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("module", [dl_any, dl_modelscope], ids=["generic", "mini"])
@pytest.mark.parametrize("interactive", [True, False])
def test_modelscope_script_exits_without_starting_transfer(module, monkeypatch, tmp_path, capsys, interactive):
    out = tmp_path / "model.bin"
    transport = ModelScopeTransport([], files=[{"Path": "file", "Size": "7000"}])
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *handlers: transport)
    def decline():
        if interactive:
            return "no"
        raise EOFError

    monkeypatch.setattr("builtins.input", decline)
    monkeypatch.setenv("HY3D_WEIGHTS_ROOT", str(tmp_path))
    args = ["org/repo", "file", str(out)] if module is dl_any else []
    monkeypatch.setattr(sys, "argv", [module.__file__, *args])
    with pytest.raises(SystemExit) as failure:
        runpy.run_path(module.__file__, run_name="__main__")
    assert failure.value.code == 1
    assert not list(tmp_path.iterdir())
    assert not transport.requests
    output = capsys.readouterr().out
    assert "Proceed? [y/N]" in output
    if not interactive:
        assert "Refusing to download without --yes" in output
