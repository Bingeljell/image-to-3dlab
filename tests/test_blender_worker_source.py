"""Check the payload sent to a Blender process that cannot see the checkout."""

import ast
import json
from pathlib import Path

import pytest

from scripts.blender_worker_source import build_worker_code


def test_worker_payload_carries_source_and_arguments_without_local_paths(tmp_path):
    worker = tmp_path / "worker 'é'.py"
    source = b"# coding: latin-1\r\ndef run(arguments):\r\n    return arguments, 'caf\xe9'\r\n"
    worker.write_bytes(source)
    arguments = {"path": "C:\\art\\café.blend", "name": "'\n); raise RuntimeError('injected')", "flag": True}

    payload = build_worker_code(worker, arguments)

    compile(payload, "<Blender request>", "exec")
    tree = ast.parse(payload)
    compiler, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile"]
    assert ast.literal_eval(compiler.args[0]) == source
    assert ast.literal_eval(compiler.args[1]) == worker.name
    assert {item.arg: ast.literal_eval(item.value) for item in compiler.keywords} == {"dont_inherit": True}
    assert str(tmp_path) not in payload

    invocation, = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                   and ast.literal_eval(node.func.slice) == "run"]
    executor, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exec"]
    assert invocation.func.value.id == executor.args[1].id
    assert executor.args[0] is compiler
    decode, = invocation.args
    assert (decode.func.value.id, decode.func.attr) == ("json", "loads")
    assert json.loads(ast.literal_eval(decode.args[0])) == arguments


def test_worker_payload_reports_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_worker_code(tmp_path / "removed.py", {})


def test_worker_payload_publishes_result_without_worker_globals(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "scene_name = 'turntable'\n"
        "def run(arguments):\n"
        "    return {'scene': scene_name, 'frames': arguments['frames']}\n",
        encoding="utf-8",
    )
    payload = build_worker_code(worker, {"frames": 4})
    bridge_namespace = {"scene_name": "original scene"}

    exec(payload, bridge_namespace)

    assert bridge_namespace["result"] == {"scene": "turntable", "frames": 4}
    assert bridge_namespace["scene_name"] == "original scene"
    assert "run" not in bridge_namespace


def test_worker_payload_rejects_nonfinite_arguments():
    worker = Path(__file__)
    with pytest.raises(ValueError, match="Out of range float"):
        build_worker_code(worker, {"scale": float("nan")})
