"""Exercise the scene agent's conversation and logging with local substitutes."""

import json
import urllib.error
from io import BytesIO

import blender_agent as agent
import pytest


def tool(name, arguments=None):
    return {
        "id": f"call-{name}",
        "function": {"name": name, "arguments": arguments},
    }


def test_agent_logs_tool_errors_and_images_across_rounds(tmp_path, monkeypatch):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    out = tmp_path / "renders"
    explicit = out / "round01.png"
    auto = out / "round00-auto.png"
    replies = iter(
        [
            {
                "content": "adjust the cube",
                "tool_calls": [
                    tool("move_object", '{"name":"Cube","dx":2}'),
                    tool("describe_scene", "{bad"),
                    tool("absent"),
                ],
            },
            {"tool_calls": [tool("render_view")]},
            {"content": "done"},
        ]
    )
    observed_messages = []
    sent = []

    def chat(api, model, messages, schemas):
        assert api == "http://localhost/api"
        assert model == "local"
        assert any(schema["function"]["name"] == "move_object" for schema in schemas)
        observed_messages.append(list(messages))
        return {"choices": [{"message": next(replies)}]}

    def response(result):
        return json.dumps(
            {"result": {"result": agent.RESULT_SENTINEL + json.dumps(result)}}
        )

    def send(code, host, port):
        assert (host, port) == ("localhost", 8765)
        sent.append(code)
        if len(sent) == 1:
            assert "Cube" in code
            return response({"moved": "Cube"})
        path = auto if len(sent) == 2 else explicit
        assert str(path.resolve()) in code
        path.write_bytes(b"render")
        return response({"rendered": str(path)})

    monkeypatch.setattr(agent, "chat", chat)
    monkeypatch.setattr(agent, "send", send)
    assert (
        agent.main(
            [
                "--task",
                "stage cube",
                "--reference",
                str(reference),
                "--out-dir",
                str(out),
                "--api",
                "http://localhost/api",
                "--model",
                "local",
                "--host",
                "localhost",
                "--port",
                "8765",
                "--resolution",
                "64",
            ]
        )
        == 0
    )
    calls = json.loads((out / "calls.json").read_text())
    assert [c["tool"] for c in calls] == [
        "move_object",
        "describe_scene",
        "absent",
        "render_view",
        "render_view",
    ]
    assert calls[0]["result"] == {"moved": "Cube"}
    assert "valid JSON" in calls[1]["result"]["error"]
    assert "absent" in calls[2]["result"]["error"]
    assert calls[3]["arguments"] == {"forced": True}
    assert calls[4]["arguments"] == {"resolution": 64}
    images = [
        part
        for message in observed_messages[-1]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert len(images) == 3
    transcript = (out / "transcript.json").read_text()
    assert "image_url" not in transcript
    assert transcript.count("<render attached>") == 3


def test_agent_reports_model_connection_failure(tmp_path, monkeypatch, capsys):
    def denied(*args):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(agent, "chat", denied)
    assert agent.main(["--task", "stage", "--out-dir", str(tmp_path)]) == 2
    assert "offline" in capsys.readouterr().err


def test_agent_checks_asset_before_contacting_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        agent.main(
            [
                "--task",
                "stage",
                "--out-dir",
                str(tmp_path),
                "--asset",
                str(tmp_path / "none.glb"),
            ]
        )
    assert exc.value.code == 2
    assert "no such asset" in capsys.readouterr().err


def test_chat_posts_tool_schema_and_messages(monkeypatch):
    messages = [{"role": "user", "content": "stage cube"}]
    tools = [agent.TOOLS["move_object"].schema()]

    def reply(request, *, timeout):
        assert request.full_url == "http://localhost/api/chat/completions"
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert timeout == 2
        assert json.loads(request.data) == {
            "model": "local",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
        }
        return BytesIO(b'{"choices":[]}')

    monkeypatch.setattr(agent.urllib.request, "urlopen", reply)
    assert agent.chat("http://localhost/api/", "local", messages, tools, timeout=2) == {
        "choices": [],
    }


def test_agent_excludes_render_files_that_did_not_arrive(monkeypatch, tmp_path):
    asset = tmp_path / "asset.glb"
    asset.touch()
    replies = iter([
        {"tool_calls": [tool("move_object", '{"name": "Cube", "dx": 1}')]},
        {"tool_calls": [tool("render_view")]},
    ])
    sent = []

    def chat(api, model, messages, schemas):
        assert "asset.glb" in json.dumps(messages)
        return {"choices": [{"message": next(replies)}]}

    def send(code, host, port):
        sent.append(code)
        path = tmp_path / ("round00-auto.png" if len(sent) == 2 else "round01.png")
        response = {"moved": "Cube"} if len(sent) == 1 else {"rendered": str(path)}
        return json.dumps({"result": agent.RESULT_SENTINEL + json.dumps(response)})

    monkeypatch.setattr(agent, "chat", chat)
    monkeypatch.setattr(agent, "send", send)
    assert agent.run("check the model", asset, "local", "unused", tmp_path, 2, "localhost", 8765, 64) == 0
    assert len(sent) == 3
    assert "<render attached>" not in (tmp_path / "transcript.json").read_text()
    assert len(json.loads((tmp_path / "calls.json").read_text())) == 3


def test_agent_empty_run_and_reply_without_sentinel(tmp_path):
    assert agent.run("inspect", None, "unused", "unused", tmp_path, 0, "localhost", 8765, 64) == 0
    assert json.loads((tmp_path / "calls.json").read_text()) == []
    assert agent.parse_response('{"result": {"result": "raw log"}}') == {
        "error": "tool produced no result line", "stdout": "raw log",
    }


def test_agent_entrypoint_prints_help_without_contacting_model(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["blender_agent", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(agent.__file__, run_name="__main__")
    assert stopped.value.code == 0
    assert "--task" in capsys.readouterr().out
