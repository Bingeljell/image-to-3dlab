"""Tests for the local-model Blender harness.

Check that client builders send valid requests to the shipped Blender worker.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from inspect import signature
from pathlib import Path

import blender_agent_worker as worker
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("blender_agent", SCRIPTS / "blender_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent = _load()


def worker_request(code):
    """Inspect the worker bytes and JSON packed into the generated request."""
    tree = ast.parse(code)
    compiler, = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "compile"
    ]
    decoder, = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "json"
        and node.func.attr == "loads"
    ]
    return ast.literal_eval(compiler.args[0]), json.loads(ast.literal_eval(decoder.args[0]))


# Representative arguments per tool. Anything the harness owns rather than the model goes
# in the second dict, mirroring how run() calls build_call.
SAMPLE_CALLS: dict[str, tuple[dict, dict]] = {
    "describe_scene": ({}, {}),
    "render_view": ({"resolution": 512, "samples": 16}, {"output_path": "/tmp/r.png"}),
    "import_asset": ({"path": "/tmp/a.glb", "clear_scene": True}, {}),
    "set_camera": ({"azimuth": 35, "elevation": 12, "distance_factor": 2.5,
                    "lens_mm": 85, "target": "Mesh"}, {}),
    "set_light": ({"role": "key", "azimuth": -40, "elevation": 30, "energy": 250,
                   "size": 1.5, "color": [1.0, 0.95, 0.9]}, {}),
    "set_world": ({"strength": 0.4, "color": [0.05, 0.06, 0.08]}, {}),
    "move_object": ({"name": "Mesh", "dx": 0.1, "dy": 0.0, "dz": -0.2}, {}),
    "set_render_settings": ({"engine": "eevee", "film_transparent": False,
                             "exposure": 0.5}, {}),
    "add_primitive": ({"kind": "cube", "name": "cubelet", "location": [0, 0, 0],
                       "size": 0.98, "rotation": [0, 0, 45], "scale": [1, 0.5, 2]}, {}),
    "set_object_transform": ({"name": "cubelet", "location": [1, 0, 0],
                              "rotation": [0, 0, 90], "scale": [1, 1, 2]}, {}),
    "set_material": ({"name": "cubelet", "color": [0.9, 0.1, 0.1], "roughness": 0.4,
                      "metallic": 0.0}, {}),
    "duplicate_grid": ({"name": "cubelet", "counts": [3, 3, 3],
                        "spacing": [1.0, 1.0, 1.0], "prefix": "cube_"}, {}),
    "delete_object": ({"name": "cubelet"}, {}),
    "clear_scene": ({"keep_lights": True}, {}),
    "mirror_object": ({"name": "eye_l", "new_name": "eye_r", "axis": "x"}, {}),
}


def test_every_tool_has_a_sample_call():
    """A tool added without a sample here would never be compile-checked."""
    assert set(SAMPLE_CALLS) == set(agent.TOOLS)


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_generated_code_compiles(name):
    arguments, injected = SAMPLE_CALLS[name]
    code = agent.build_call(name, arguments, injected)
    compile(code, f"<{name}>", "exec")


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_generated_code_carries_the_shipped_worker_and_compatible_arguments(name):
    arguments, injected = SAMPLE_CALLS[name]
    source, request = worker_request(agent.build_call(name, arguments, injected))
    assert source == agent.WORKER_PATH.read_bytes()
    assert request["action"] == name
    signature(worker.ACTIONS[name]).bind(None, **request["parameters"])


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_required_arguments_only_are_enough(name):
    """The model will call these with the minimum. If a builder needs an argument the
    schema does not mark required, that is a crash mid-run."""
    tool = agent.TOOLS[name]
    arguments, injected = SAMPLE_CALLS[name]
    minimal = {k: v for k, v in arguments.items() if k in tool.parameters.get("required", [])}
    code = agent.build_call(name, minimal, injected)
    compile(code, f"<{name}>", "exec")


def test_schemas_are_wellformed():
    for name, tool in agent.TOOLS.items():
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert schema["function"]["description"].strip()
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        # Every required argument must actually be declared.
        assert set(params.get("required", [])) <= set(params["properties"])
        assert tool.kind in {"observe", "modify"}
        # The whole schema has to survive the trip to the API as JSON.
        json.dumps(schema)


def test_library_is_split_into_observation_and_modification():
    """The split is the design, not decoration: it is what the benchmark attributes the
    gain to. Losing either half silently would gut the harness."""
    assert agent.OBSERVE_TOOLS and agent.MODIFY_TOOLS
    assert "describe_scene" in agent.OBSERVE_TOOLS
    assert "render_view" in agent.OBSERVE_TOOLS
    assert "set_camera" in agent.MODIFY_TOOLS


def test_unknown_tool_is_refused():
    with pytest.raises(KeyError):
        agent.build_call("sculpt_a_dragon", {})


def test_invented_argument_is_refused_not_dropped():
    with pytest.raises(TypeError) as excinfo:
        agent.build_call("set_camera", {"azimuth": 0, "elevation": 0, "roll": 45})
    assert "roll" in str(excinfo.value)


def test_nonfinite_arguments_are_refused_at_the_request_boundary():
    with pytest.raises(TypeError, match="JSON-compatible"):
        agent.build_call("move_object", {"name": "body", "dx": float("nan")})


def test_missing_required_argument_is_refused():
    with pytest.raises(TypeError) as excinfo:
        agent.build_call("set_camera", {"azimuth": 0})
    assert "elevation" in str(excinfo.value)


def test_render_path_is_not_model_controlled():
    """output_path is injected by the harness and absent from the schema, so a model
    cannot aim a render at a file that matters."""
    assert "output_path" not in agent.TOOLS["render_view"].parameters["properties"]
    with pytest.raises(TypeError):
        agent.build_call("render_view", {"output_path": "/etc/passwd"})


def test_string_arguments_round_trip_as_data():
    name = "/tmp/café'\n\".glb"
    code = agent.build_call("import_asset", {"path": name})
    compile(code, "<asset name>", "exec")
    _, request = worker_request(code)
    assert request["parameters"] == {"path": name, "clear_scene": True}


def test_parse_response_reads_the_result_line():
    stdout = f"Info: some blender chatter\n{agent.RESULT_SENTINEL}{{\"camera\": \"AGENT_CAM\"}}\n"
    raw = json.dumps({"status": "success", "result": {"result": stdout}})
    assert agent.parse_response(raw) == {"camera": "AGENT_CAM"}


def test_parse_response_prefers_the_last_result_line():
    """Blender's stdout is shared and can still hold a previous call's line."""
    stdout = (f"{agent.RESULT_SENTINEL}{{\"stale\": true}}\n"
              f"{agent.RESULT_SENTINEL}{{\"fresh\": true}}\n")
    raw = json.dumps({"status": "success", "result": {"result": stdout}})
    assert agent.parse_response(raw) == {"fresh": True}


def test_parse_response_surfaces_blender_errors():
    raw = json.dumps({"status": "error", "message": "NameError: bpy is not defined"})
    parsed = agent.parse_response(raw)
    assert "NameError" in parsed["error"]


def test_parse_response_survives_garbage():
    for raw in ("", "not json at all", '{"status": "success"'):
        assert "error" in agent.parse_response(raw)


def test_image_message_carries_a_data_uri(tmp_path):
    png = tmp_path / "r.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    message = agent.image_message(png, "look")
    assert message["role"] == "user"
    kinds = [part["type"] for part in message["content"]]
    assert kinds == ["text", "image_url"]
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_run_log_strips_images_from_the_transcript(tmp_path):
    log = agent.RunLog(tmp_path / "run")
    log.record("set_camera", {"azimuth": 0, "elevation": 0}, {"camera": "AGENT_CAM"})
    messages = [
        {"role": "user", "content": "go"},
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]
    log.write(messages)
    written = (tmp_path / "run" / "transcript.json").read_text()
    assert "base64" not in written
    assert "<render attached>" in written
    calls = json.loads((tmp_path / "run" / "calls.json").read_text())
    assert calls[0]["tool"] == "set_camera"


def test_list_tools_needs_no_blender(capsys):
    assert agent.main(["--task", "x", "--out-dir", "/tmp/unused", "--list-tools"]) == 0
    printed = capsys.readouterr().out
    assert "describe_scene" in printed and "set_camera" in printed


def test_transform_arguments_survive_being_omitted():
    code = agent.build_call("set_object_transform", {"name": "cubelet", "scale": [2, 2, 2]})
    _, request = worker_request(code)
    assert request["parameters"] == {
        "name": "cubelet", "location": None, "rotation": None, "scale": [2, 2, 2],
    }


def test_duplicate_grid_clamps_counts_to_at_least_one():
    code = agent.build_call("duplicate_grid", {"name": "c", "counts": [0, -4, 3], "spacing": [1]})
    _, request = worker_request(code)
    assert request["parameters"] == {
        "name": "c", "counts": [1, 1, 3], "spacing": [1, 0, 0], "prefix": None,
    }


def test_primitive_kind_survives_the_request():
    kinds = agent.TOOLS["add_primitive"].parameters["properties"]["kind"]["enum"]
    for kind in kinds:
        _, request = worker_request(agent.build_add_primitive(kind, "sample"))
        assert request["parameters"]["kind"] == kind


def test_short_vectors_get_padded():
    _, request = worker_request(agent.build_add_primitive(
        "cube", "a", location=[1, 2], rotation=[5], scale=[2],
    ))
    parameters = request["parameters"]
    assert parameters["location"] == [1, 2, 0]
    assert parameters["rotation"] == [5, 0, 0]
    assert parameters["scale"] == [2, 1, 1]


def test_opening_messages_without_a_reference():
    messages = agent.opening_messages("build a cube", None, None)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "build a cube" in messages[1]["content"]


def test_opening_messages_carry_the_reference_image(tmp_path):
    png = tmp_path / "ref.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    messages = agent.opening_messages("match this", None, png)
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    parts = messages[2]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "reference" in parts[0]["text"].lower()


def test_opening_messages_mention_the_asset(tmp_path):
    messages = agent.opening_messages("light it", tmp_path / "creature.glb", None)
    assert "creature.glb" in messages[1]["content"]


def test_missing_reference_file_is_caught_before_the_model_loads():
    """Loading a 20 GB model and then failing on a typo'd path is a five-minute mistake."""
    with pytest.raises(SystemExit):
        agent.main(["--task", "x", "--out-dir", "/tmp/u", "--reference", "/nope/absent.png"])


def test_render_engine_schema_offers_only_loose_names():
    """The model should name an engine loosely; exact identifiers are version-specific."""
    enum = agent.TOOLS["set_render_settings"].parameters["properties"]["engine"]["enum"]
    assert all(not name.startswith("BLENDER_") for name in enum), enum


def test_a_round_that_modifies_without_looking_gets_a_forced_render():
    """The measured failure on the first live run: eight calls, seven modifications, zero
    renders. The feedback loop is the mechanism, so the harness enforces it."""
    assert agent.needs_forced_render(["add_primitive", "set_material", "set_camera"])


def test_a_round_that_rendered_is_left_alone():
    assert not agent.needs_forced_render(["add_primitive", "render_view"])
    assert not agent.needs_forced_render(["set_camera", "render_view", "set_light"])


def test_a_round_that_only_looked_is_left_alone():
    """describe_scene changes nothing, so there is nothing new to see."""
    assert not agent.needs_forced_render(["describe_scene"])
    assert not agent.needs_forced_render([])


def test_forced_render_ignores_unknown_tool_names():
    """A hallucinated tool name reaches this list too; it must not count as a change."""
    assert not agent.needs_forced_render(["sculpt_a_dragon"])


def test_run_makes_the_out_dir_absolute(tmp_path, monkeypatch):
    """The generated code runs inside Blender, whose cwd is not ours. A relative path made
    every render fail with a read-only filesystem error while every other tool worked."""
    monkeypatch.chdir(tmp_path)
    sent: list[str] = []

    def fake_chat(api, model, messages, tools, timeout=600):
        return {"choices": [{"message": {"role": "assistant", "content": "done",
                                         "tool_calls": []}}]}

    monkeypatch.setattr(agent, "chat", fake_chat)
    monkeypatch.setattr(agent, "send", lambda code, *a, **k: sent.append(code) or "{}")
    agent.run("x", None, "m", "http://localhost", Path("relative/out"), 1,
              "127.0.0.1", 9876, 256)
    assert (tmp_path / "relative" / "out" / "transcript.json").exists()


def test_set_material_colours_many_at_once():
    _, request = worker_request(agent.build_set_material([1, 0, 0], names=["a", "b", "c"]))
    assert request["parameters"]["targets"] == ["a", "b", "c"]


def test_set_material_still_takes_a_single_name():
    _, request = worker_request(agent.build_set_material([0, 1], name="solo"))
    assert request["parameters"]["targets"] == ["solo"]
    assert request["parameters"]["color"] == [0, 1, 0]


def test_set_material_can_reach_worker_without_a_target():
    _, request = worker_request(agent.build_set_material([1, 1, 1]))
    assert request["parameters"]["targets"] == []


def test_add_primitive_can_shape_in_one_call():
    _, request = worker_request(agent.build_add_primitive("cube", "torso", scale=[1.0, 0.5, 1.4]))
    assert request["parameters"]["scale"] == [1.0, 0.5, 1.4]


def test_add_primitive_scale_defaults_to_unit():
    _, request = worker_request(agent.build_add_primitive("sphere", "head"))
    assert request["parameters"]["scale"] == [1.0, 1.0, 1.0]


def test_mirror_defaults_to_the_left_right_axis():
    _, request = worker_request(agent.build_mirror_object("a", "b"))
    assert request["parameters"]["axis"] == "x"


@pytest.mark.parametrize("last", [b'"success"}', b"", TimeoutError()])
def test_socket_sends_the_supplied_code_and_collects_split_response(monkeypatch, last):
    chunks = iter([b'{"status":', last])
    sent = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            sent.append("closed")

        def settimeout(self, timeout):
            assert timeout == 12

        def sendall(self, request):
            sent.append(json.loads(request))

        def recv(self, size):
            assert size == 65536
            chunk = next(chunks)
            if isinstance(chunk, TimeoutError):
                raise chunk
            return chunk

    def connect(address, *, timeout):
        assert address == ("localhost", 4000)
        assert timeout == 10
        return Connection()

    monkeypatch.setattr(agent.socket, "create_connection", connect)
    code = 'print("arbitrary Blender code supplied by the caller")'
    result = agent.send(code, "localhost", 4000, timeout=12)
    assert sent == [{"type": "execute_code", "params": {"code": code}}, "closed"]
    assert result == ('{"status":"success"}' if last == b'"success"}' else '{"status":')


def test_default_blender_error_and_invalid_result_are_readable():
    assert agent.parse_response('{"status":"error"}') == {"error": "Blender reported an error"}
    invalid = json.dumps({"result": agent.RESULT_SENTINEL + "{garbage"})
    assert agent.parse_response(invalid)["error"].startswith("unparseable result line:")


def test_package_entrypoint_uses_the_same_worker():
    from scripts import blender_agent

    assert blender_agent.build_describe_scene() == agent.build_describe_scene()
