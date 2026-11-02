"""Tests for the local-model Blender harness.

The expensive failure this guards against is a malformed bpy snippet reaching Blender
halfway through an agent run: the model gets back a syntax error it cannot fix, burns its
remaining rounds, and the whole run is wasted. Compiling every builder's output costs
milliseconds and catches exactly that, which is AGENTS.md's "verify the environment before
the expensive step".

The builders are imported and called, never re-derived. A test that re-implemented the
string building would be testing a copy of the code rather than the code.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

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
def test_generated_code_reports_through_the_sentinel(name):
    """parse_response looks for exactly one marker; a builder that forgets it produces a
    tool the model sees as permanently broken."""
    arguments, injected = SAMPLE_CALLS[name]
    code = agent.build_call(name, arguments, injected)
    assert agent.RESULT_SENTINEL in code
    assert "payload" in code


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


def test_string_arguments_cannot_break_out_of_the_snippet():
    """Paths reach Blender as Python literals via repr, not by concatenation. A quote or
    a newline in an asset name must stay data."""
    nasty = "/tmp/a'; import os; os.system('echo pwned'); x='.glb"
    code = agent.build_call("import_asset", {"path": nasty})
    compile(code, "<nasty>", "exec")
    tree = ast.parse(code)
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                and isinstance(node.value, str)}
    assert nasty in literals  # present as one literal, not spliced into source
    assert "os.system" not in {
        getattr(getattr(node.func, "value", None), "id", None)
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }


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
    """set_object_transform takes three optional vectors; a None must reach Blender as a
    literal None, not as the string "None"."""
    code = agent.build_call("set_object_transform", {"name": "cubelet", "scale": [2, 2, 2]})
    compile(code, "<transform>", "exec")
    tree = ast.parse(code)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "None" not in names  # None is a Constant, never a bare Name
    assert '"None"' not in code and "'None'" not in code


def test_duplicate_grid_refuses_a_runaway():
    """A model that means 3 and types 30 gets an error, not a hung Blender. The cap lives
    in the generated code, so assert it is actually in there."""
    code = agent.build_call("duplicate_grid",
                            {"name": "cubelet", "counts": [30, 30, 30], "spacing": [1, 1, 1]})
    assert "200" in code
    compile(code, "<grid>", "exec")


def test_duplicate_grid_clamps_counts_to_at_least_one():
    code = agent.build_call("duplicate_grid",
                            {"name": "c", "counts": [0, -4, 3], "spacing": [1, 1, 1]})
    assert "[1, 1, 3]" in code


def test_primitive_kinds_in_the_schema_all_exist_in_the_builder():
    """A kind the schema advertises but the builder cannot make is a tool that fails only
    once the model tries it, several minutes into a run."""
    kinds = agent.TOOLS["add_primitive"].parameters["properties"]["kind"]["enum"]
    code = agent.build_call("add_primitive", {"kind": "cube", "name": "x"})
    for kind in kinds:
        assert f'"{kind}"' in code, f"{kind} is offered but the builder has no adder"


def test_short_vectors_do_not_crash_the_builder():
    """Models emit [0, 0] for a 3-vector often enough that it must not raise."""
    for args in ({"kind": "cube", "name": "a", "location": []},
                 {"kind": "cube", "name": "b", "location": [1, 2]},
                 {"kind": "cube", "name": "c", "rotation": [5]}):
        compile(agent.build_call("add_primitive", args), "<short>", "exec")


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


def test_render_engine_is_resolved_against_the_running_blender():
    """Blender renamed the realtime engine (BLENDER_EEVEE_NEXT in 4.2+, BLENDER_EEVEE in
    5.2). A hardcoded identifier crashes on the other version, which it did. The snippet
    must ask Blender what it has rather than assuming."""
    code = agent.build_call("set_render_settings", {"engine": "eevee"})
    assert "enum_items" in code, "engine must be resolved from Blender's own enum"
    assert "BLENDER_EEVEE_NEXT" not in code
    compile(code, "<engine>", "exec")


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


def test_default_import_clears_meshes_only():
    """A default import used to delete armatures and joint-marker empties too."""
    code = agent.build_import_asset("/tmp/a.glb")
    assert '"MESH"' in code
    assert '"ARMATURE"' not in code
    assert '"EMPTY"' not in code


def test_clear_scene_removes_only_meshes_and_lights():
    """The tool text says meshes and lights. Cameras, rigs and markers stay."""
    code = agent.build_clear_scene()
    assert '"MESH"' in code
    assert '"LIGHT"' in code
    assert '"ARMATURE"' not in code
    assert '"EMPTY"' not in code
    assert '"CAMERA"' not in code
    compile(code, "<clear>", "exec")


def test_clear_scene_keeps_the_camera():
    """Deleting the camera mid-run leaves the model unable to render, and it has no verb
    to make a new one except set_camera, which it may not think to call."""
    code = agent.build_call("clear_scene", {})
    assert '"CAMERA"' not in code
    compile(code, "<clear>", "exec")


def test_clear_scene_can_spare_the_lights():
    assert "keep_lights" in agent.build_call("clear_scene", {"keep_lights": True})


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


def test_duplicate_grid_gives_each_copy_its_own_mesh():
    """Linked duplicates share a mesh datablock, and material slots live on the mesh. The
    first live Rubik's cube came out uniformly blue because of this: every set_material
    call repainted all 27 cubelets."""
    code = agent.build_call("duplicate_grid",
                            {"name": "c", "counts": [3, 3, 3], "spacing": [1, 1, 1]})
    assert "source.data.copy()" in code
    assert "copy.data = source.data\n" not in code


def test_set_material_colours_many_at_once():
    code = agent.build_call("set_material", {"names": ["a", "b", "c"], "color": [1, 0, 0]})
    compile(code, "<mat>", "exec")
    assert "['a', 'b', 'c']" in code


def test_set_material_still_takes_a_single_name():
    code = agent.build_call("set_material", {"name": "solo", "color": [0, 1, 0]})
    assert "['solo']" in code


def test_set_material_with_nothing_to_colour_tells_the_model_so():
    """Neither name nor names is required by the schema, so a bare colour is a call an
    honest model will make. It must get a readable error back, not a crash."""
    code = agent.build_call("set_material", {"color": [1, 1, 1]})
    compile(code, "<mat>", "exec")
    assert "needs name" in code


def test_set_camera_ignores_flat_ground_by_default():
    """Framing at 3x the radius of everything put the camera 21 units from a 2-unit
    sphere, because a 10-unit floor dominated the bounds. Measured on the reference run."""
    code = agent.build_call("set_camera", {"azimuth": 0, "elevation": 0})
    assert "_flat" in code
    compile(code, "<cam>", "exec")


def test_set_camera_honours_an_explicit_target_even_if_flat():
    """Naming the ground as the target must still frame the ground."""
    code = agent.build_call("set_camera", {"azimuth": 0, "elevation": 0, "target": "floor"})
    assert "if target_name:" in code
    compile(code, "<cam>", "exec")


def test_add_primitive_can_shape_in_one_call():
    """Creating then shaping as two calls doubles the round cost of every body part, and
    a forced render sits between rounds. A fifteen-part figure has to fit the budget."""
    code = agent.build_call("add_primitive",
                            {"kind": "cube", "name": "torso", "scale": [1.0, 0.5, 1.4]})
    assert "obj.scale" in code
    assert "[1.0, 0.5, 1.4]" in code
    compile(code, "<prim>", "exec")


def test_add_primitive_scale_defaults_to_unit():
    code = agent.build_call("add_primitive", {"kind": "sphere", "name": "head"})
    assert "[1.0, 1.0, 1.0]" in code


@pytest.mark.parametrize("name,args,reader", [
    ("add_primitive", {"kind": "cube", "name": "arm", "scale": [0.1, 0.1, 0.6]},
     "obj.dimensions"),
    ("set_object_transform", {"name": "arm", "scale": [0.1, 0.1, 0.6]}, "obj.dimensions"),
    ("describe_scene", {}, "obj.matrix_world"),
])
def test_dimensions_are_read_after_the_depsgraph_refreshes(name, args, reader):
    """obj.dimensions derives from the object matrix and does not refresh until the
    dependency graph does. Without the update the payload reported the UNSCALED size: the
    model was told its 0.02-wide arm was a 0.2 cube, built a figure with thread limbs, and
    could not see why. A stale number is worse than no number.

    The comparison is against the first read of a derived attribute, not the word
    "dimensions", which also appears in the comment explaining this very trap.
    """
    code = agent.build_call(name, args)
    assert "view_layer.update()" in code
    assert code.index("view_layer.update()") < code.index(reader)


def test_mirror_negates_both_location_and_scale():
    """A mirrored part that keeps its original scale is not a mirror, it is a translation,
    and asymmetric geometry gives it away."""
    code = agent.build_call("mirror_object", {"name": "arm_l", "new_name": "arm_r"})
    assert "location[index] = -location[index]" in code
    assert "scale[index] = -scale[index]" in code
    compile(code, "<mirror>", "exec")


def test_mirror_defaults_to_the_left_right_axis():
    """x is left/right in Blender space, which is the axis a character needs."""
    code = agent.build_call("mirror_object", {"name": "a", "new_name": "b"})
    assert "'x'" in code


def test_mirror_gives_the_copy_its_own_mesh():
    """Same trap as duplicate_grid: a shared datablock means colouring one recolours both."""
    code = agent.build_call("mirror_object", {"name": "a", "new_name": "b"})
    assert "source.data.copy()" in code


def test_mirror_refuses_a_name_already_in_use():
    code = agent.build_call("mirror_object", {"name": "a", "new_name": "b"})
    assert "already exists" in code
