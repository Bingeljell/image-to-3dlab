"""Live Blender helpers must not wipe the user's open file.

A wind demo, a normal fix, and a solidify bake used to delete every
object and purge orphans. That emptied unsaved scenes. The generated code is the
artifact sent to Blender, so these tests import it and read it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_scene import isolated_scene_prelude, restore_previous_scene


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene_request(code):
    tree = ast.parse(code)
    source, = (
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
    )
    request, = (
        json.loads(ast.literal_eval(node.args[0]))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loads"
    )
    return source, request


@pytest.mark.parametrize("culled", [False, True])
def test_stage_request_bundles_its_worker_and_culling_choice(culled, tmp_path):
    stage = _load("blender_stage")
    path = tmp_path / "subject.glb"
    source, request = _scene_request(stage.build_code([str(path)], ["A_mesh"], [1.2], culled))
    assert source == stage.WORKER_PATH.read_bytes()
    assert request == {
        "action": "stage_assets",
        "parameters": {"items": [{"path": str(path), "name": "A_mesh", "dy": 1.2}], "culled": culled},
    }


@pytest.mark.parametrize(
    ("command", "builder", "args", "action", "parameter", "expected"),
    [
        ("fix_normals", "blender_code", (Path("/tmp/a.glb"), Path("/tmp/out.glb")),
         "fix_normals", "output", "/tmp/out.glb"),
        ("light_orbit", "blender_code", (Path("/tmp/a.glb"), Path("/tmp/f_"), 1, 52.0, 64, "[[0,0,0]]"),
         "light_orbit", "rotations", [[0, 0, 0]]),
        ("stage_bake", "build_code", ("Walk", 0.0, 0.0, None, None, None, 1, 256, "/tmp/out", "test"),
         "stage_bake", "action_name", "Walk"),
        ("joint_markers", "spawn_code", (Path("/tmp/a.glb"),),
         "spawn_markers", "confirm", False),
        ("joint_markers", "spawn_code", (Path("/tmp/a.glb"), True),
         "spawn_markers", "confirm", True),
        ("joint_markers", "load_code", ({"base": [1, 2, 3]},),
         "load_markers", "markers", {"base": [1, 2, 3]}),
        ("joint_markers", "read_code", (), "read_markers", "prefix", "JOINT_"),
        ("pick_pixel", "build_code", (0, 0, 4, 2, "profile_yneg"),
         "pick_pixel", "ndc", [-0.75, 0.5]),
        ("orbit_camera", "build_code", ("Cam", (1, 2, 3), (0, 0, -1), 32, None, (64, 32)),
         "orbit_camera", "resolution", [64, 32]),
    ],
)
def test_scene_tools_bundle_the_dispatched_worker(command, builder, args, action, parameter, expected):
    client = _load("blender_" + command)
    code = getattr(client, builder)(*args)
    compile(code, "<Blender request>", "exec")
    source, request = _scene_request(code)
    assert source == client.WORKER_PATH.read_bytes()
    assert request["action"] == action
    assert request["parameters"][parameter] == expected


def test_prelude_only_clears_its_own_scene():
    code = isolated_scene_prelude("IMG3D turntable")
    assert "bpy.data.scenes.new" in code or "bpy.data.scenes.get" in code
    assert "for obj in list(render_scene.objects)" in code
    assert "bpy.data.objects)" not in code.split("render_scene.objects")[0]
    assert restore_previous_scene() == "bpy.context.window.scene = previous_scene\n"
