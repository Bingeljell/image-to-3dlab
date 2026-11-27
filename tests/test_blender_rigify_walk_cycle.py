"""Check the action name and worker shipped to the remote Blender host."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import blender_rigify_walk_cycle as walk


def test_a_quote_in_the_action_name_stays_a_string():
    for name in ("O'Brian", 'say "hi"'):
        code = walk.build_code(4, "trot", 30.0, 30.0, 35.0, 35.0, 30.0, 8.0, 4.0, name)
        tree = ast.parse(code)
        (bundled_source,) = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
        )
        worker = Path(walk.__file__).with_name("blender_rigify_walk_cycle_worker.py")
        assert bundled_source == worker.read_bytes()
        (arguments,) = (
            json.loads(ast.literal_eval(node.args[0]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
        )
        assert arguments["action_name"] == name
        assert arguments["poses"] == walk.sample(
            4, "trot", 30.0, 30.0, 35.0, 35.0, 30.0, 8.0, 4.0
        )
