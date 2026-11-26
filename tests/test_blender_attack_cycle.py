"""Check the action name and worker shipped to the remote Blender host."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import blender_attack_cycle as attack


def test_a_quote_in_the_action_name_stays_a_string():
    for name in ("O'Brian", 'say "hi"'):
        code = attack.build_code(4, name, 0.35)
        tree = ast.parse(code)
        (bundled_source,) = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
        )
        worker = Path(attack.__file__).with_name("blender_attack_cycle_worker.py")
        assert bundled_source == worker.read_bytes()
        (arguments,) = (
            json.loads(ast.literal_eval(node.args[0]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
        )
        assert arguments["action_name"] == name
        assert arguments["poses"] == attack.sample(4)
        assert arguments["toe"] == 0.35
