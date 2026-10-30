"""An action name is quoted into the Blender snippet, so a quote stays a quote."""

from __future__ import annotations

import ast

import blender_rigify_walk_cycle as walk


def test_a_quote_in_the_action_name_stays_a_string():
    for name in ("O'Brian", 'say "hi"'):
        code = walk.build_code(4, "trot", 30.0, 30.0, 35.0, 35.0, 30.0, 8.0, 4.0, name)
        tree = ast.parse(code)
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert name in literals
