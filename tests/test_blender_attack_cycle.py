"""An action name is quoted into the Blender snippet, so a quote stays a quote."""

from __future__ import annotations

import ast

import blender_attack_cycle as attack


def test_a_quote_in_the_action_name_stays_a_string():
    for name in ("O'Brian", 'say "hi"'):
        code = attack.build_code(4, name, 0.35)
        tree = ast.parse(code)
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert name in literals
