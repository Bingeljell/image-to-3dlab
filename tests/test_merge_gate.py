"""Check the actual CI merge gate and its proposed GitHub Ruleset."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/pre-commit.yml"
DEPENDENCIES = {
    "pre-commit": "LINTERS",
    "pytest": "LINUX",
    "pytest-macos": "MACOS",
    "mlx-unit": "MLX",
    "coverage-macos": "COMBINED_COVERAGE",
    "pytest-windows": "WINDOWS",
    "blender-real": "BLENDER",
    "browser-real": "BROWSER",
}


def test_gate_requires_all_checks_and_is_available_to_merge_queue():
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    gate = workflow["jobs"]["merge-gate"]
    assert gate["name"] == "Merge gate"
    assert gate["if"] == "${{ always() }}"
    assert gate["permissions"] == {}
    assert set(gate["needs"]) == DEPENDENCIES.keys()
    assert gate["env"] == {variable: f"${{{{ needs.{job}.result }}}}" for job, variable in DEPENDENCIES.items()}
    assert all("continue-on-error" not in workflow["jobs"][name] for name in DEPENDENCIES)
    assert "merge_group:\n    types: [checks_requested]" in text


@pytest.mark.posix
@pytest.mark.parametrize(("dependency", "result"), [
    (None, "success"),
    *((job, result) for job in DEPENDENCIES.values() for result in ("failure", "skipped", "cancelled", "")),
])
def test_gate_executes_workflow_script_and_demands_success(tmp_path, dependency, result):
    gate = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["merge-gate"]
    fixture = tmp_path / "merge-gate"
    fixture.write_text("#!/bin/bash\nset -eo pipefail\n" + gate["steps"][0]["run"], encoding="utf-8")
    fixture.chmod(0o755)
    env = {**os.environ, **dict.fromkeys(DEPENDENCIES.values(), "success")}
    if dependency:
        env[dependency] = result
    process = subprocess.run([str(fixture)], env=env, capture_output=True, text=True, check=False)
    assert process.returncode == (1 if dependency else 0)
    if dependency:
        assert f"{dependency}: {result or 'missing'}" in process.stdout


def test_proposed_ruleset_requires_fresh_actions_gate_without_bypasses():
    ruleset = json.loads((ROOT / "docs/main-merge-ruleset.json").read_text(encoding="utf-8"))
    assert ruleset["target"] == "branch"
    assert ruleset["enforcement"] == "active"
    assert ruleset["conditions"]["ref_name"] == {"include": ["refs/heads/main"], "exclude": []}
    assert ruleset["bypass_actors"] == []
    rules = {rule["type"]: rule for rule in ruleset["rules"]}
    assert {"deletion", "non_fast_forward", "pull_request"} <= rules.keys()
    required = rules["required_status_checks"]["parameters"]
    assert required == {
        "do_not_enforce_on_create": False,
        "strict_required_status_checks_policy": True,
        "required_status_checks": [{"context": "Merge gate", "integration_id": 15368}],
    }
