"""Collect real coverage in two checkouts and merge it through the shipped CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import coverage
import pytest
from unit_isolation import COVERAGE_ENV

pytestmark = pytest.mark.posix
ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("acceptance_reporting") / "run_coverage.py"
BRANCH_SOURCE = """def choose(condition):
    if condition:
        return 1
    return 2
"""
JS_SOURCE = """export function choose(condition) {
  if (condition) {
    return 1;
  }
  return 2;
}
"""
BADGE_TITLES = {
    "python-lines.svg": "Python lines: 4 of 4 covered (100.00%)",
    "python-branches.svg": "Python branches: 2 of 2 covered (100.00%)",
    "javascript-lines.svg": "JavaScript lines: 3 of 3 covered (100.00%)",
    "javascript-branches.svg": "JavaScript branches: 2 of 2 covered (100.00%)",
}


def _run(args, cwd, env, *, check=True):
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def _checkout(root, low_ratio, env):
    files = {
        ".gitignore": "__pycache__/\n.pytest_cache/\nnode_modules/\n",
        ".node-version": (ROOT / ".node-version").read_text(encoding="utf-8"),
        "pyproject.toml": '[tool.coverage.run]\nbranch = true\nparallel = true\nomit = ["*/tests/*"]\n',
        "install.sh": "#!/bin/sh\n",
        "sample.py": BRANCH_SOURCE,
        "sample.mjs": JS_SOURCE,
        "tests/conftest.py": """
def pytest_addoption(parser):
    parser.addoption("--run-mlx-unit", action="store_true")
""",
        "tests/test_portable.py": """
from sample import choose
import subprocess

def test_installer_runs():
    subprocess.run(["./install.sh"], check=True)

def test_true():
    assert choose(True) == 1
""",
        "tests/mlx/test_mlx.py": """
import pytest
from sample import choose

def test_false(request):
    if not request.config.getoption("--run-mlx-unit"):
        pytest.skip("optional suite")
    assert choose(False) == 2
""",
        "tests/js/sample.test.js": """
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {choose} from '../../sample.mjs';
test('both outcomes', () => {
  assert.equal(choose(true), 1);
  assert.equal(choose(false), 2);
});
""",
    }
    if low_ratio:
        # 983/1024 rounds to 96.00% for a badge but fails the unrounded 96% gate.
        files["sample.py"] += "value = 1\n" * 979
        files["untouched.py"] = "value = 1\n" * 41
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    (root / "install.sh").chmod(0o755)
    _run(["git", "init", "-q"], root, env)
    _run(["git", "add", "."], root, env)
    _run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
          "commit", "-qm", "Fixture sources"], root, env)


def _cli(root, env, *args, check=True):
    # The parent's process guard propagates coverage variables. Remove those
    # for this child so its collectors use their own data and instrumentation.
    with pytest.MonkeyPatch.context() as isolated:
        for key in COVERAGE_ENV:
            isolated.delenv(key, raising=False)
        return _run([sys.executable, str(RUNNER), str(root), *map(str, args)], root, env, check=check)


@pytest.fixture(scope="module", params=[False, True], ids=["complete", "untouched"])
def artifacts(request, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("coverage-handoff")
    home = workspace / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ALLOW_PROTOCOL": "file",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(ROOT / "tests/support"),
    }
    portable_root = workspace / "portable-checkout"
    mlx_root = workspace / "mlx-checkout"
    _checkout(portable_root, request.param, env)
    env["GITHUB_SHA"] = _run(["git", "rev-parse", "HEAD"], portable_root, env).stdout.strip()
    _run(["git", "clone", "-q", portable_root.as_uri(), str(mlx_root)], workspace, env)
    (portable_root / "node_modules").symlink_to(ROOT / "node_modules", target_is_directory=True)
    portable = workspace / "portable"
    mlx = workspace / "mlx"
    _cli(portable_root, env, portable, "-q")
    _cli(mlx_root, env, mlx, "--mlx-unit")
    return portable_root, portable, mlx, env, request.param


def test_coverage_cli_collects_and_merges_exact_counts(tmp_path, artifacts):
    root, portable, mlx, env, low_ratio = artifacts
    manifests = [json.loads((folder / "run.json").read_text(encoding="utf-8")) for folder in (portable, mlx)]
    assert [run["kind"] for run in manifests] == ["portable", "mlx-unit"]
    assert [Path(run["root"]).name for run in manifests] == ["portable-checkout", "mlx-checkout"]
    assert all(run["complete"] and run["python"] == list(sys.version_info[:2]) for run in manifests)

    for folder, run, hit, missed in zip((portable, mlx), manifests, ((2, 3), (2, 4)), ((2, 4), (2, 3))):
        data = coverage.CoverageData(basename=str(folder / ".coverage"))
        data.read()
        arcs = data.arcs(str(Path(run["root"]) / "sample.py"))
        assert arcs is not None and hit in arcs and missed not in arcs

    assert any((portable / "js-raw").glob("*.json"))
    output = tmp_path / "combined"
    summary = tmp_path / "summary.md"
    result = _cli(root, {**env, "GITHUB_STEP_SUMMARY": str(summary)}, output,
                  "--merge", portable, mlx, "--enforce", check=False)
    assert result.returncode == int(low_ratio), result.stdout + result.stderr
    lines = "983/1024 (95.9961%) below target" if low_ratio else "4/4 (100.0000%) meets target"
    assert result.stdout.splitlines() == [
        "Combined macOS unit coverage",
        f"Python lines: {lines}",
        "Python branches: 2/2 (100.0000%) meets target",
        "JavaScript lines: 3/3 (100.0000%) meets target",
        "JavaScript branches: 2/2 (100.0000%) meets target",
    ]
    js_report = (output / "javascript.json").read_text(encoding="utf-8")
    assert json.loads(js_report) == json.loads((portable / "javascript.json").read_text(encoding="utf-8"))
    python = json.loads((output / "python.json").read_text(encoding="utf-8"))
    if low_ratio:
        assert len(python["files"]["untouched.py"]["missing_lines"]) == 41
        assert "| 983 | 1,024 | 95.9961% | 96% | Below target |" in summary.read_text(encoding="utf-8")
        assert not (output / "badges").exists()
    else:
        assert set(python["files"]) == {"sample.py"}
        badges = output / "badges"
        assert {path.name: ElementTree.parse(path).getroot().findtext("{*}title")
                for path in badges.iterdir()} == BADGE_TITLES


@pytest.mark.parametrize(("problem", "error"), [
    ("missing data", "missing MLX unit coverage"),
    ("checkout", "GitHub checkout differs"),
    ("installer", "coverage sources differ"),
    ("execute permission", "coverage sources differ"),
])
def test_coverage_cli_rejects_incompatible_artifacts(tmp_path, artifacts, problem, error):
    root, portable, mlx, env, _ = artifacts
    checkout = tmp_path / "merge-checkout"
    _run(["git", "clone", "-q", root.as_uri(), str(checkout)], tmp_path, env)
    env = dict(env)
    if problem == "missing data":
        mlx = shutil.copytree(mlx, tmp_path / "incomplete-artifact", ignore=shutil.ignore_patterns(".coverage"))
    elif problem == "checkout":
        env["GITHUB_SHA"] = "0" * 40
    elif problem == "execute permission":
        env.pop("GITHUB_SHA")
        (checkout / "install.sh").chmod(0o644)
        with pytest.raises(PermissionError):
            subprocess.run([str(checkout / "install.sh")], check=True, env=env)
    else:
        env.pop("GITHUB_SHA")
        (checkout / "install.sh").write_text("#!/bin/sh\n# New installer\n", encoding="utf-8")
    output = tmp_path / "rejected"
    result = _cli(checkout, env, output, "--merge", portable, mlx, "--enforce", check=False)
    assert result.returncode == 1
    assert error in result.stderr
    assert not (output / "badges").exists()


@pytest.mark.parametrize("artifact", [".coverage", "javascript.json", "missing hashes", "missing JavaScript"])
def test_coverage_cli_rejects_mixed_or_unbound_reports(tmp_path, artifacts, artifact):
    root, portable, mlx, env, _ = artifacts
    mixed = shutil.copytree(portable, tmp_path / "mixed")
    if artifact == ".coverage":
        shutil.copy2(mlx / ".coverage", mixed / ".coverage")
    elif artifact == "javascript.json":
        # A different, still-passing report with exactly the same filenames.
        path = mixed / artifact
        report = json.loads(path.read_text())
        report["totals"]["branches"] = {"covered": 100, "total": 100}
        path.write_text(json.dumps(report))
    elif artifact == "missing JavaScript":
        (mixed / "javascript.json").unlink()
    else:
        path = mixed / "run.json"
        manifest = json.loads(path.read_text())
        manifest.pop("artifacts", None)
        path.write_text(json.dumps(manifest))
    output = tmp_path / "rejected"
    result = _cli(root, env, output, "--merge", mixed, mlx, "--enforce", check=False)
    assert result.returncode == 1
    assert "coverage artifact" in result.stderr
    assert not (output / "badges").exists()


def test_old_passing_javascript_cannot_replace_a_fresh_failing_report(tmp_path, artifacts):
    root, portable, _, env, _ = artifacts
    checkout = tmp_path / "checkout"
    _run(["git", "clone", "-q", root.as_uri(), str(checkout)], tmp_path, env)
    (checkout / "node_modules").symlink_to(ROOT / "node_modules", target_is_directory=True)
    test = checkout / "tests/js/sample.test.js"
    test.write_text(test.read_text().replace("  assert.equal(choose(false), 2);\n", ""))
    fresh = tmp_path / "fresh-portable"
    mlx = tmp_path / "fresh-mlx"
    _cli(checkout, env, fresh, "-q")
    _cli(checkout, env, mlx, "--mlx-unit")
    valid = tmp_path / "valid-below-target"
    result = _cli(checkout, env, valid, "--merge", fresh, mlx, "--enforce", check=False)
    assert result.returncode == 1
    assert "JavaScript branches: 1/2 (50.0000%) below target" in result.stdout
    assert not (valid / "badges").exists()

    shutil.copy2(portable / "javascript.json", fresh / "javascript.json")
    output = tmp_path / "mixed-rejected"
    result = _cli(checkout, env, output, "--merge", fresh, mlx, "--enforce", check=False)
    assert result.returncode == 1
    assert "coverage artifact differs: javascript.json" in result.stderr
    assert not (output / "badges").exists()
