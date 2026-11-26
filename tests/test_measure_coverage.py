"""Regression tests for coverage scope, counters, and child processes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from scripts import measure_coverage
from scripts.measure_coverage import (
    inventory,
    javascript_report,
    python_report,
    require_inventory,
    run_javascript_tests,
    threshold_met,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("covered", "total", "passes"), [(95999, 100000, False), (96, 100, True), (97, 100, True), (0, 0, False)])
def test_threshold_uses_raw_counts(covered, total, passes):
    assert threshold_met(covered, total) is passes


def test_report_must_match_inventory():
    with pytest.raises(ValueError, match=r"missing=\['a.py'\].*unexpected=\['b.py'\]"):
        require_inventory(["a.py"], {"b.py": {}})


@pytest.mark.parametrize(("arguments", "test_status", "expected"), [
    ([], 0, 0), (["--enforce"], 0, 1), ([], 2, 2),
])
def test_cli_reports_below_target_and_preserves_test_failures(
    tmp_path, monkeypatch, capsys, arguments, test_status, expected,
):
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", str(tmp_path / "report"), *arguments])
    counts = {"Python": {"lines": (95999, 100000), "branches": (96, 100)}}
    monkeypatch.setattr(measure_coverage, "run", lambda output, pytest_args: (test_status, counts))
    assert measure_coverage.main() == expected
    assert "95999/100000 (95.9990%) below target" in capsys.readouterr().out


@pytest.mark.parametrize("selector", ["argument", "environment"])
def test_enforcement_refuses_a_filtered_portable_run(tmp_path, monkeypatch, selector):
    args = ["measure_coverage.py", str(tmp_path / "report"), "--enforce"]
    if selector == "argument":
        args.extend(["--", "-k", "small"])
    else:
        monkeypatch.setenv("PYTEST_ADDOPTS", "-k small")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(measure_coverage, "run", lambda *_: pytest.fail("started a filtered enforced run"))
    with pytest.raises(SystemExit) as stopped:
        measure_coverage.main()
    assert stopped.value.code == 2


def test_inventory_checks_inlined_javascript(tmp_path):
    page = tmp_path / "page.html"
    page.write_text('<script type="module">import "./demo.js";</script>', encoding="utf-8")
    with pytest.raises(ValueError, match="page.html"):
        inventory(tmp_path, ["page.html"])


def test_inventory_excludes_only_tests_and_bundles(tmp_path):
    page = tmp_path / "page.html"
    page.write_text('<script type="importmap">{}</script><script src="demo.js"></script>', encoding="utf-8")
    assert inventory(tmp_path, [
        "pipeline.py", "hunyuan_mlx/paint/scripts/run_paint.py",
        "hunyuan_mlx/paint/tests/test_weights.py", "tests/support/test_hook.mjs",
        "viewer/vendor/three.module.js", "viewer/vendor/new-maintained.js",
        "understanding_3d/demo.js", "page.html",
    ]) == {
        "python": ["hunyuan_mlx/paint/scripts/run_paint.py", "pipeline.py"],
        "javascript": ["understanding_3d/demo.js", "viewer/vendor/new-maintained.js"],
    }


def test_python_report_counts_unloaded_file_and_child_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in ["loaded.py", "unloaded.py"]:
        (tmp_path / path).write_text("def choose(flag):\n    if flag:\n        return 1\n    return 2\n", encoding="utf-8")
    (tmp_path / "parent.py").write_text(
        "import os, subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', "
        "'import os; from loaded import choose; assert choose(True) == 1; os._exit(0)'], "
        "check=True, env={key: value for key, value in os.environ.items() if key.upper() in {'PATH', 'SYSTEMROOT'}})\n",
        encoding="utf-8",
    )
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nbranch = True\npatch = subprocess, _exit\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "coverage", "run", "--rcfile", str(config), "parent.py"],
                   env={**{key: value for key, value in os.environ.items() if not key.startswith("COVERAGE_")},
                        "COVERAGE_PROCESS_CONFIG": "",
                        "COVERAGE_FILE": str(tmp_path / ".coverage")}, check=True)
    report = python_report(tmp_path, ["loaded.py", "unloaded.py"], config)
    assert report["files"]["loaded.py"]["summary"]["covered_branches"] == 1
    assert report["files"]["unloaded.py"]["summary"]["num_branches"] == 2
    assert report["files"]["unloaded.py"]["summary"]["covered_branches"] == 0


@pytest.mark.skipif(shutil.which("node") is None or not (REPO / "node_modules/istanbul-lib-instrument").is_dir(),
                    reason="Install the pinned Node and npm dependencies to test JavaScript coverage")
def test_javascript_report_counts_unloaded_file_and_child_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("loaded.mjs", "unloaded.mjs"):
        (tmp_path / name).write_text("export function choose(flag) {\n  if (flag) return 1;\n  return 2;\n}\n", encoding="utf-8")
    raw_dir = tmp_path / "js-raw"
    raw_dir.mkdir()
    scope = tmp_path / "inventory.json"
    files = ["loaded.mjs", "unloaded.mjs"]
    scope.write_text(json.dumps({"javascript": files}), encoding="utf-8")
    support = REPO / "tests/support"
    monkeypatch.setenv("NODE_OPTIONS", f"--import={(support / 'coverage-hook.mjs').as_uri()}")
    monkeypatch.setenv("AF_COVERAGE_INVENTORY", str(scope))
    monkeypatch.setenv("AF_COVERAGE_JS_RAW", str(raw_dir))
    monkeypatch.setenv("AF_COVERAGE_ROOT", str(tmp_path))
    env = {key: os.environ[key] for key in ("PATH", "SystemRoot") if key in os.environ}
    env["NODE_OPTIONS"] = ""
    program = "import { choose } from './loaded.mjs'; if (choose(true) !== 1) process.exit(1);"
    subprocess.run(["node", "--input-type=module", "--eval", program], check=True, env=env)
    fixture = tmp_path / "runner.test.mjs"
    fixture.write_text("import {test} from 'node:test'; test('count a branch', async () => { "
                       "const {choose} = await import('./loaded.mjs'); "
                       "if (choose(true) !== 1) throw Error('wrong decision'); });\n", encoding="utf-8")
    assert run_javascript_tests(env, str(fixture)) == 0
    assert len(list(raw_dir.glob("*.json"))) == 2
    report = javascript_report(tmp_path, files)
    assert report["files"]["loaded.mjs"]["branches"] == {"covered": 1, "total": 2}
    assert report["files"]["unloaded.mjs"]["branches"] == {"covered": 0, "total": 2}


def test_python_union_combines_branch_hits_but_counts_untouched_sources(tmp_path, monkeypatch):
    from coverage import CoverageData

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True, cwd=tmp_path)
    test_source = tmp_path / "tests/test_choice.py"
    test_source.parent.mkdir()
    test_source.write_text("def test_choice(): pass\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n', encoding="utf-8")
    studio = tmp_path / "viewer/studio.html"
    studio.parent.mkdir()
    studio.write_text("<main>Original Studio</main>", encoding="utf-8")
    recipes = tmp_path / "image_to_3dlab/prompt_recipes.json"
    recipes.parent.mkdir()
    recipes.write_text('{"recipe":"practical"}', encoding="utf-8")
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nbranch = True\n", encoding="utf-8")
    files = {"python": ["choice.py", "untouched.py"], "javascript": []}
    source = "def choose(flag):\n    if flag:\n        return 1\n    return 2\n"
    for name in files["python"]:
        (tmp_path / name).write_text(source, encoding="utf-8")

    inputs = []
    for kind, branch in (("portable", 3), ("mlx-unit", 4)):
        folder = tmp_path / kind
        folder.mkdir()
        measured_root = tmp_path / f"{kind}-machine"
        data = CoverageData(basename=str(folder / ".coverage"))
        data.add_arcs({str(measured_root / "choice.py"): [(2, branch)]})
        data.write()
        if kind == "portable":
            (folder / "javascript.json").write_text("{}")
        measure_coverage.write_manifest(folder, kind, measured_root,
                                        measure_coverage.source_fingerprint(tmp_path, files, config), complete=True)
        inputs.append(folder)

    mlx_manifest = inputs[1] / "run.json"
    metadata = json.loads(mlx_manifest.read_text(encoding="utf-8"))
    metadata["python"] = [3, 12]
    mlx_manifest.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "merged"
    output.mkdir()
    report = measure_coverage.merge_python(output, files, config, *inputs)
    assert report["files"]["choice.py"]["summary"]["covered_branches"] == 2
    assert report["files"]["untouched.py"]["summary"]["num_branches"] == 2
    assert report["files"]["untouched.py"]["summary"]["covered_branches"] == 0

    test_source.write_text("def test_replacement(): pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sources differ"):
        measure_coverage.merge_python(output, files, config, *inputs)
    test_source.write_text("def test_choice(): pass\n", encoding="utf-8")

    lock.write_text('version = 2\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sources differ"):
        measure_coverage.merge_python(output, files, config, *inputs)
    lock.write_text('version = 1\n', encoding="utf-8")

    studio.write_text("<main>New Studio</main>", encoding="utf-8")
    with pytest.raises(ValueError, match="sources differ"):
        measure_coverage.merge_python(output, files, config, *inputs)
    studio.write_text("<main>Original Studio</main>", encoding="utf-8")

    recipes.write_text('{"recipe":"impressionist"}', encoding="utf-8")
    with pytest.raises(ValueError, match="sources differ"):
        measure_coverage.merge_python(output, files, config, *inputs)
    recipes.write_text('{"recipe":"practical"}', encoding="utf-8")

    with monkeypatch.context() as checkout:
        checkout.setenv("GITHUB_SHA", "another-checkout")
        with pytest.raises(ValueError, match="GitHub checkout differs"):
            measure_coverage.merge_python(output, files, config, *inputs)

    (tmp_path / "choice.py").write_text(source + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sources differ"):
        measure_coverage.merge_python(output, files, config, *inputs)
    (tmp_path / "choice.py").write_text(source, encoding="utf-8")

    manifest = inputs[1] / "run.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["complete"] = False
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="complete MLX unit run"):
        measure_coverage.merge_python(output, files, config, *inputs)

    (inputs[1] / "run.json").unlink()
    with pytest.raises(ValueError, match="missing MLX unit coverage"):
        measure_coverage.merge_python(output, files, config, *inputs)


@pytest.mark.parametrize(("launcher", "after"), [
    ("install.sh", "updated"),
    ("install.ps1", "updated"),
    ("lab", "updated"),
    ("lab.cmd", "updated"),
    ("lab", None),
    ("another-launcher", "new"),
])
def test_merge_rejects_changed_root_inputs_with_same_code_inventory(tmp_path, monkeypatch, launcher, after):
    from coverage import CoverageData

    contents = {
        ".gitignore": "/machine-only\n",
        "pyproject.toml": "[tool.coverage.run]\nbranch = true\n",
        "choose.py": "def choose(flag):\n    if flag:\n        return 1\n    return 2\n",
        "choose.js": "export function choose(flag) { return flag ? 1 : 2; }\n",
        **dict.fromkeys(("install.sh", "install.ps1", "lab", "lab.cmd"), "original"),
    }
    portable_checkout, mlx_checkout, checkout = [tmp_path / name for name in ("portable", "mlx", "current")]
    for root in (portable_checkout, mlx_checkout, checkout):
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        for name, content in contents.items():
            (root / name).write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", *contents], cwd=root, check=True)

    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(measure_coverage, "REPO", checkout)
    monkeypatch.chdir(checkout)
    files = inventory(checkout)
    portable, mlx = checkout / "arbitrary-portable-report", checkout / "arbitrary-mlx-report"
    for kind, root, output, branch in (
        ("portable", portable_checkout, portable, 3),
        ("mlx-unit", mlx_checkout, mlx, 4),
    ):
        output.mkdir()
        data = CoverageData(basename=str(output / ".coverage"))
        data.add_arcs({str(root / "choose.py"): [(-1, 1), (1, -1), (-1, 2), (2, branch), (branch, -1)]})
        data.write()
        if kind == "portable":
            (portable / "javascript.json").write_text(json.dumps({
                "files": {"choose.js": {}},
                "totals": {"lines": {"covered": 1, "total": 1}, "branches": {"covered": 2, "total": 2}},
            }), encoding="utf-8")
        stamp = measure_coverage.source_fingerprint(root, files, root / "pyproject.toml")
        measure_coverage.write_manifest(output, kind, root, stamp, complete=True)

    (checkout / "machine-only").write_text("ignored local data", encoding="utf-8")
    verified = checkout / "arbitrary-combined-report"
    totals = measure_coverage.merge(verified, portable, mlx)
    assert totals["Python"] == {"lines": (4, 4), "branches": (2, 2)}
    assert len(list((verified / "badges").glob("*.svg"))) == 4

    changed = checkout / launcher
    if after is None:
        changed.unlink()
    else:
        changed.write_text(after, encoding="utf-8")
    assert inventory(checkout) == files
    stale = checkout / "stale-report"
    with pytest.raises(ValueError, match="coverage sources differ"):
        measure_coverage.merge(stale, portable, mlx)
    assert not (stale / "badges").exists()


@pytest.mark.parametrize(("case", "accepted"), [
    ("<testcase/>", True),
    ("<testcase><skipped/></testcase>", False),
    ("<testcase><failure/></testcase>", False),
    ("", False),
])
def test_mlx_requires_real_passing_cases(tmp_path, case, accepted):
    junit = tmp_path / "junit.xml"
    junit.write_text(f"<testsuites><testsuite>{case}</testsuite></testsuites>", encoding="utf-8")
    assert measure_coverage.mlx_cases_passed(junit) is accepted


@pytest.mark.parametrize(("args", "message"), [
    (["--", "--run-integration"], "run integration tests separately"),
    (["--mlx-unit", "--enforce"], "Apply --enforce"),
    (["--mlx-unit", "--", "-k", "pick"], "Only the portable runner accepts"),
    (["--merge", "portable", "mlx", "--", "-k", "pick"], "Only the portable runner accepts"),
])
def test_report_cli_rejects_incompatible_options(tmp_path, monkeypatch, capsys, args, message):
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", str(tmp_path / "report"), *args])
    with pytest.raises(SystemExit) as stopped:
        measure_coverage.main()
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_report_cli_dispatches_numerical_only(tmp_path, monkeypatch):
    report = tmp_path / "mlx"
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", str(report), "--mlx-unit"])
    called = []
    monkeypatch.setattr(measure_coverage, "run_mlx_unit", lambda output: called.append(output) or 3)
    assert measure_coverage.main() == 3
    assert called == [report.resolve()]


@pytest.mark.parametrize(("option", "status"), [(False, 0), (True, 1)])
def test_report_only_setting_applies_to_merged_cli(tmp_path, monkeypatch, capsys, option, status):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measure_coverage, "REPORT_ONLY", not option)
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", "out", "--merge", "first", "second"])
    def merge(output, portable, mlx):
        assert (output, portable, mlx) == tuple(tmp_path / name for name in ["out", "first", "second"])
        return {"Python": {"lines": (0, 0), "branches": (96, 100)},
                "JavaScript": {"lines": (97, 100), "branches": (98, 100)}}
    monkeypatch.setattr(measure_coverage, "merge", merge)
    assert measure_coverage.main() == status
    assert "Combined macOS unit coverage" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", "portable"])
    monkeypatch.setattr(measure_coverage, "run", lambda *_: (0, {"Python": {"lines": (95, 100)}}))
    assert measure_coverage.main() == 0


@pytest.mark.parametrize("shortfall", [
    None, ("Python", "lines"), ("Python", "branches"),
    ("JavaScript", "lines"), ("JavaScript", "branches"),
])
def test_default_combined_check_requires_all_four_exact_ratios(tmp_path, monkeypatch, shortfall):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["measure_coverage.py", "out", "--merge", "portable", "mlx"])
    counts = {lang: {"lines": (96, 100), "branches": (96, 100)} for lang in ("Python", "JavaScript")}
    if shortfall:
        language, metric = shortfall
        counts[language][metric] = (9599999, 10000000)
    monkeypatch.setattr(measure_coverage, "merge", lambda *_: counts)
    assert measure_coverage.main() == int(shortfall is not None)


@pytest.mark.parametrize("below_target", [False, True])
def test_combined_presentation_summarizes_exact_counts_and_only_badges_passes(tmp_path, monkeypatch, below_target):
    summary = tmp_path / "summary.md"
    summary.write_text("Earlier summary\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    counts = {
        "Python": {"lines": (40116, 40673), "branches": (9599999, 10000000) if below_target else (11505, 11804)},
        "JavaScript": {"lines": (3625, 3629), "branches": (3147, 3249)},
    }

    measure_coverage.write_coverage_presentation(tmp_path, counts)

    markdown = summary.read_text(encoding="utf-8")
    assert markdown.startswith("Earlier summary\n\n## Combined unit coverage\n")
    assert "| Python | Lines | 40,116 | 40,673 | 98.6305% | 96% | Pass |" in markdown
    assert "| JavaScript | Branches | 3,147 | 3,249 | 96.8606% | 96% | Pass |" in markdown
    badges = tmp_path / "badges"
    if below_target:
        assert "| Python | Branches | 9,599,999 | 10,000,000 | 96.0000% | 96% | Below target |" in markdown
        assert not badges.exists()
    else:
        assert sorted(path.name for path in badges.iterdir()) == [
            "javascript-branches.svg", "javascript-lines.svg", "python-branches.svg", "python-lines.svg",
        ]
        assert {path.name: ElementTree.parse(path).getroot().attrib["aria-label"] for path in badges.iterdir()} == {
            "python-lines.svg": "Python lines: 98.63%",
            "python-branches.svg": "Python branches: 97.47%",
            "javascript-lines.svg": "JavaScript lines: 99.89%",
            "javascript-branches.svg": "JavaScript branches: 96.86%",
        }
        svg = ElementTree.parse(badges / "python-lines.svg").getroot()
        assert "40,116 of 40,673" in "".join(svg.itertext())


@pytest.mark.parametrize(("actual", "message"), [("v1.0.0", "Use Node"), ("v26.13.0", "Run npm ci")])
def test_portable_reports_toolchain_before_running_tests(tmp_path, monkeypatch, actual, message):
    monkeypatch.setattr(measure_coverage, "REPO", tmp_path)
    (tmp_path / ".node-version").write_text("26.13.0\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: actual + "\n")
    with pytest.raises(RuntimeError, match=message):
        measure_coverage.run(tmp_path / "out", [])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("status", "case", "expected"), [
    (0, "<testcase/>", 0), (0, "<testcase><skipped/></testcase>", 1), (5, "", 5),
])
def test_numerical_runner_stamps_only_passing_cases(tmp_path, monkeypatch, status, case, expected):
    from coverage import CoverageData

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True, cwd=tmp_path)
    monkeypatch.setattr(measure_coverage, "REPO", tmp_path)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k deliberate_filter")
    (tmp_path / "pyproject.toml").write_text("[tool.coverage.run]\nbranch = true\n")
    source = tmp_path / "example.py"
    source.write_text("value = 1\n")
    output = tmp_path / "results"

    def execute(command, *, check, env):
        assert "--run-mlx-unit" in command and "--run-mlx-metal" not in command
        assert env["AF_OFFLINE_TESTS"] == "1"
        assert "PYTEST_ADDOPTS" not in env
        assert not check
        Path(command[-1]).write_text(f"<testsuite>{case}</testsuite>")
        data = CoverageData(basename=env["COVERAGE_FILE"], suffix="fixture")
        data.add_arcs({str(source): [(-1, 1), (1, -1)]})
        data.write()
        return subprocess.CompletedProcess(command, status)

    real_run = subprocess.run
    def route(command, *args, **kwargs):
        return real_run(command, *args, **kwargs) if command[0] == "git" else execute(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", route)
    assert measure_coverage.run_mlx_unit(output) == expected
    assert (output / "run.json").exists() == (status == expected == 0)


@pytest.mark.parametrize("windows", [False, True])
def test_fingerprint_tracks_executable_permissions(tmp_path, monkeypatch, windows):
    if not windows and os.name == "nt":
        pytest.skip("Windows uses Git executable metadata")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    launcher = tmp_path / "lab"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o644)
    subprocess.run(["git", "add", "lab"], cwd=tmp_path, check=True)
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nbranch = True\n")
    files = {"python": [], "javascript": []}
    if windows:
        monkeypatch.setattr(measure_coverage, "os", SimpleNamespace(name="nt", fsdecode=os.fsdecode))
    before = measure_coverage.source_fingerprint(tmp_path, files, config)
    if windows:
        subprocess.run(["git", "update-index", "--chmod=+x", "lab"], cwd=tmp_path, check=True)
    else:
        # The index is unchanged: local chmod must invalidate the collected run.
        launcher.chmod(0o755)
    after = measure_coverage.source_fingerprint(tmp_path, files, config)
    assert after != before
