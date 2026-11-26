"""Run the offline unit suites and report line and branch coverage separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import coverage

REPO = Path(__file__).resolve().parents[1]
SUPPORT = REPO / "tests/support"
TARGET = 96
REPORT_ONLY = False
BUNDLED_JAVASCRIPT = {
    "viewer/vendor/controls/OrbitControls.js",
    "viewer/vendor/environments/RoomEnvironment.js",
    "viewer/vendor/loaders/GLTFLoader.js",
    "viewer/vendor/three.module.js",
    "viewer/vendor/utils/BufferGeometryUtils.js",
}
JAVASCRIPT_EXTENSIONS = {".js", ".mjs", ".cjs"}
NESTED_RUNNER_INPUTS = {".github/workflows/pre-commit.yml", "docs/main-merge-ruleset.json"}
RESOURCE_ROOTS = {"hunyuan_mlx", "image_to_3dlab", "manifests", "rigs", "scripts", "understanding_3d", "viewer", "workflows"}
REPORT_OPTIONS = {"-q", "-ra", "--tb=short", "--color=yes"}


class InlineScripts(HTMLParser):
    """Find executable inline scripts, excluding data blocks and import maps."""

    def __init__(self):
        super().__init__()
        self.inline = False
        self.executable = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            options = dict(attrs)
            kind = (options.get("type") or "").strip().lower().split(";", 1)[0]
            self.inline = "src" not in options and (
                kind in {"", "module"} or kind.endswith(("javascript", "ecmascript"))
            )

    def handle_endtag(self, tag):
        if tag == "script":
            self.inline = False

    def handle_data(self, data):
        if self.inline and data.strip():
            self.executable = True


def versionable_paths(root: Path) -> list[str]:
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    )
    paths = {os.fsdecode(name) for name in listed.split(b"\0") if name}
    return sorted(path for path in paths if (root / path).is_file())


def inventory(root: Path, paths: list[str] | None = None) -> dict[str, list[str]]:
    """Use Git's versionable files so local assets and installed code stay out."""
    if paths is None:
        paths = versionable_paths(root)
    files: dict[str, list[str]] = {"python": [], "javascript": []}
    for file in sorted(set(paths)):
        path = PurePosixPath(file)
        if "tests" in path.parts[:-1]:
            continue
        if path.suffix == ".py":
            files["python"].append(file)
        elif path.suffix in JAVASCRIPT_EXTENSIONS and file not in BUNDLED_JAVASCRIPT:
            files["javascript"].append(file)
        elif path.suffix == ".html":
            parser = InlineScripts()
            parser.feed((root / file).read_text(encoding="utf-8"))
            if parser.executable:
                raise ValueError(f"Move executable JavaScript in {file} to a source file so it can be measured")
    return files


def require_inventory(expected: list[str], actual: dict) -> None:
    missing = set(expected) - actual.keys()
    unexpected = actual.keys() - set(expected)
    if missing or unexpected:
        raise ValueError(f"Coverage inventory mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


def threshold_met(covered: int, total: int) -> bool:
    return total > 0 and covered * 100 >= total * TARGET


def python_report(output: Path, files: list[str], config: Path, *, combine: bool = True) -> dict:
    destination = output / "python.json"
    data_file = output / ".coverage"
    measured = coverage.Coverage(data_file=data_file, config_file=config)
    if combine:
        measured.combine(strict=True, keep=True)
        measured.save()
    else:
        measured.load()
    measured.json_report(morfs=files, outfile=str(destination))
    report = json.loads(destination.read_text(encoding="utf-8"))
    if os.sep != "/":
        report["files"] = {name.replace(os.sep, "/"): data for name, data in report["files"].items()}
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    require_inventory(files, report["files"])
    return report


def source_fingerprint(root: Path, files: dict[str, list[str]], config: Path) -> str:
    """Hash root files and nested code, tests, resources and runner inputs."""
    digest = hashlib.sha256(config.read_bytes())
    inputs = set(files["python"] + files["javascript"])
    for path in versionable_paths(root):
        relative = PurePosixPath(path)
        if (len(relative.parts) == 1 or path in NESTED_RUNNER_INPUTS
                or "tests" in relative.parts[:-1] or relative.parts[0] in RESOURCE_ROOTS):
            inputs.add(path)
    for path in sorted(inputs):
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((root / path).read_bytes()).digest())
    return digest.hexdigest()


def write_manifest(output: Path, kind: str, root: Path, fingerprint: str, *, complete: bool) -> None:
    manifest = {
        "kind": kind,
        "root": str(root),
        "source": fingerprint,
        "complete": complete,
        "python": list(sys.version_info[:2]),
        "coverage": coverage.__version__,
        "ci_checkout": os.environ.get("GITHUB_SHA"),
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_coverage_input(folder: Path, kind: str, fingerprint: str) -> tuple[Path, coverage.CoverageData]:
    label = "MLX unit" if kind == "mlx-unit" else "portable unit"
    try:
        manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        data_path = folder / ".coverage"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} coverage in {folder}") from exc
    if manifest["kind"] != kind or not manifest["complete"]:
        raise ValueError(f"{folder} does not contain a complete {label} run")
    if manifest["source"] != fingerprint:
        raise ValueError(f"coverage sources differ for {folder} (code, tests or configuration changed)")
    if (checkout := os.environ.get("GITHUB_SHA")) and manifest.get("ci_checkout") != checkout:
        raise ValueError(f"GitHub checkout differs for {folder}")
    if manifest["coverage"] != coverage.__version__:
        raise ValueError(f"coverage version differs for {folder}")
    data = coverage.CoverageData(basename=str(data_path))
    data.read()
    if not data.has_arcs() or not data.measured_files():
        raise ValueError(f"no Python branch data in {folder}")
    return Path(manifest["root"]), data


def merge_python(output: Path, files: dict[str, list[str]], config: Path, portable: Path, mlx: Path) -> dict:
    root = Path.cwd()
    fingerprint = source_fingerprint(root, files, config)
    merged = coverage.CoverageData(basename=str(output / ".coverage"))
    for kind, folder in (("portable", portable), ("mlx-unit", mlx)):
        origin, data = load_coverage_input(folder, kind, fingerprint)
        # Both jobs can have different absolute checkout paths. Copy arcs for the
        # declared files only and let coverage.py compute the shared denominator.
        arcs = {str(root / file): data.arcs(str(origin / file)) or [] for file in files["python"]}
        if not any(arcs.values()):
            raise ValueError(f"no maintained Python code was measured in {folder}")
        merged.add_arcs(arcs)
    merged.write()
    return python_report(output, files["python"], config, combine=False)


def mlx_cases_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    cases = list(ElementTree.parse(path).iter("testcase"))
    return bool(cases) and not any(child.tag in {"skipped", "failure", "error"} for case in cases for child in case)


def javascript_report(output: Path, files: list[str]) -> dict:
    destination = output / "javascript.json"
    result = subprocess.check_output([
        "node", str(SUPPORT / "coverage-report.mjs"), str(Path.cwd()), str(output / "js-raw"),
        *files,
    ])
    destination.write_bytes(result)
    report = json.loads(result)
    require_inventory(files, report["files"])
    return report


def run_javascript_tests(env: dict[str, str], test_pattern: str = "tests/js/**/*.test.js") -> int:
    return subprocess.run(["node", "--import", (SUPPORT / "browser-imports.mjs").as_uri(),
                           "--test", test_pattern], check=False, env=env).returncode


def offline_environment(output: Path, config: Path) -> dict[str, str]:
    support_path = os.pathsep.join(filter(None, [str(SUPPORT), os.environ.get("PYTHONPATH")]))
    return {
        **os.environ,
        "COVERAGE_FILE": str(output / ".coverage"),
        "COVERAGE_RCFILE": str(config),
        "AF_OFFLINE_TESTS": "1",
        "PYTHONPATH": support_path,
    }


def write_unchanged_run(output: Path, kind: str, files: dict[str, list[str]], config: Path,
                        fingerprint: str, *, complete: bool) -> None:
    if inventory(REPO) != files or source_fingerprint(REPO, files, config) != fingerprint:
        raise ValueError("coverage sources differ from the start of the run")
    write_manifest(output, kind, REPO, fingerprint, complete=complete)


def coverage_totals(py: dict, js: dict) -> dict[str, dict[str, tuple[int, int]]]:
    return {
        "Python": {
            "lines": (py["covered_lines"], py["num_statements"]),
            "branches": (py["covered_branches"], py["num_branches"]),
        },
        "JavaScript": {metric: (counts["covered"], counts["total"]) for metric, counts in js.items()},
    }


def coverage_percent(covered: int, total: int, digits: int) -> str:
    """Format a percentage for display. Enforcement uses the original counts."""
    percent = covered * 100 / total if total else 0
    return f"{percent:.{digits}f}%"


def coverage_badge(label: str, covered: int, total: int) -> str:
    value = coverage_percent(covered, total, 2)
    label_width = 7 * len(label) + 12
    value_width = 7 * len(value) + 12
    width = label_width + value_width
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20"
    role="img" aria-label="{label}: {value}">
  <title>{label}: {covered:,} of {total:,} covered ({value})</title>
  <rect width="{width}" height="20" rx="3" fill="#276738"/>
  <rect width="{label_width}" height="20" rx="3" fill="#454545"/>
  <rect x="{label_width - 3}" width="3" height="20" fill="#454545"/>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,sans-serif" font-size="11">
    <text x="{label_width / 2}" y="14">{label}</text>
    <text x="{label_width + value_width / 2}" y="14">{value}</text>
  </g>
</svg>
"""


def write_coverage_presentation(output: Path, totals: dict[str, dict[str, tuple[int, int]]]) -> None:
    rows = [
        "",
        "## Combined unit coverage",
        "",
        "| Language | Measure | Covered | Total | Coverage | Target | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    badges = {}
    for language in ("Python", "JavaScript"):
        for metric in ("lines", "branches"):
            covered, total = totals[language][metric]
            percent = coverage_percent(covered, total, 4)
            passes = threshold_met(covered, total)
            status = "Pass" if passes else "Below target"
            rows.append(f"| {language} | {metric.capitalize()} | {covered:,} | {total:,} | {percent} | {TARGET}% | {status} |")
            if passes:
                badges[f"{language.lower()}-{metric}.svg"] = coverage_badge(f"{language} {metric}", covered, total)

    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(rows) + "\n")
    if len(badges) == 4:
        destination = output / "badges"
        destination.mkdir()
        for name, badge in badges.items():
            (destination / name).write_text(badge, encoding="utf-8")


def full_portable_suite(pytest_args: list[str]) -> bool:
    return not os.environ.get("PYTEST_ADDOPTS") and set(pytest_args) <= REPORT_OPTIONS


def run(output: Path, pytest_args: list[str]) -> tuple[int, dict[str, dict[str, tuple[int, int]]]]:
    required_node = (REPO / ".node-version").read_text(encoding="utf-8").strip()
    actual_node = subprocess.check_output(["node", "--version"], text=True).strip()
    if actual_node != f"v{required_node}":
        raise RuntimeError(f"Use Node {required_node} for coverage (found {actual_node})")
    if not (REPO / "node_modules/istanbul-lib-instrument").is_dir():
        raise RuntimeError("Run npm ci before measuring coverage")
    # Use a new data directory per run, so stale subprocess hits can never merge in.
    output.mkdir(parents=True, exist_ok=False)
    files = inventory(REPO)
    scope = output / "inventory.json"
    scope.write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    raw = output / "js-raw"
    raw.mkdir()
    config = REPO / "pyproject.toml"
    fingerprint = source_fingerprint(REPO, files, config)
    node_preloads = [
        (SUPPORT / name).as_uri() for name in ("coverage-hook.mjs", "offline.mjs")
    ]
    env = {
        **offline_environment(output, config),
        "AF_COVERAGE_INVENTORY": str(scope),
        "AF_COVERAGE_JS_RAW": str(raw),
        "AF_COVERAGE_ROOT": str(REPO),
        "NODE_OPTIONS": " ".join(f"--import={uri}" for uri in node_preloads),
    }
    python_tests = subprocess.run([sys.executable, "-m", "coverage", "run", "-m", "pytest",
                                   "tests", *pytest_args], check=False, env=env).returncode
    javascript_tests = run_javascript_tests(env)
    py = python_report(output, files["python"], config)["totals"]
    js = javascript_report(output, files["javascript"])["totals"]
    status = python_tests or javascript_tests
    complete = not status and full_portable_suite(pytest_args)
    write_unchanged_run(output, "portable", files, config, fingerprint, complete=complete)
    return status, coverage_totals(py, js)


def run_mlx_unit(output: Path) -> int:
    """Run the optional suite with an already installed MLX test environment."""
    output.mkdir(parents=True, exist_ok=False)
    files = inventory(REPO)
    config = REPO / "pyproject.toml"
    fingerprint = source_fingerprint(REPO, files, config)
    junit = output / "pytest.xml"
    env = offline_environment(output, config)
    env.pop("PYTEST_ADDOPTS", None)
    status = subprocess.run([
        sys.executable, "-m", "coverage", "run", "-m", "pytest", "tests/mlx",
        "--run-mlx-unit", "-q", "--junitxml", str(junit),
    ], check=False, env=env).returncode
    complete = not status and mlx_cases_passed(junit)
    if not complete:
        print("MLX coverage requires a passing, nonempty numerical suite with no skips.", file=sys.stderr)
        return status or 1
    python_report(output, files["python"], config)
    write_unchanged_run(output, "mlx-unit", files, config, fingerprint, complete=True)
    return 0


def merge(output: Path, portable: Path, mlx: Path) -> dict[str, dict[str, tuple[int, int]]]:
    output.mkdir(parents=True, exist_ok=False)
    files = inventory(REPO)
    config = REPO / "pyproject.toml"
    python = merge_python(output, files, config, portable, mlx)
    javascript = json.loads((portable / "javascript.json").read_text(encoding="utf-8"))
    require_inventory(files["javascript"], javascript["files"])
    (output / "javascript.json").write_text(json.dumps(javascript, indent=2) + "\n", encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    totals = coverage_totals(python["totals"], javascript["totals"])
    write_coverage_presentation(output, totals)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the two reports and inventory")
    parser.add_argument("--enforce", action="store_true", help="Fail when any score is below 96%%")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mlx-unit", action="store_true", help="Measure only the optional MLX unit suite")
    mode.add_argument("--merge", nargs=2, metavar=("PORTABLE", "MLX"), type=Path, help="Combine two complete runs")
    args, pytest_args = parser.parse_known_args()
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if any(flag in (*pytest_args, *os.environ.get("PYTEST_ADDOPTS", "").split())
           for flag in ("--run-integration", "--run-mlx-unit", "--run-mlx-metal")):
        parser.error("Use --mlx-unit for numerical coverage and run integration tests separately.")
    if (args.mlx_unit or args.merge) and pytest_args:
        parser.error("Only the portable runner accepts pytest arguments.")
    if args.mlx_unit and args.enforce:
        parser.error("Apply --enforce to the combined report.")
    if args.enforce and not args.merge and not full_portable_suite(pytest_args):
        parser.error("--enforce requires the full portable suite or a combined report.")
    output = args.output.resolve()
    inputs = [path.resolve() for path in args.merge] if args.merge else []
    os.chdir(REPO)
    if args.mlx_unit:
        return run_mlx_unit(output)
    tests_status, results = (0, merge(output, *inputs)) if inputs else run(output, pytest_args)
    print("Combined macOS unit coverage" if inputs else "Portable unit coverage", flush=True)
    passed = True
    for language, metrics in results.items():
        for metric, (covered, total) in metrics.items():
            meets = threshold_met(covered, total)
            passed &= meets
            percent = coverage_percent(covered, total, 4)
            status = "meets target" if meets else "below target"
            print(f"{language} {metric}: {covered}/{total} ({percent}) {status}", flush=True)
    return tests_status or int(not passed and (args.enforce or (bool(inputs) and not REPORT_ONLY)))


if __name__ == "__main__":
    raise SystemExit(main())
