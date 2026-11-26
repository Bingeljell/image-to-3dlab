# Coverage and unit tests

Install Python dependencies with `uv sync --frozen`. Use the Node version in
`.node-version`, then run `npm ci`. Run Python tests with `uv run pytest -q`
and JavaScript tests with `npm test`. JavaScript tests go in
`tests/js/*.test.js` and can use jsdom for browser events and markup.

Run both suites and make coverage reports with:

```sh
uv run --frozen python scripts/measure_coverage.py coverage-report
```

Use a new output directory each time. To select Python tests, pass pytest
options after `--`. The JavaScript suite still runs in full.

- `inventory.json` lists versionable source files, including the command-line
  entry point, reusable scripts, both Hunyuan subprojects and the mesh-winding
  demo. It excludes tests and the five bundled Three.js files. Git ignores
  installed environments, local assets and backend checkouts.
- `python.json` includes coverage from the root unit suite and Python child
  processes started with its coverage environment. Explicit file reporting
  includes files the tests never import. The report lists structural exclusions
  such as type-checker-only code. Legacy `no cover` and `no branch` comments
  do not remove executable code from coverage.
- `javascript.json` includes Node tests and Node processes launched by pytest.
  Istanbul parses all source files before merging the execution counts, so
  branches in unimported files count in the total. The report lists missed lines,
  branch locations and hit counts.

Shipped Python helpers and workers are measured as source files. For example,
the live Blender bridge sends the same worker source that the tests import.
Thin dispatch, import, configuration and API replacement snippets sent to
external processes or inserted into third-party code are outside that
file-based measurement. Building the string earns coverage for its enclosing
Python code only. It provides no execution credit for the statements inside.

The targets are 96% of executable lines and 96% of branch outcomes for each
language. The check uses integer counts. Linux publishes its portable report.
On macOS, CI combines Python coverage from the portable suite and a separate
MLX numerical suite. JavaScript coverage comes from the portable macOS run.
Windows runs its own Python test job. The combined CI check fails when any of
the four ratios falls below 96%. Test failures and inconsistent inventories
also fail the check.

When both suites produce valid reports, the combined CI job summary shows the
covered and total counts, percentages, and result for all four measures.
Each README badge links to the CI workflow, where the combined report is
available as an artifact. A successful run on `main` publishes the four badge
images to the `coverage-badges` branch. The badges show the last published result
and appear after the first successful publication. Badges round percentages
to two decimal places. The check and summary status use the exact counts.

## Protecting main on GitHub

The **Merge gate** job requires successful lint, Linux, macOS, Windows, Apple
MLX, combined coverage, real Blender and real browser jobs. It runs on pull requests and merge-queue
checks. Badge publication on main also waits for the gate.

An administrator can activate the gate for `main` in **Settings → Rules →
Rulesets**. First, run the updated workflow on a pull request so GitHub can
offer **Merge gate** as a status check. In the active main ruleset, require a
pull request and require **Merge gate** from **GitHub Actions**. Enable
**Require branches to be up to date before merging** and leave the bypass
list empty. Keep the rules that prevent deletion and force pushes. The gate
includes Windows, so its standalone status check can be removed.

The [proposed main ruleset](main-merge-ruleset.json) shows the API fields,
including GitHub Actions integration ID `15368`. Compare it to existing rules
before applying it and preserve any additional protections. Check the active
rules after saving with `gh api repos/OWNER/REPO/rules/branches/main`.

## Apple MLX numerical tests

The tests in `tests/mlx/` exercise tiny synthetic arrays with Apple's MLX
library on the CPU. They use the same offline test guards as the portable suite. Install
the library in a separate Python 3.12 environment on an Apple Silicon Mac.

```sh
uv venv --python 3.12 .venv-mlx-unit
uv pip install --python .venv-mlx-unit/bin/python -r tests/mlx/requirements.txt
uv run --frozen python scripts/measure_coverage.py coverage-report/portable
.venv-mlx-unit/bin/python scripts/measure_coverage.py coverage-report/mlx --mlx-unit
uv run --frozen python scripts/measure_coverage.py coverage-report/combined \
  --merge coverage-report/portable coverage-report/mlx
```

To run the numerical tests without coverage, invoke `.venv-mlx-unit/bin/python
-m pytest tests/mlx --run-mlx-unit`. The portable suite leaves this directory
out during collection. On a Metal-capable Mac, add `--run-mlx-metal -m mlx_metal`
to run the separate GPU checks. A coverage merge requires both runs to pass, the
numerical suite to run without skips, and source, tests, runner settings and
dependency files to match the current checkout. The comparison also includes
root files that Git tracks or would add, such as the installers, launchers and
README. Git-ignored local files stay out. CI also checks the checkout commit.
Both runs use the same coverage.py version and record their Python
versions. The merge counts the union of Python execution data and keeps untested
files, including GPU code, in the denominator. It rejects portable runs that select only some tests.
The merge enforces all four targets locally and in CI. Portable CI reports
stay informational. Use `--enforce` to check the targets on a complete
portable run locally.

Root unit tests use loopback servers and small local fixtures. The test guards
check standard Python and Node networking and process APIs. Treat native
extensions and in-process model calls as test boundaries and use fixtures
there. Run root integration tests with `uv run pytest --run-integration`.
Hunyuan shape and paint comparison suites live under their own subproject
`tests/` directories.

## Real Blender and browser checks

The separate [real-tool suites](real-tool-tests.md) exercise Blender and Studio
with small generated models. Both must pass the Merge gate. They do not load
model weights or contribute to the unit coverage percentages.
