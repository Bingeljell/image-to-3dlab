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

The combined report requires at least 96% of executable lines and branch
outcomes for each language. It rejects stale reports or different source files.

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

