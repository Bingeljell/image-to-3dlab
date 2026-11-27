"""Run the shipped coverage CLI against a miniature checkout."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))

import measure_coverage

if __name__ == "__main__":
    measure_coverage.REPO = Path(sys.argv.pop(1)).resolve()
    raise SystemExit(measure_coverage.main())
