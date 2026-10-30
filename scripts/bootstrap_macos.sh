#!/usr/bin/env bash
# The installer that asks, and that names the weights, is the Python one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${ROOT}/scripts/bootstrap_sf3d.py"
if command -v python3 >/dev/null 2>&1; then
  exec python3 "${INSTALLER}" "$@"
fi
echo "Run: python3 scripts/bootstrap_sf3d.py" >&2
exit 1
