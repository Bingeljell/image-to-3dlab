#!/usr/bin/env bash
# The pinned Mac TRELLIS installer is the Python one. It asks before it fetches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${ROOT}/scripts/bootstrap_trellis_space_macos.py"
if command -v python3 >/dev/null 2>&1; then
  exec python3 "${INSTALLER}" "$@"
fi
echo "Run: python3 scripts/bootstrap_trellis_space_macos.py" >&2
exit 1
