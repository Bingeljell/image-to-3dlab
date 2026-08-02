#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRELLIS_DIR="${PROJECT_DIR}/vendor/trellis-mac"

if [[ ! -d "${TRELLIS_DIR}/.git" ]]; then
  mkdir -p "${PROJECT_DIR}/vendor"
  git clone https://github.com/shivampkumar/trellis-mac.git "${TRELLIS_DIR}"
fi

pushd "${TRELLIS_DIR}" >/dev/null
if [[ -d /Applications/Xcode.app/Contents/Developer ]] \
  && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -f metal >/dev/null 2>&1; then
  echo "Xcode Metal compiler detected; installing Metal acceleration backends."
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer bash setup.sh
else
  echo "Xcode Metal compiler not found; installing the slower CPU bake fallback."
  SKIP_METAL=1 bash setup.sh
fi
popd >/dev/null

python3 "${PROJECT_DIR}/scripts/patch_trellis_no_bria.py"
echo "TRELLIS setup complete; BRIA background removal is disabled."
