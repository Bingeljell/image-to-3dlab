#!/bin/bash
# Rebuild, install, and explicitly ad-hoc sign MtlBVH's native extension.
# Stop all Python/TRELLIS workers before running: overwriting a loaded Mach-O image causes
# macOS to kill those processes with CODESIGNING: Invalid Page.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MTLBVH="$REPO/vendor/trellis-mac/deps/mtlbvh"
PYTHON="$REPO/vendor/trellis-mac/.venv/bin/python"
RUNTIME="$REPO/vendor/trellis-mac/.venv/lib/python3.11/site-packages/mtlbvh"

if [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

cd "$MTLBVH"
"$PYTHON" setup.py build_ext --inplace

SOURCE_SO="$MTLBVH/mtlbvh/_C.cpython-311-darwin.so"
RUNTIME_SO="$RUNTIME/_C.cpython-311-darwin.so"
test -f "$SOURCE_SO"
cp -f "$SOURCE_SO" "$RUNTIME_SO"

codesign --force --sign - "$SOURCE_SO"
codesign --force --sign - "$RUNTIME_SO"
codesign --verify --verbose=4 "$RUNTIME_SO"

"$PYTHON" -c 'import mtlbvh._C; print("mtlbvh native import PASS", mtlbvh._C.__file__)'
shasum -a 256 "$SOURCE_SO" "$RUNTIME_SO"
