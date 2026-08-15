#!/bin/bash
# Rebuild and install only MtlBVH's Metal shader library after editing bvh.metal.
# The Obj-C++ extension does not embed the shaders, so recompiling it is unnecessary.
set -euo pipefail

if [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    echo "error: no Metal compiler; install full Xcode or the Metal toolchain" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MTLBVH="$REPO/vendor/trellis-mac/deps/mtlbvh"
SOURCE="$MTLBVH/src/metal/bvh.metal"
AIR="$MTLBVH/src/metal/bvh.air"
LIBRARY="$MTLBVH/src/mtlbvh.metallib"

echo "== compiling bvh.metal =="
xcrun -sdk macosx metal -c "$SOURCE" -o "$AIR" \
    -std=metal4.0 -O2 \
    -D__HAVE_ATOMIC_ULONG__=1 \
    -D__HAVE_ATOMIC_ULONG_MIN_MAX__=1 \
    -I "$MTLBVH/src"

echo "== linking mtlbvh.metallib =="
xcrun -sdk macosx metallib "$AIR" -o "$LIBRARY"
rm -f "$AIR"

RUNTIME_PKG="$("$REPO/vendor/trellis-mac/.venv/bin/python" -c \
    'import mtlbvh, os; print(os.path.dirname(mtlbvh.__file__))')"
for dest in \
    "$RUNTIME_PKG/mtlbvh.metallib" \
    "$MTLBVH/mtlbvh/mtlbvh.metallib" \
    "$MTLBVH/build/lib.macosx-26.0-arm64-cpython-311/mtlbvh/mtlbvh.metallib"
do
    if [ -e "$dest" ] || [ -d "$(dirname "$dest")" ]; then
        cp -f "$LIBRARY" "$dest"
        echo "   -> $dest"
    fi
done
shasum -a 256 "$LIBRARY"
