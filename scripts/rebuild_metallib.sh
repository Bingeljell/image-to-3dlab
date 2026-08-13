#!/bin/bash
# Rebuild cumesh's Metal shader library after patching a .metal source.
#
# Mirrors setup.py's own commands (compile each .metal to .air with xcrun metal, link them
# with xcrun metallib) rather than running a full `pip install`, which would also rebuild the
# Obj-C++ extension and take far longer for no benefit — only the shaders changed.
#
# The installed copy in site-packages is what actually loads at runtime, so the rebuilt
# library is copied there too. Patching src/metal/ alone changes nothing that executes.
set -euo pipefail

# xcode-select often points at Command Line Tools, which has no Metal compiler. The full
# Xcode does. Set DEVELOPER_DIR explicitly rather than asking the user to switch globally.
if [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    echo "error: no Metal compiler. Install Xcode, or the Metal Toolchain component:" >&2
    echo "  xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MTLMESH="$REPO/vendor/trellis-mac/deps/mtlmesh"
METAL_DIR="$MTLMESH/src/metal"
OUT="$MTLMESH/src/cumesh.metallib"

echo "== compiling shaders =="
AIR_FILES=()
for mf in "$METAL_DIR"/*.metal; do
    air="${mf%.metal}.air"
    echo "   $(basename "$mf")"
    xcrun -sdk macosx metal -c "$mf" -o "$air" \
        -std=metal4.0 -O2 \
        -D__HAVE_ATOMIC_ULONG__=1 \
        -D__HAVE_ATOMIC_ULONG_MIN_MAX__=1 \
        -I "$METAL_DIR"
    AIR_FILES+=("$air")
done

echo "== linking metallib =="
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$OUT"
rm -f "${AIR_FILES[@]}"
ls -la "$OUT"

echo "== installing over every copy that could load =="
# Find where the runtime actually loads cumesh from, and overwrite that one first.
INSTALLED_PKG="$("$REPO/vendor/trellis-mac/.venv/bin/python" -c \
    'import cumesh, os; print(os.path.dirname(cumesh.__file__))')"
echo "   runtime package: $INSTALLED_PKG"

for dest in \
    "$INSTALLED_PKG/cumesh.metallib" \
    "$MTLMESH/cumesh/cumesh.metallib" \
    "$MTLMESH/build/lib.macosx-26.0-arm64-cpython-311/cumesh/cumesh.metallib"
do
    if [ -e "$dest" ] || [ -d "$(dirname "$dest")" ]; then
        cp -f "$OUT" "$dest" && echo "   -> $dest"
    fi
done

echo "== done =="
