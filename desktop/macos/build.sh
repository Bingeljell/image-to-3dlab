#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build_dir="${TMPDIR:-/tmp}/assetfurnace-desktop-build"
output="${1:-$here/build/AssetFurnace.app}"
swift build --disable-sandbox --package-path "$here" --scratch-path "$build_dir"
bin="$(swift build --disable-sandbox --package-path "$here" --scratch-path "$build_dir" --show-bin-path)"
mkdir -p "$output/Contents/MacOS"
cp "$bin/AssetFurnace" "$output/Contents/MacOS/AssetFurnace"
cp "$here/Info.plist" "$output/Contents/Info.plist"
codesign --force --sign - "$output"
