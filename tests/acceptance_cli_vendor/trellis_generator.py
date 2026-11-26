"""Stand-in for the vendored generator. Emit one triangle with no native libraries."""

import json
import os
import struct
import sys
from pathlib import Path


def write_triangle(path: Path) -> None:
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(positions), "target": 34962}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0, 0, 0], "max": [1, 1, 0]},
        ],
    }
    metadata = json.dumps(document).encode()
    metadata += b" " * (-len(metadata) % 4)
    chunks = struct.pack("<I4s", len(metadata), b"JSON") + metadata
    chunks += struct.pack("<I4s", len(positions), b"BIN\0") + positions
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def main() -> None:
    argv = sys.argv[1:]
    Path(os.environ["FIXTURE_INVOCATION"]).write_text(
        json.dumps({"argv": argv, "cwd": str(Path.cwd()),
                    "pre_cap": os.environ.get("I2L_PRE_CAP")}),
        encoding="utf-8",
    )
    write_triangle(Path(argv[argv.index("--output") + 1]).with_suffix(".glb"))
    if os.environ.get("FIXTURE_GENERATION_FAIL") == "1":
        raise SystemExit(23)


if __name__ == "__main__":
    main()
