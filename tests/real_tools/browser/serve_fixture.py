"""Run the shipped Studio handler against a temporary library and a tiny GLB."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[3]


def write_triangle(path: Path) -> None:
    # A flat magenta triangle with an unlit material, so its pixels are easy to
    # distinguish from the grid and background regardless of the lighting.
    vertices = struct.pack("<9f", -0.5, 0, 0, 0.5, 0, 0, 0, 1, 0)
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{
            "doubleSided": True,
            "extensions": {"KHR_materials_unlit": {}},
            "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 1, 1]},
        }],
        "accessors": [{
            "bufferView": 0, "componentType": 5126, "count": 3,
            "type": "VEC3", "min": [-0.5, 0, 0], "max": [0.5, 1, 0],
        }],
        "bufferViews": [{"buffer": 0, "byteLength": len(vertices)}],
        "buffers": [{"byteLength": len(vertices)}],
    }
    scene = json.dumps(document).encode("ascii")
    scene += b" " * (-len(scene) % 4)
    chunks = struct.pack("<I4s", len(scene), b"JSON") + scene
    chunks += struct.pack("<I4s", len(vertices), b"BIN\x00") + vertices
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def main() -> None:
    with TemporaryDirectory(prefix="assetfurnace-browser-") as temporary:
        root = Path(temporary)
        (root / "viewer").symlink_to(REPO / "viewer", target_is_directory=True)
        write_triangle(root / "output" / "browser-triangle" / "browser-triangle.glb")
        os.environ.update(
            HOME=str(root),
            USERPROFILE=str(root),
            I3D_NO_USAGE="1",
            I3D_NO_NOTICES="1",
            I3D_NO_UPDATE_CHECK="1",
        )
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "viewer"))
        os.chdir(root)
        import generate_api
        from serve import ThreadingHTTPServer

        generate_api.OUTPUT_ROOT = root / "output"
        generate_api.MODELS_SETTING_REPO = root

        class FixtureHandler(generate_api.Handler):
            def do_POST(self) -> None:
                self.send_error(405, "The browser fixture is read-only")

        with ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(FixtureHandler, directory=str(root))
        ) as server:
            print(f"http://127.0.0.1:{server.server_address[1]}", flush=True)
            # The test owns stdin. EOF ends the fixture and removes its files.
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                sys.stdin.read()
            finally:
                server.shutdown()


if __name__ == "__main__":
    main()
