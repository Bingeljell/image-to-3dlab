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


def write_triangle(path: Path, *, animated: bool = False) -> None:
    # A flat magenta triangle with an unlit material, so its pixels are easy to
    # distinguish from the grid and background regardless of the lighting.
    binary = struct.pack("<9f", -0.5, 0, 0, 0.5, 0, 0, 0, 1, 0)
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
        "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
    }
    if animated:
        document["animations"] = []
        for name, duration, shift in [("Attack", 1, -0.5), ("Attack.001", 2, 0.5), ("Idle", 3, 0)]:
            animation = struct.pack("<8f", 0, duration, 0, 0, 0, shift, 0, 0)
            offset = len(binary)
            binary += animation
            time = len(document["accessors"])
            for start, length, accessor in [
                (offset, 8, {"type": "SCALAR", "min": [0], "max": [duration]}),
                (offset + 8, 24, {"type": "VEC3"}),
            ]:
                document["accessors"].append(dict(accessor, bufferView=len(document["bufferViews"]),
                                                  componentType=5126, count=2))
                document["bufferViews"].append({"buffer": 0, "byteOffset": start, "byteLength": length})
            document["animations"].append({
                "name": name,
                "samplers": [{"input": time, "output": time + 1}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            })
    document["buffers"] = [{"byteLength": len(binary)}]
    scene = json.dumps(document).encode("ascii")
    scene += b" " * (-len(scene) % 4)
    chunks = struct.pack("<I4s", len(scene), b"JSON") + scene
    chunks += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def main() -> None:
    with TemporaryDirectory(prefix="assetfurnace-browser-") as temporary:
        root = Path(temporary)
        (root / "viewer").symlink_to(REPO / "viewer", target_is_directory=True)
        (root / "understanding_3d").symlink_to(REPO / "understanding_3d", target_is_directory=True)
        write_triangle(root / "output" / "browser-triangle" / "browser-triangle.glb")
        write_triangle(root / "browser-moves.glb", animated=True)
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

        class FixtureServer(ThreadingHTTPServer):
            # Chromium can fetch dozens of Studio modules together on a cold load.
            request_queue_size = 64

        with FixtureServer(
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
