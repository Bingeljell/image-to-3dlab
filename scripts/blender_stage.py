#!/usr/bin/env python3
"""Put assets side by side in the RUNNING Blender, lit, textured and ready to look at.

Run (live): python scripts/blender_stage.py A.glb B.glb --labels "old" "new"

    python scripts/blender_stage.py A.glb B.glb --labels "old" "new"

**Why this exists.** The render helpers in this repo wipe the scene and swap in their own
materials -- `compare_to_source.py --grey` in particular replaces every material with plain
grey. Run one of those after staging a comparison and the user is left looking at grey
blobs wondering where the texture went. That happened repeatedly. This script is the
counterpart: it stages for *looking at*, not for rendering, and it is the last thing to run
before handing the viewport back.

What it guarantees:

* every asset normalised to the same height and spread along Y, so size differences do not
  masquerade as quality differences
* three-point lighting, so the surface is readable
* viewport set to **Material Preview**, so textures actually show
* **backface culling on**, because a double-sided preview hides a hollow mesh completely
  and that is exactly the defect that went unnoticed here for weeks
* nothing selected, so no orange highlight is mistaken for a property of the mesh

This drives the live GUI over the socket on 9876. The lineup goes in a private scene,
so the objects already in the file stay where they are. The window is left on that
scene so you can look at it.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from .blender_worker_source import build_worker_code
else:
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_scene_stage_worker.py")


def spread_offsets(count: int, gap: float = 1.25) -> list[float]:
    """Y offsets that keep the group centred on the origin.

    Centred rather than starting at zero so `view_all` frames the set symmetrically and
    a two-asset comparison sits either side of the middle instead of off to one edge.
    """
    if count <= 0:
        return []
    middle = (count - 1) / 2.0
    return [(index - middle) * gap for index in range(count)]


def labels_for(assets: list[str], labels: list[str] | None) -> list[str]:
    """Collection names: given labels, else the file stems, always unique and prefixed.

    The index prefix matters: Blender's outliner sorts alphabetically, so without it the
    left-to-right order in the viewport and the top-to-bottom order in the outliner
    disagree, which makes "the second one" ambiguous.
    """
    chosen = list(labels) if labels else [Path(a).stem for a in assets]
    if len(chosen) != len(assets):
        raise SystemExit(f"got {len(assets)} assets but {len(chosen)} labels")
    cleaned = []
    for index, name in enumerate(chosen):
        safe = "".join(character if character.isalnum() else "_" for character in name)
        cleaned.append(f"{chr(ord('A') + index)}_{safe}"[:60])
    return cleaned


def build_code(assets: list[str], names: list[str], offsets: list[float], culled: bool) -> str:
    items = [
        {"path": str(Path(asset).expanduser().resolve()), "name": name, "dy": offset}
        for asset, name, offset in zip(assets, names, offsets)
    ]
    return build_worker_code(WORKER_PATH, {
        "action": "stage_assets", "parameters": {"items": items, "culled": culled},
    })


def send(code: str, host: str, port: int) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(300)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
    return b"".join(chunks).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("assets", nargs="+", type=str)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--gap", type=float, default=1.25)
    parser.add_argument(
        "--double-sided",
        action="store_true",
        help="turn backface culling OFF. Hides hollow meshes -- only for showing what a "
             "double-sided preview flatters",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    names = labels_for(args.assets, args.labels)
    offsets = spread_offsets(len(args.assets), args.gap)
    print(send(build_code(args.assets, names, offsets, not args.double_sided), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
