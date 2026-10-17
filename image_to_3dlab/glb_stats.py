"""What is in a GLB, in the numbers a game dev checks: triangles, vertices, textures, bones.

Read straight from the file's JSON and its embedded images, so it works on every model
the lab makes (raw, finished, rigged, compressed by gltfpack) without Blender or a GPU.

There is no quad count, on purpose: glTF stores triangles only, so any quads a model had
were split on export and cannot be read back.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

from image_to_3dlab.glb_turn import parse_glb

TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = 4, 5, 6
KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"


def _triangles(mode: int, count: int) -> int:
    if mode == TRIANGLES:
        return count // 3
    if mode in (TRIANGLE_STRIP, TRIANGLE_FAN):
        return max(count - 2, 0)
    return 0  # points and lines draw no faces


def image_size(data: bytes, mime: str | None) -> tuple[int | None, int | None, str]:
    """(width, height, format) of one embedded image; sizes are None if it can't be read."""
    kind = (mime or "").rsplit("/", 1)[-1] or "unknown"
    if data.startswith(KTX2_MAGIC) and len(data) >= 28:
        width, height = struct.unpack_from("<II", data, 20)
        return width, height, "ktx2"
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            return image.width, image.height, kind
    except Exception:
        return None, None, kind


def _image_bytes(document: dict, binary: bytes, image: dict) -> bytes | None:
    view_index = image.get("bufferView")
    if view_index is None:
        return None  # an external file; the lab never writes those
    view = document["bufferViews"][view_index]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def glb_stats(data: bytes) -> dict[str, Any]:
    """Counts for a whole GLB. A mesh placed twice counts twice: that is what gets drawn."""
    try:
        document, binary = parse_glb(data)
    except struct.error as exc:  # too short to hold a GLB header
        raise ValueError("not a GLB file") from exc
    meshes = document.get("meshes", [])
    accessors = document.get("accessors", [])
    placed = [node["mesh"] for node in document.get("nodes", []) if "mesh" in node]
    if not placed:
        placed = list(range(len(meshes)))

    triangles = vertices = 0
    for mesh_index in placed:
        for primitive in meshes[mesh_index].get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            corners = accessors[position]["count"] if position is not None else 0
            vertices += corners
            if "indices" in primitive:
                corners = accessors[primitive["indices"]]["count"]
            triangles += _triangles(primitive.get("mode", TRIANGLES), corners)

    textures = []
    for image in document.get("images", []):
        raw = _image_bytes(document, binary, image)
        width, height, kind = image_size(raw or b"", image.get("mimeType"))
        textures.append({"width": width, "height": height, "format": kind})

    joints = {joint for skin in document.get("skins", []) for joint in skin.get("joints", [])}
    return {
        "triangles": triangles,
        "vertices": vertices,
        "meshes": len(meshes),
        "materials": len(document.get("materials", [])),
        "textures": textures,
        "bones": len(joints),
        "moves": [a.get("name") or f"move {i + 1}" for i, a in enumerate(document.get("animations", []))],
        "bytes": len(data),
    }


def file_stats(path: Path) -> dict[str, Any]:
    return glb_stats(path.read_bytes())
