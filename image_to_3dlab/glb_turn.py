"""Turn a generated GLB half a turn about its vertical axis, and the camera that saw it.

Pixal3D writes its models facing -Z. glTF's front is +Z, so every viewer, engine and
importer shows the back first. Turning the model as it is saved fixes that for
everything downstream, but Pixel Match projects the source photo through the camera
pixal3d.cpp saved beside the run (`<run>.svviews/transforms.json`), and that camera
must turn with the model or the photo lands on the back.

The turn goes into the vertex data, not a node rotation: Pixel Match reads raw vertex
positions and requires a GLB with no node transforms.

A turned GLB is stamped in `asset.extras`, so a second turn is a no-op and readers that
assumed the old facing (the prop splitter's reading order) can tell the two apart.
"""

from __future__ import annotations

import json
import struct

import numpy as np

# asset.extras key and value marking a GLB whose front is +Z.
FACING_KEY = "image_to_3dlab_facing"
FRONT = "+Z"
# transforms.json key marking a camera turned to match.
CAMERA_KEY = "image_to_3dlab_turned"

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
FLOAT = 5126
# Half a turn about +Y negates X and Z. Positions, normals and tangents' xyz all turn;
# a tangent's w (handedness) does not, because a rotation keeps handedness.
TURNED_ATTRIBUTES = ("POSITION", "NORMAL", "TANGENT")
NODE_TRANSFORMS = ("matrix", "rotation", "translation", "scale")


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB file")
    offset, document, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + length]
        if kind == JSON_CHUNK:
            document = json.loads(chunk)
        elif kind == BIN_CHUNK:
            binary = chunk
        offset += 8 + length
    if document is None:
        raise ValueError("GLB has no JSON chunk")
    return document, binary


def build_glb(document: dict, binary: bytes) -> bytes:
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(text), JSON_CHUNK) + text
    if binary:
        chunks += struct.pack("<II", len(binary), BIN_CHUNK) + binary
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks


def faces_front(document: dict) -> bool:
    """Has this GLB been turned to face +Z?"""
    return (document.get("asset", {}).get("extras") or {}).get(FACING_KEY) == FRONT


def glb_faces_front(data: bytes) -> bool:
    return faces_front(parse_glb(data)[0])


def turn_glb(data: bytes) -> bytes:
    """The GLB turned half a turn about Y and stamped as facing front. Already turned:
    returned unchanged.

    Refuses what it cannot turn correctly rather than turning it half-way: node
    transforms, sparse or non-float vertex data.
    """
    document, binary = parse_glb(data)
    if faces_front(document):
        return data
    for node in document.get("nodes", []):
        if any(key in node for key in NODE_TRANSFORMS):
            raise ValueError("GLB has node transforms; only raw vertex data is turned")
    accessors = document.get("accessors", [])
    turned: set[int] = set()
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            targets = primitive.get("targets", [])
            for source in [attributes, *targets]:
                for name in TURNED_ATTRIBUTES:
                    if name in source:
                        turned.add(source[name])
    out = bytearray(binary)
    views = document.get("bufferViews", [])
    for index in sorted(turned):
        accessor = accessors[index]
        if accessor.get("componentType") != FLOAT or "sparse" in accessor:
            raise ValueError(f"accessor {index} is not plain float data; cannot turn it")
        width = {"VEC3": 3, "VEC4": 4}[accessor["type"]]
        view = views[accessor["bufferView"]]
        stride = view.get("byteStride", width * 4)
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        count = accessor["count"]
        rows = np.ndarray((count, width), dtype="<f4", buffer=out, offset=start,
                          strides=(stride, 4))
        rows[:, 0] *= -1
        rows[:, 2] *= -1
        if "min" in accessor and "max" in accessor:
            low, high = list(accessor["min"]), list(accessor["max"])
            for axis in (0, 2):
                low[axis], high[axis] = -accessor["max"][axis], -accessor["min"][axis]
            accessor["min"], accessor["max"] = low, high
    asset = document.setdefault("asset", {"version": "2.0"})
    asset.setdefault("extras", {})[FACING_KEY] = FRONT
    return build_glb(document, bytes(out))


# The same half turn, as seen from pixal3d.cpp's camera frame. photo_paint maps GLB
# (x, y, z) to view space as (-x, z, y), so negating GLB x and z negates view x and y:
# half a turn about view Z.
VIEW_TURN = np.diag([-1.0, -1.0, 1.0, 1.0])


def turn_cameras(transforms: dict) -> dict:
    """transforms.json with every camera turned to match a turned GLB. Idempotent."""
    if transforms.get(CAMERA_KEY):
        return transforms
    turned = dict(transforms)
    turned["frames"] = [
        {**frame, "transform_matrix": (VIEW_TURN @ np.asarray(frame["transform_matrix"],
                                                             dtype=float)).tolist()}
        for frame in transforms.get("frames", [])
    ]
    turned[CAMERA_KEY] = True
    return turned
