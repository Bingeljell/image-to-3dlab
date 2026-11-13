"""Size finished models by baking their scene transforms into vertex data."""

import copy
import math

import numpy as np
import trimesh

from image_to_3dlab.glb_turn import build_glb, parse_glb


def set_height(data: bytes, height: float) -> bytes:
    """Set Y-up height and bottom centre before rigging, preserving materials."""
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height must be a positive finite number")
    doc, binary = parse_glb(data)
    if doc.get("skins") or doc.get("animations"):
        raise ValueError("set height before rigging or animation")
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if len(scenes) != 1:
        raise ValueError("height requires one scene")
    meshes = copy.deepcopy(doc.get("meshes", []))
    doc["meshes"] = []
    entries = []
    visited = set()

    def rows(index):
        a = doc["accessors"][index]
        if a.get("componentType") != 5126 or "sparse" in a:
            raise ValueError("height requires plain float vertex data")
        width = {"VEC3": 3, "VEC4": 4}[a["type"]]
        v = doc["bufferViews"][a["bufferView"]]
        return np.ndarray((a["count"], width), dtype="<f4", buffer=binary,
                          offset=v.get("byteOffset", 0) + a.get("byteOffset", 0),
                          strides=(v.get("byteStride", width * 4), 4)).copy()

    def visit(index, parent):
        if index in visited:
            raise ValueError("scene contains a repeated node")
        visited.add(index)
        node = nodes[index]
        if "matrix" in node:
            local = np.array(node["matrix"]).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            local = trimesh.transformations.quaternion_matrix([w, x, y, z])
            local[:3, :3] *= np.array(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        world = parent @ local
        if np.linalg.det(world[:3, :3]) <= 0:
            raise ValueError("height requires non-mirrored, non-zero transforms")
        if "mesh" in node:
            mesh = copy.deepcopy(meshes[node["mesh"]])
            for primitive in mesh["primitives"]:
                if primitive.get("targets") or primitive.get("extensions"):
                    raise ValueError("height does not support morph or compressed geometry")
                attrs = primitive["attributes"]
                points = trimesh.transform_points(rows(attrs["POSITION"]), world)
                entries.append((attrs, world, points))
            node["mesh"] = len(doc["meshes"])
            doc["meshes"].append(mesh)
        for key in ("matrix", "rotation", "translation", "scale"):
            node.pop(key, None)
        for child in node.get("children", []):
            visit(child, world)

    for root in scenes[0].get("nodes", []):
        visit(root, np.eye(4))
    if not entries:
        raise ValueError("model has no measurable height")
    points = np.concatenate([e[2] for e in entries])
    low, high = points.min(0), points.max(0)
    if not np.isfinite(points).all() or high[1] <= low[1]:
        raise ValueError("model has no measurable height")
    scale = height / (high[1] - low[1])
    origin = (low + high) / 2
    origin[1] = low[1]
    output = bytearray(binary)

    def append_attribute(attrs, name, values):
        output.extend(b"\0" * (-len(output) % 4))
        raw = np.asarray(values, dtype="<f4").tobytes()
        view = len(doc["bufferViews"])
        doc["bufferViews"].append({"buffer": 0, "byteOffset": len(output), "byteLength": len(raw)})
        output.extend(raw)
        a = copy.deepcopy(doc["accessors"][attrs[name]])
        a.update(bufferView=view, byteOffset=0, min=values.min(0).tolist(), max=values.max(0).tolist())
        attrs[name] = len(doc["accessors"])
        doc["accessors"].append(a)

    for attrs, world, points in entries:
        append_attribute(attrs, "POSITION", (points - origin) * scale)
        for name in ("NORMAL", "TANGENT"):
            if name in attrs:
                values = rows(attrs[name])
                matrix = np.linalg.inv(world[:3, :3]).T if name == "NORMAL" else world[:3, :3]
                vectors = values[:, :3] @ matrix.T
                lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
                values[:, :3] = vectors / np.maximum(lengths, 1e-30)
                append_attribute(attrs, name, values)
    doc["buffers"][0]["byteLength"] = len(output)
    return build_glb(doc, bytes(output))


def size_file(path, height):
    """Size a finished file and refresh its existing licence record's identity."""
    import hashlib
    import json

    from image_to_3dlab.provenance import write_provenance

    result = set_height(path.read_bytes(), height)
    temporary = path.with_suffix('.sized.glb')
    temporary.write_bytes(result)
    temporary.replace(path)
    sidecar = path.with_suffix('.provenance.json')
    if sidecar.is_file():
        record = json.loads(sidecar.read_text(encoding='utf-8'))
        record.setdefault('output', {})['sha256'] = hashlib.sha256(result).hexdigest()
        write_provenance(sidecar, record, atomic=True)
