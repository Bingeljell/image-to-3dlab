"""Fix a spot the user points at: cut the triangles around it and cap the opening.

The studio's "Fix a spot" brush. Pixal3D sometimes leaves pockets (a pupil, under a brow,
in a beard) that read as holes; finding them automatically also catches real shape (gaps
between fur tufts or mohawk spikes), so a person picks the spot and this only does the
how: every triangle whose centre lies within `radius` of the point goes, and each opening
left behind is capped with a fan, its centre relaxed toward the rim so the patch sits
flush. The cap takes the paint, UVs and tangents of the rim vertex it starts from, so it
reads as the colour around it. Pure numpy on the GLB's own buffers: textures, materials
and everything else in the file are kept as they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import surface_calm as sc

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
FORMATS = {5126: ("<f4", np.float32), 5125: ("<u4", np.uint32), 5123: ("<u2", np.uint16),
           5121: ("<u1", np.uint8)}
# Brush sizes as a share of the model's height.
RADII = {"small": 0.012, "medium": 0.025, "large": 0.045}


def _read(doc: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    width = COMPONENTS[accessor["type"]]
    dtype = FORMATS[accessor["componentType"]][0]
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride", item)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    out = np.ndarray((accessor["count"], width), dtype=dtype, buffer=binary, offset=start,
                     strides=(stride, np.dtype(dtype).itemsize))
    return np.array(out)


def _boundary_loops(faces: np.ndarray, key: np.ndarray) -> list[list[tuple[int, int]]]:
    """Open edges of `faces` (welded by `key`), as loops of (from, to) original indices."""
    welded = key[faces]
    edges = {}
    for f, wf in zip(faces, welded):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            ka, kb = int(wf[i]), int(wf[(i + 1) % 3])
            if ka == kb:
                continue
            edges.setdefault((min(ka, kb), max(ka, kb)), []).append((a, b, ka, kb))
    open_edges = [e[0] for e in edges.values() if len(e) == 1]
    by_start: dict[int, list[tuple[int, int, int, int]]] = {}
    for edge in open_edges:
        by_start.setdefault(edge[2], []).append(edge)
    loops, used = [], set()
    for edge in open_edges:
        if id(edge) in used:
            continue
        loop, current = [], edge
        while current is not None and id(current) not in used:
            used.add(id(current))
            loop.append((current[0], current[1]))
            nxt = [e for e in by_start.get(current[3], []) if id(e) not in used]
            current = nxt[0] if nxt else None
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def fix_spot(source: Path, output: Path, point: np.ndarray, radius: float) -> dict:
    """Cut triangles with centres within `radius` (model units) of `point` and cap the opening."""
    doc, binary = pp._split_glb(Path(source).read_bytes())
    primitive = doc["meshes"][0]["primitives"][0]
    attributes = {name: _read(doc, binary, index) for name, index in primitive["attributes"].items()}
    faces = _read(doc, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    positions = attributes["POSITION"].astype(np.float64)
    centres = positions[faces].mean(1)
    cut = np.linalg.norm(centres - np.asarray(point, dtype=np.float64), axis=1) < radius
    if not cut.any():
        return {"cut": 0, "capped": 0}
    keep = faces[~cut]
    key = sc.weld(positions)
    loops = _boundary_loops(keep, key)
    # only the openings this cut made: every rim vertex was on a removed triangle
    touched = set(np.unique(key[faces[cut]]).tolist())
    loops = [loop for loop in loops if all(int(key[a]) in touched for a, _ in loop)]
    new_rows = {name: [] for name in attributes}
    new_faces = []
    count = len(positions)
    for loop in loops:
        ring = np.array([a for a, _ in loop])
        first = int(ring[0])
        centre = positions[ring].mean(0)
        for name, values in attributes.items():
            if name == "POSITION":
                new_rows[name].append(centre)
            elif name == "NORMAL":
                n = values[ring].astype(np.float64).mean(0)
                new_rows[name].append(n / max(np.linalg.norm(n), 1e-12))
            else:
                new_rows[name].append(values[first])
        centre_index = count + len(new_rows["POSITION"]) - 1
        for a, b in loop:
            new_faces.append((b, a, centre_index))   # the kept face runs a->b, so the cap runs b->a
    for name, rows in new_rows.items():
        if rows:
            attributes[name] = np.vstack([attributes[name], np.array(rows, dtype=attributes[name].dtype)])
    faces_out = np.vstack([keep, np.array(new_faces, dtype=np.int64).reshape(-1, 3)])
    # relax the cap centres toward their rims so the patch sits flush
    weights = np.zeros(len(attributes["POSITION"]))
    weights[count:] = 1.0
    attributes["POSITION"] = sc.taubin(attributes["POSITION"].astype(np.float64), faces_out, weights,
                                       rounds=5).astype(np.float32)
    _write(doc, binary, primitive, attributes, faces_out, output)
    return {"cut": int(cut.sum()), "capped": len(loops)}


def _write(doc: dict, binary: bytes, primitive: dict, attributes: dict, faces: np.ndarray, output: Path) -> None:
    """The GLB with the mesh's buffers replaced; images and every other buffer view kept."""
    mesh_accessors = set(primitive["attributes"].values()) | {primitive["indices"]}
    mesh_views = {doc["accessors"][i]["bufferView"] for i in mesh_accessors}
    chunks = []
    for index, view in enumerate(doc["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunks.append(b"" if index in mesh_views else binary[start:start + view["byteLength"]])
    for name, index in primitive["attributes"].items():
        accessor = doc["accessors"][index]
        values = np.ascontiguousarray(attributes[name], dtype=FORMATS[accessor["componentType"]][0])
        doc["bufferViews"].append({"buffer": 0, "byteLength": values.nbytes})
        chunks.append(values.tobytes())
        accessor.update(bufferView=len(doc["bufferViews"]) - 1, byteOffset=0, count=len(values))
        if name == "POSITION":
            accessor["min"], accessor["max"] = values.min(0).tolist(), values.max(0).tolist()
    indices = doc["accessors"][primitive["indices"]]
    data = np.ascontiguousarray(faces.ravel(), dtype="<u4")
    doc["bufferViews"].append({"buffer": 0, "byteLength": data.nbytes})
    chunks.append(data.tobytes())
    indices.update(bufferView=len(doc["bufferViews"]) - 1, byteOffset=0, count=int(data.size), componentType=5125)
    indices.pop("min", None)
    indices.pop("max", None)
    packed = b""
    for view, chunk in zip(doc["bufferViews"], chunks):
        packed += b"\0" * (-len(packed) % 4)
        view["byteOffset"] = len(packed)
        view["byteLength"] = len(chunk)
        view.pop("byteStride", None)
        packed += chunk
    doc["buffers"][0]["byteLength"] = len(packed)
    Path(output).write_bytes(pp._join_glb(doc, packed))


def model_height(source: Path) -> float:
    positions = pp.read_glb(source)[0]
    return float(np.ptp(positions[:, 1]))

