"""Cut a GLB into one file per part (mesh primitive) and join parts back into one GLB.

Tools written for Finish's single-part models (Fix a spot, Pixel Match) read one primitive.
A Hero model has two, body and head, each with its own textures so a skin can replace the
body's alone. Rather than teach every tool about parts, they run on each part's own file and
the parts are joined again. Each part file carries only its own data: accessors, buffer
views, material, textures, images and samplers it uses, renumbered.
"""
from __future__ import annotations

import copy
from pathlib import Path

from image_to_3dlab.photo_paint import _join_glb, _split_glb


def primitives(doc: dict) -> list[dict]:
    return [p for mesh in doc.get("meshes", []) for p in mesh["primitives"]]


def count(path: Path) -> int:
    doc, _ = _split_glb(Path(path).read_bytes())
    return len(primitives(doc))


def _texture_refs(node, found: set[int]) -> None:
    """Every texture index a material points at: glTF's texture-info objects carry an `index`."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "index" and isinstance(value, int):
                found.add(value)
            else:
                _texture_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _texture_refs(item, found)


def _renumber_textures(node, mapping: dict[int, int]):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "index" and isinstance(value, int) and value in mapping:
                out[key] = mapping[value]
            else:
                out[key] = _renumber_textures(value, mapping)
        return out
    if isinstance(node, list):
        return [_renumber_textures(item, mapping) for item in node]
    return node


def _image_sources(texture: dict) -> list[int]:
    sources = [texture["source"]] if "source" in texture else []
    for extension in texture.get("extensions", {}).values():
        if "source" in extension:
            sources.append(extension["source"])
    return sources


def extract(doc: dict, binary: bytes, index: int) -> bytes:
    """Part INDEX (in primitive order) as a GLB of its own, carrying only what it uses."""
    primitive = copy.deepcopy(primitives(doc)[index])
    accessors = sorted(set(primitive["attributes"].values()) | ({primitive["indices"]} if "indices" in primitive else set()))
    material = doc["materials"][primitive["material"]] if "material" in primitive else None
    textures: set[int] = set()
    if material is not None:
        _texture_refs(material, textures)
    textures_list = sorted(t for t in textures if t < len(doc.get("textures", [])))
    images = sorted({s for t in textures_list for s in _image_sources(doc["textures"][t])})
    samplers = sorted({doc["textures"][t]["sampler"] for t in textures_list if "sampler" in doc["textures"][t]})
    views = sorted({doc["accessors"][a]["bufferView"] for a in accessors if "bufferView" in doc["accessors"][a]}
                   | {doc["images"][i]["bufferView"] for i in images if "bufferView" in doc["images"][i]})

    view_map = {old: new for new, old in enumerate(views)}
    out = {"asset": doc.get("asset", {"version": "2.0"}), "buffers": [{"byteLength": 0}], "bufferViews": [],
           "accessors": [], "meshes": [{"primitives": []}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    used = set(doc.get("extensionsUsed", []))
    if used:
        out["extensionsUsed"] = sorted(used)
    packed = b""
    for old in views:
        view = dict(doc["bufferViews"][old])
        start = view.get("byteOffset", 0)
        packed += b"\0" * (-len(packed) % 4)
        chunk = binary[start:start + view["byteLength"]]
        view.update(buffer=0, byteOffset=len(packed), byteLength=len(chunk))
        out["bufferViews"].append(view)
        packed += chunk
    out["buffers"][0]["byteLength"] = len(packed)
    accessor_map = {}
    for old in accessors:
        accessor = dict(doc["accessors"][old])
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
        accessor_map[old] = len(out["accessors"])
        out["accessors"].append(accessor)
    image_map = {}
    for old in images:
        image = dict(doc["images"][old])
        if "bufferView" in image:
            image["bufferView"] = view_map[image["bufferView"]]
        image_map[old] = len(out.setdefault("images", []))
        out["images"].append(image)
    sampler_map = {}
    for old in samplers:
        sampler_map[old] = len(out.setdefault("samplers", []))
        out["samplers"].append(dict(doc["samplers"][old]))
    texture_map = {}
    for old in textures_list:
        texture = copy.deepcopy(doc["textures"][old])
        if "source" in texture:
            texture["source"] = image_map[texture["source"]]
        if "sampler" in texture:
            texture["sampler"] = sampler_map[texture["sampler"]]
        for extension in texture.get("extensions", {}).values():
            if "source" in extension:
                extension["source"] = image_map[extension["source"]]
        texture_map[old] = len(out.setdefault("textures", []))
        out["textures"].append(texture)
    primitive["attributes"] = {k: accessor_map[v] for k, v in primitive["attributes"].items()}
    if "indices" in primitive:
        primitive["indices"] = accessor_map[primitive["indices"]]
    if material is not None:
        out["materials"] = [_renumber_textures(copy.deepcopy(material), texture_map)]
        primitive["material"] = 0
    out["meshes"][0]["primitives"].append(primitive)
    return _join_glb(out, packed)


def split(source: Path, folder: Path) -> list[Path]:
    """One GLB per part of SOURCE, written into FOLDER, in primitive order."""
    doc, binary = _split_glb(Path(source).read_bytes())
    for node in doc.get("nodes", []):
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("node transforms are not supported; expected Finish's output")
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(len(primitives(doc))):
        path = folder / f"part{index}.glb"
        path.write_bytes(extract(doc, binary, index))
        paths.append(path)
    return paths


def join(parts: list[Path], output: Path) -> None:
    """One GLB holding every part's primitive, each keeping its own material and textures."""
    out = None
    packed = b""
    for path in parts:
        doc, binary = _split_glb(Path(path).read_bytes())
        if out is None:
            out = {"asset": doc.get("asset", {"version": "2.0"}), "buffers": [{"byteLength": 0}], "bufferViews": [],
                   "accessors": [], "meshes": [{"primitives": []}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}],
                   "scene": 0}
        base = {key: len(out.get(key, [])) for key in ("bufferViews", "accessors", "images", "textures", "samplers", "materials")}
        offset = len(packed) + (-len(packed) % 4)
        packed += b"\0" * (-len(packed) % 4) + binary
        for view in doc.get("bufferViews", []):
            out["bufferViews"].append({**view, "buffer": 0, "byteOffset": view.get("byteOffset", 0) + offset})
        for accessor in doc.get("accessors", []):
            moved = dict(accessor)
            if "bufferView" in moved:
                moved["bufferView"] += base["bufferViews"]
            out["accessors"].append(moved)
        for image in doc.get("images", []):
            moved = dict(image)
            if "bufferView" in moved:
                moved["bufferView"] += base["bufferViews"]
            out.setdefault("images", []).append(moved)
        for sampler in doc.get("samplers", []):
            out.setdefault("samplers", []).append(dict(sampler))
        for texture in doc.get("textures", []):
            moved = copy.deepcopy(texture)
            if "source" in moved:
                moved["source"] += base["images"]
            if "sampler" in moved:
                moved["sampler"] += base["samplers"]
            for extension in moved.get("extensions", {}).values():
                if "source" in extension:
                    extension["source"] += base["images"]
            out.setdefault("textures", []).append(moved)
        texture_map = {i: i + base["textures"] for i in range(len(doc.get("textures", [])))}
        for material in doc.get("materials", []):
            out.setdefault("materials", []).append(_renumber_textures(copy.deepcopy(material), texture_map))
        for primitive in primitives(doc):
            moved = copy.deepcopy(primitive)
            moved["attributes"] = {k: v + base["accessors"] for k, v in moved["attributes"].items()}
            if "indices" in moved:
                moved["indices"] += base["accessors"]
            if "material" in moved:
                moved["material"] += base["materials"]
            out["meshes"][0]["primitives"].append(moved)
        used = set(out.get("extensionsUsed", [])) | set(doc.get("extensionsUsed", []))
        if used:
            out["extensionsUsed"] = sorted(used)
    out["buffers"][0]["byteLength"] = len(packed)
    Path(output).write_bytes(_join_glb(out, packed))
