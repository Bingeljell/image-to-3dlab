"""Put a skin's paint on any model that shares its mesh: a rigged character, a move, one file
holding every skin.

A skin (viewer/skins_api.py) is the same mesh with new paint, and rigging keeps the mesh's
UV map (checked on the gnoll, 2026-10-19: every UV the same), so dressing a rigged model or
a move in a skin is only a swap of two pictures: the colour map and the metal/shine map. The
normal map is the shape's own and stays.

`with_variants` writes one GLB carrying the original paint and every skin, switchable with
glTF's KHR_materials_variants (Blender, three.js and <model-viewer> list them by name).
"""

from __future__ import annotations

import copy
from typing import Any

from image_to_3dlab.glb_turn import build_glb, parse_glb

VARIANTS = "KHR_materials_variants"
WEBP = "EXT_texture_webp"   # Finish's compression stores some pictures as WebP, listed here
# The pictures a skin replaces. The normal map belongs to the shape and is never swapped.
PAINT = ("baseColorTexture", "metallicRoughnessTexture")


def _source(texture: dict) -> int:
    """A texture's image, whether plain or through EXT_texture_webp."""
    if "source" in texture:
        return texture["source"]
    return texture["extensions"][WEBP]["source"]


def _set_source(texture: dict, image: int) -> None:
    if "source" in texture:
        texture["source"] = image
    else:
        texture["extensions"][WEBP]["source"] = image


def paint_images(document: dict, material: int = 0) -> dict[str, int]:
    """{"baseColorTexture": image index, ...} for one material's paint pictures."""
    pbr = (document.get("materials") or [{}])[material].get("pbrMetallicRoughness") or {}
    images = {}
    for role in PAINT:
        texture = (pbr.get(role) or {}).get("index")
        if texture is not None:
            images[role] = _source(document["textures"][texture])
    return images


def _image_bytes(document: dict, binary: bytes, image: int) -> tuple[bytes, str]:
    entry = document["images"][image]
    view = document["bufferViews"][entry["bufferView"]]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]], entry.get("mimeType", "image/png")


def _repack(document: dict, binary: bytes, extra: list[bytes]) -> tuple[dict, bytes]:
    """Rebuild the binary chunk from the buffer views still used, plus `extra` blobs appended as
    new views (returned in order as the last views). Unused views (swapped-out pictures) drop."""
    used = {i for i, _ in enumerate(document["bufferViews"])}
    out, views = bytearray(), []
    for index, view in enumerate(document["bufferViews"]):
        if index not in used:
            continue
        start = view.get("byteOffset", 0)
        out += b"\0" * (-len(out) % 4)
        views.append({**view, "buffer": 0, "byteOffset": len(out)})
        out += binary[start:start + view["byteLength"]]
    for blob in extra:
        out += b"\0" * (-len(out) % 4)
        views.append({"buffer": 0, "byteOffset": len(out), "byteLength": len(blob)})
        out += blob
    document["bufferViews"] = views
    document["buffers"] = [{"byteLength": len(out)}]
    return document, bytes(out)


def _texture_refs(node: Any):
    """Every {"index": n} texture reference inside a material (any key ending in "Texture")."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("Texture") and isinstance(value, dict) and "index" in value:
                yield value
            else:
                yield from _texture_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _texture_refs(value)


def _drop_unused_images(document: dict, binary: bytes) -> tuple[dict, bytes]:
    """Remove textures no material uses, images no texture uses, and buffer views nothing uses,
    renumbering what stays."""
    refs = list(_texture_refs(document.get("materials", [])))
    keep_textures = sorted({r["index"] for r in refs})
    texture_map = {old: new for new, old in enumerate(keep_textures)}
    document["textures"] = [document["textures"][i] for i in keep_textures]
    for ref in refs:
        ref["index"] = texture_map[ref["index"]]
    keep_images = sorted({_source(t) for t in document.get("textures", [])})
    remap = {old: new for new, old in enumerate(keep_images)}
    document["images"] = [document["images"][i] for i in keep_images]
    for texture in document.get("textures", []):
        _set_source(texture, remap[_source(texture)])
    referenced = {a["bufferView"] for a in document.get("accessors", []) if "bufferView" in a}
    referenced |= {img["bufferView"] for img in document["images"] if "bufferView" in img}
    views = sorted(referenced)
    view_map = {old: new for new, old in enumerate(views)}
    for accessor in document.get("accessors", []):
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
    for image in document["images"]:
        if "bufferView" in image:
            image["bufferView"] = view_map[image["bufferView"]]
    old_views = document["bufferViews"]
    document["bufferViews"] = [old_views[i] for i in views]
    return _repack(document, binary, [])


def _add_image(document: dict, data: bytes, mime: str, blobs: list[bytes], first_new_view: int) -> int:
    """Register a picture whose bytes go in `blobs` (appended as views by _repack). Returns its texture."""
    view = first_new_view + len(blobs)
    blobs.append(data)
    document["images"].append({"bufferView": view, "mimeType": mime})
    image = len(document["images"]) - 1
    sampler = (document.get("textures") or [{}])[0].get("sampler")
    texture = ({"extensions": {WEBP: {"source": image}}} if mime == "image/webp" else {"source": image})
    if sampler is not None:
        texture["sampler"] = sampler
    if mime == "image/webp":   # a viewer without WebP must not show it unpainted, so it is required
        for key in ("extensionsUsed", "extensionsRequired"):
            document[key] = sorted(set(document.get(key, [])) | {WEBP})
    document.setdefault("textures", []).append(texture)
    return len(document["textures"]) - 1


def _paint_of(skin: bytes) -> dict[str, tuple[bytes, str]]:
    document, binary = parse_glb(skin)
    return {role: _image_bytes(document, binary, image) for role, image in paint_images(document).items()}


def dress(model: bytes, skin: bytes) -> bytes:
    """`model` (a rigged character, a move, the finished model) wearing `skin`'s paint."""
    document, binary = parse_glb(model)
    document = copy.deepcopy(document)
    paint = _paint_of(skin)
    if not paint:
        raise ValueError("the skin has no paint pictures to copy")
    blobs: list[bytes] = []
    first = len(document["bufferViews"])
    for index, material in enumerate(document.get("materials") or []):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        for role, (data, mime) in paint.items():
            if role in paint_images(document, index):
                pbr[role] = {**pbr[role], "index": _add_image(document, data, mime, blobs, first)}
    document, binary = _repack(document, binary, blobs)
    document, binary = _drop_unused_images(document, binary)
    return build_glb(document, binary)


def with_variants(model: bytes, skins: list[tuple[str, bytes]], original: str = "Original") -> bytes:
    """One GLB: `model` with its own paint, plus every skin as a named variant of its material."""
    document, binary = parse_glb(model)
    document = copy.deepcopy(document)
    materials = document.get("materials") or []
    if not materials:
        raise ValueError("the model has no material to make skins of")
    blobs: list[bytes] = []
    first = len(document["bufferViews"])
    names = [original] + [name for name, _ in skins]
    # per original material: the material index each variant uses
    per_variant: list[list[int]] = [list(range(len(materials)))]
    originals = list(materials)   # each skin copies these, never an earlier skin's copies
    for _, skin in skins:
        paint = _paint_of(skin)
        indices = []
        for material in originals:
            dressed = copy.deepcopy(material)
            pbr = dressed.setdefault("pbrMetallicRoughness", {})
            for role, (data, mime) in paint.items():
                if role in pbr:
                    pbr[role] = {**pbr[role], "index": _add_image(document, data, mime, blobs, first)}
            dressed["name"] = f"{material.get('name', 'material')} ({names[len(per_variant)]})"
            materials.append(dressed)
            indices.append(len(materials) - 1)
        per_variant.append(indices)
    document["materials"] = materials
    document.setdefault("extensions", {})[VARIANTS] = {"variants": [{"name": n} for n in names]}
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            base = primitive.get("material")
            if base is None or base >= len(per_variant[0]):
                continue
            mappings: dict[int, list[int]] = {}
            for variant, indices in enumerate(per_variant):
                mappings.setdefault(indices[base], []).append(variant)
            primitive.setdefault("extensions", {})[VARIANTS] = {
                "mappings": [{"material": m, "variants": v} for m, v in mappings.items()]}
    used = set(document.get("extensionsUsed", []))
    document["extensionsUsed"] = sorted(used | {VARIANTS})
    document, binary = _repack(document, binary, blobs)
    return build_glb(document, binary)


def variant_names(data: bytes) -> list[str]:
    document, _ = parse_glb(data)
    return [v["name"] for v in ((document.get("extensions") or {}).get(VARIANTS) or {}).get("variants", [])]


def describe(data: bytes) -> dict[str, Any]:
    """What a GLB carries, for tests and checks: its paint pictures' sizes per material."""
    document, binary = parse_glb(data)
    return {"materials": len(document.get("materials") or []),
            "paint": [{role: len(_image_bytes(document, binary, img)[0]) for role, img in paint_images(document, i).items()}
                      for i in range(len(document.get("materials") or []))]}
