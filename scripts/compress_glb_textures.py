#!/usr/bin/env python3
"""Re-encode a GLB's textures, without touching its geometry.

    .venv/bin/python scripts/compress_glb_textures.py in.glb out.glb

**Why.** Retopology cut the Snag's geometry from 11.8 MB to 1.6 MB and the file still
weighed 32 MB, because Hunyuan's paint stage writes a 4096x4096 albedo *and* a 4096x4096
metallic-roughness map as uncompressed PNG: 30.5 MB of the 32. The mesh was never the
problem. Measured on that albedo, WebP at quality 90 holds 36.3 dB PSNR for 4.36 MB
against PNG's 16.0 — same pixels, same resolution, a quarter of the bytes.

**Resolution.** The paint stage emits 4096 and the default here is 2048, because a
generated asset does not carry 4096 worth of real detail — the rendered difference sits
inside the renderer's own sampling noise. `--max-size 0` keeps whatever came in.

**Colour maps and data maps are not the same thing** and this treats them differently.
An albedo is looked at, so perceptual coding is exactly right. A metallic-roughness or
normal map is *read as numbers* by the shader, so its resolution can usually drop far
below the albedo's without anyone seeing it, but heavy quantisation shows up as shading
artifacts rather than as blur. The defaults follow that: 2048 for colour, 1024 for the
rest, and quality 90 for both.

**A game budget is one size, and sometimes one map.** Finish passes the chosen size for
colour and normal, and caps metal-roughness at 1024 (`scripts/retopo_repaint.texture_arguments`),
so a 512 budget is 512 everywhere while the default file stays the size it always was.
`drop_surface_maps` goes further and removes the normal and metallic-roughness maps
altogether — the material stays valid, just flat (`metallicFactor` 0 and a fixed roughness)
— which is what a phone budget or a matte art style wants. Their images, textures, samplers
and bytes leave the file with them (#239).

**Format.** JPEG is the default, not WebP, even though WebP is smaller — 4.7 MB against
3.5 on the Snag. JPEG is core glTF: no extension, nothing to negotiate, and it survives
every converter on the way to a game engine, which for this project means USDZ. WebP has
to travel as `EXT_texture_webp`, and the honest position on that is that there is no real
fallback: the extension is declared in `extensionsUsed` rather than `extensionsRequired`
and the texture keeps its plain `source`, but that source now points at the same WebP
bytes, so a reader that refuses a non-core mime type has nothing else to read. Blender and
trimesh both load it; a USD converter may not. `--format webp` when the destination is
known to handle it, which is the case for anything built on three.js.

The GLB is rewritten chunk-by-chunk rather than round-tripped through a mesh library:
accessors, skins, morph targets and extras all keep their indices, and only the image
byte ranges move.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

# A texture that feeds one of these slots is data, not colour: the shader reads its
# channels as numbers. Everything else is looked at directly.
DATA_SLOTS = ("metallicRoughnessTexture", "normalTexture", "occlusionTexture")
# What "drop the surface maps" takes out of a material: bumps (normal) and shine
# (metallic-roughness), and the occlusion map, which is a data map too and is often packed
# into the same image as the other two.
SURFACE_SLOTS = ("normalTexture", "occlusionTexture")

MIME = {"webp": "image/webp", "jpeg": "image/jpeg", "png": "image/png"}


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    """Split a GLB into its JSON and binary chunks."""
    if data[:4] != b"glTF":
        raise ValueError("not a GLB (missing glTF magic)")
    offset = 12
    document: dict | None = None
    binary = b""
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == JSON_CHUNK:
            document = json.loads(chunk)
        elif kind == BIN_CHUNK:
            binary = chunk
        offset += 8 + length + (-length % 4)
    if document is None:
        raise ValueError("GLB has no JSON chunk")
    return document, binary


def build_glb(document: dict, binary: bytes) -> bytes:
    """Reassemble a GLB, padding both chunks as the spec requires."""
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary = binary + b"\x00" * (-len(binary) % 4)

    parts = [struct.pack("<II", len(json_chunk), JSON_CHUNK), json_chunk]
    if binary:
        parts += [struct.pack("<II", len(binary), BIN_CHUNK), binary]
    body = b"".join(parts)
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


def normal_texture_images(document: dict) -> set[int]:
    """Image indices feeding a material's normal map."""
    return _slot_images(document, normal_only=True)


def data_texture_images(document: dict) -> set[int]:
    """Image indices feeding a slot the shader reads as numbers rather than colour."""
    return _slot_images(document, normal_only=False)


def _slot_images(document: dict, normal_only: bool) -> set[int]:
    textures = document.get("textures", [])
    found: set[int] = set()

    def note(slot: dict | None) -> None:
        if not slot or "index" not in slot:
            return
        texture = textures[slot["index"]]
        source = texture.get("source")
        if source is None:
            source = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source")
        if source is not None:
            found.add(source)

    for material in document.get("materials", []):
        if normal_only:
            note(material.get("normalTexture"))
            continue
        pbr = material.get("pbrMetallicRoughness", {})
        note(pbr.get("metallicRoughnessTexture"))
        for slot in DATA_SLOTS:
            note(material.get(slot))
    return found


def material_texture_indices(node: object) -> set[int]:
    """Every texture index a material mentions, at any depth (extensions included)."""
    found: set[int] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("Texture") and isinstance(value, dict) and "index" in value:
                found.add(value["index"])
            else:
                found |= material_texture_indices(value)
    elif isinstance(node, list):
        for item in node:
            found |= material_texture_indices(item)
    return found


def strip_surface_maps(document: dict, roughness: float = 0.65) -> None:
    """Make every material flat: no normal or metallic-roughness map, plain factors.

    Metallic goes to 0 (a matte surface reflects almost nothing) and roughness to the
    run's own value, so the asset keeps the look its other settings gave it. Occlusion is
    removed too: it is a data map as well, and an ORM-packed one shares the metal-roughness
    image, so keeping the slot would keep the bytes the drop exists to save.
    """
    for material in document.get("materials", []):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr.pop("metallicRoughnessTexture", None)
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = roughness
        for slot in SURFACE_SLOTS:
            material.pop(slot, None)


def drop_views(document: dict, candidates: set[int]) -> set[int]:
    """Remove bufferViews from `candidates` and move every reference to the shifted index.

    A view is only removed when no accessor and no remaining image points at it: buffer
    0 is shared by mesh attributes and textures, and a view holding vertices is not an
    image's to delete. Returns the indices (old numbering) that went.
    """
    views = document.get("bufferViews", [])
    used = {accessor["bufferView"] for accessor in document.get("accessors", [])
            if "bufferView" in accessor}
    for accessor in document.get("accessors", []):
        sparse = accessor.get("sparse") or {}
        for part in ("indices", "values"):
            ref = sparse.get(part)
            if isinstance(ref, dict) and "bufferView" in ref:
                used.add(ref["bufferView"])
    used |= {image["bufferView"] for image in document.get("images", [])
             if "bufferView" in image}
    dropped = {index for index in candidates if index not in used}
    if not dropped:
        return set()
    after = {old: new for new, old in enumerate(i for i in range(len(views)) if i not in dropped)}
    document["bufferViews"] = [views[i] for i in range(len(views)) if i not in dropped]
    for accessor in document.get("accessors", []):
        if "bufferView" in accessor:
            accessor["bufferView"] = after[accessor["bufferView"]]
        sparse = accessor.get("sparse") or {}
        for part in ("indices", "values"):
            ref = sparse.get(part)
            if isinstance(ref, dict) and "bufferView" in ref:
                ref["bufferView"] = after[ref["bufferView"]]
    for image in document.get("images", []):
        if "bufferView" in image:
            image["bufferView"] = after[image["bufferView"]]
    return dropped


def prune_unused(document: dict) -> int:
    """Drop textures, images, samplers and image bytes no material points at any more.

    Called after the surface maps are stripped. Everything is found by reachability rather
    than by remembering what was removed, so a hand-edited glTF that arrives with an orphan
    to begin with is cleaned up the same way. Returns how many images went.
    """
    materials = document.get("materials", [])
    used_textures = set().union(*(material_texture_indices(m) for m in materials)) if materials else set()
    textures = document.get("textures", [])
    kept = sorted(index for index in used_textures if 0 <= index < len(textures))
    if len(kept) != len(textures):
        after = {old: new for new, old in enumerate(kept)}
        document["textures"] = textures = [textures[i] for i in kept]

        def remap(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.endswith("Texture") and isinstance(value, dict) and value.get("index") in after:
                        value["index"] = after[value["index"]]
                    else:
                        remap(value)
            elif isinstance(node, list):
                for item in node:
                    remap(item)

        for material in materials:
            remap(material)

    images = document.get("images", [])
    used_images = set()
    for texture in textures:
        source = texture.get("source")
        if isinstance(source, int):
            used_images.add(source)
        for extension in texture.get("extensions", {}).values():
            if isinstance(extension, dict) and isinstance(extension.get("source"), int):
                used_images.add(extension["source"])
    kept_images = sorted(index for index in used_images if 0 <= index < len(images))
    dropped_count = len(images) - len(kept_images)
    if dropped_count:
        after_image = {old: new for new, old in enumerate(kept_images)}
        orphans = {images[i]["bufferView"] for i in range(len(images))
                   if i not in kept_images and "bufferView" in images[i]}
        document["images"] = images = [images[i] for i in kept_images]
        for texture in textures:
            if isinstance(texture.get("source"), int):
                texture["source"] = after_image[texture["source"]]
            for extension in texture.get("extensions", {}).values():
                if isinstance(extension, dict) and isinstance(extension.get("source"), int):
                    extension["source"] = after_image[extension["source"]]
        drop_views(document, orphans)

    samplers = document.get("samplers", [])
    used_samplers = sorted({texture["sampler"] for texture in textures
                            if isinstance(texture.get("sampler"), int)})
    if len(used_samplers) != len(samplers):
        after_sampler = {old: new for new, old in enumerate(used_samplers)}
        document["samplers"] = [samplers[i] for i in used_samplers]
        for texture in textures:
            if isinstance(texture.get("sampler"), int):
                texture["sampler"] = after_sampler[texture["sampler"]]
    return dropped_count


def encode(image, fmt: str, quality: int, max_size: int | None) -> tuple[bytes, str]:
    """Re-encode one image, optionally capping its longest side."""
    from PIL import Image

    if max_size and max(image.size) > max_size:
        scale = max_size / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.LANCZOS)

    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if fmt == "jpeg" and has_alpha:
        # JPEG cannot carry alpha, and silently dropping it would turn a cutout opaque.
        fmt = "png"
    image = image.convert("RGBA" if (has_alpha and fmt != "jpeg") else "RGB")

    buffer = io.BytesIO()
    if fmt == "webp":
        image.save(buffer, "WEBP", quality=quality, method=6)
    elif fmt == "jpeg":
        # No `optimize=True`: Pillow's optimising JPEG encoder needs a real file handle
        # and raises "broken data stream" writing into a BytesIO for some image sizes.
        # It buys a couple of percent and is not worth a format-dependent crash.
        image.save(buffer, "JPEG", quality=quality, subsampling=0)
    else:
        image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue(), fmt


def declare_webp(document: dict, texture_index: int, image_index: int) -> None:
    """Point a texture at a WebP image while leaving its fallback source intact.

    The extension goes in `extensionsUsed` only. Listing it in `extensionsRequired` would
    let a viewer refuse the whole file; leaving the original `source` in place means one
    that ignores the extension still renders, just from the fallback image.
    """
    used = document.setdefault("extensionsUsed", [])
    if "EXT_texture_webp" not in used:
        used.append("EXT_texture_webp")
    texture = document["textures"][texture_index]
    texture.setdefault("extensions", {})["EXT_texture_webp"] = {"source": image_index}


def compress(
    data: bytes,
    fmt: str = "jpeg",
    quality: int = 90,
    map_quality: int = 90,
    max_size: int | None = 2048,
    map_size: int | None = 1024,
    normal_size: int | None = 2048,
    drop_surface_maps: bool = False,
    roughness: float = 0.65,
) -> tuple[bytes, list[dict]]:
    """Re-encode every embedded image. Returns (glb_bytes, per-image report).

    A normal map holds the edges of every block and bevel, which is what lets a light mesh
    look crisp, so it gets `normal_size` (None: the same cap as the other data maps).
    `drop_surface_maps` removes the normal and metallic-roughness maps instead of resizing
    them; the material is left flat and valid (`roughness` is the factor it keeps)."""
    from PIL import Image

    document, binary = parse_glb(data)
    if not document.get("images") or not document.get("bufferViews"):
        return data, []

    dropped = 0
    if drop_surface_maps:
        strip_surface_maps(document, roughness)
        dropped = prune_unused(document)

    images = document.get("images", [])
    views = document.get("bufferViews", [])

    data_images = data_texture_images(document)
    normal_images = normal_texture_images(document) if normal_size else set()

    replacements: dict[int, bytes] = {}
    report: list[dict] = []
    for index, image in enumerate(images):
        view_index = image.get("bufferView")
        if view_index is None:
            continue
        view = views[view_index]
        start = view.get("byteOffset", 0)
        blob = binary[start : start + view["byteLength"]]
        is_data = index in data_images
        with Image.open(io.BytesIO(blob)) as opened:
            opened.load()
            before_size = opened.size
            encoded, used = encode(
                opened,
                fmt,
                map_quality if is_data else quality,
                normal_size if index in normal_images else map_size if is_data else max_size,
            )
        if len(encoded) >= len(blob):
            # Re-encoding made it bigger, which happens on already-compact textures.
            report.append({
                "image": index, "kind": "data" if is_data else "colour",
                "before": len(blob), "after": len(blob), "skipped": True,
            })
            continue
        replacements[view_index] = encoded
        image["mimeType"] = MIME[used]
        report.append({
            "image": index, "kind": "data" if is_data else "colour",
            "resolution": before_size, "before": len(blob), "after": len(encoded),
            "skipped": False,
        })

    if not replacements and not dropped:
        return data, report

    # Rebuild the buffer so nothing is left orphaned: every view is copied in order,
    # with the new image bytes substituted and offsets recomputed. This is also what
    # actually removes the bytes of a dropped map.
    rebuilt = bytearray()
    for view_index, view in enumerate(views):
        payload = replacements.get(view_index)
        if payload is None:
            start = view.get("byteOffset", 0)
            payload = binary[start : start + view["byteLength"]]
        rebuilt += b"\x00" * (-len(rebuilt) % 4)
        view["byteOffset"] = len(rebuilt)
        view["byteLength"] = len(payload)
        rebuilt += payload

    document["buffers"] = [{"byteLength": len(rebuilt)}]

    if fmt == "webp":
        for texture_index, texture in enumerate(document.get("textures", [])):
            source = texture.get("source")
            if source is not None and images[source].get("mimeType") == MIME["webp"]:
                declare_webp(document, texture_index, source)

    return build_glb(document, bytes(rebuilt)), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("asset", type=Path, help="GLB to read")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument("--format", choices=sorted(MIME), default="jpeg")
    parser.add_argument(
        "--quality", type=int, default=90,
        help="Quality for colour textures. 90 held 36.3 dB on the Snag's albedo",
    )
    parser.add_argument(
        "--map-quality", type=int, default=90,
        help="Quality for maps the shader reads as numbers (metallic-roughness, normal, "
             "occlusion), where artifacts show as shading rather than blur",
    )
    parser.add_argument(
        "--max-size", type=int, default=2048,
        help="Cap the longest side of colour textures; 0 keeps the resolution. The paint "
             "stage emits 4096, which is more than these assets carry: at 2048 the "
             "rendered difference is within the renderer's own sampling noise",
    )
    parser.add_argument(
        "--map-size", type=int, default=1024,
        help="Cap for data maps. These carry far less detail than an albedo and a 4096 "
             "metallic-roughness map is almost always 14 MB of nothing. 0 keeps it",
    )
    parser.add_argument(
        "--normal-size", type=int, default=2048,
        help="Cap for the normal map, which carries the crisp edges of a light mesh and is "
             "worth more than the other data maps (0: the same cap as --map-size)",
    )
    parser.add_argument(
        "--drop-surface-maps", action="store_true",
        help="Remove the normal and metallic-roughness maps instead of resizing them: flat "
             "and matte, and smaller. The material keeps the base colour and --roughness",
    )
    parser.add_argument(
        "--roughness", type=float, default=0.65,
        help="The fixed roughness a dropped surface map leaves behind (--drop-surface-maps)",
    )
    args = parser.parse_args()

    data = args.asset.read_bytes()
    out, report = compress(
        data,
        fmt=args.format,
        quality=args.quality,
        map_quality=args.map_quality,
        max_size=args.max_size or None,
        map_size=args.map_size or None,
        normal_size=args.normal_size or None,
        drop_surface_maps=args.drop_surface_maps,
        roughness=args.roughness,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)

    for row in report:
        note = " (kept, re-encoding was larger)" if row["skipped"] else ""
        resolution = row.get("resolution")
        where = f"{resolution[0]}x{resolution[1]} " if resolution else ""
        print(
            f"  image {row['image']} [{row['kind']}] {where}"
            f"{row['before'] / 1048576:.2f} -> {row['after'] / 1048576:.2f} MB{note}"
        )
    if args.drop_surface_maps:
        print("  surface maps dropped: normal and metallic-roughness are gone, flat and matte")
    print(
        f"{args.asset.name}: {len(data) / 1048576:.1f} MB -> "
        f"{len(out) / 1048576:.1f} MB ({100 * len(out) / len(data):.0f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
