"""A copy of a GLB that Roblox Studio imports cleanly, plus a check against Roblox's limits.

What changes, and why (limits from create.roblox.com's mesh, accessory and texture specs):

* **Size.** Roblox's 3D Importer reads a file's units as studs unless told otherwise, and
  glTF is in metres, so a 1.8 m character arrives 1.8 studs tall: a doll. The scale is
  baked into the points, the joint positions and any moves, so the default import is life
  size. (A scaled parent node is ignored: tried in Studio, 2026-11-16.) Nothing else
  moves: glTF is already +Y up and +Z front, as Roblox expects.
* **Pictures.** Every texture becomes PNG, at most `max_texture` on a side. WebP (what
  gltfpack writes) is converted; KTX2 cannot be decoded here and is refused.
* **Separate maps.** Roblox's SurfaceAppearance takes colour, normal, roughness and
  metalness as four pictures. glTF packs roughness (green) and metal (blue) into one, and
  embedded maps are reported to come in wrong, so the four are also written loose.
* **Packed geometry is refused.** meshopt and Draco compression are not read by the
  importer; export from the uncompressed file the lab keeps beside the packed one.

The check never changes the model's detail. Too many triangles is reported, not fixed.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_to_3dlab.glb_turn import build_glb, parse_glb

# Roblox measures in studs. Its own docs and importer treat one stud as 0.28 m.
METRES_PER_STUD = 0.28
STUDS_PER_METRE = 1 / METRES_PER_STUD
MESH_TRIANGLES = 20_000       # any one mesh
ACCESSORY_TRIANGLES = 4_000   # a hat, wings: one mesh, rigid or layered
MARKETPLACE_TEXTURE = 2048
MAX_INFLUENCES = 4            # bones pulling on one point
PACKED = {"EXT_meshopt_compression": "meshopt", "KHR_draco_mesh_compression": "Draco"}
KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"
TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = 4, 5, 6
KINDS = ("prop", "character", "accessory")  # character: has a skeleton
# Roblox Studio plays a non-R15 skeleton's moves turned half round (tried 2026-11-16), so the
# copy carries the skeleton without them: posable, but still.
MOVES_LEFT_OUT = "Comes in standing still: Roblox can't play this skeleton's moves."
# asset.extras key holding the scale already baked in, so a second pass is a no-op.
SCALED_KEY = "image_to_3dlab_roblox_scale"
FLOAT = 5126


class RobloxExportError(ValueError):
    """The file cannot be made Roblox-ready without losing something; the message says why."""


@dataclass
class RobloxExport:
    glb: bytes
    maps: dict[str, bytes]                       # file name -> PNG bytes
    checks: list[dict[str, str]]                 # {"level": pass|warn|fail, "text": ...}
    size_studs: tuple[float, float, float] | None
    triangles: int
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c["level"] == "fail" for c in self.checks)


# ---------------------------------------------------------------- counting

def _triangles(mode: int, count: int) -> int:
    if mode == TRIANGLES:
        return count // 3
    if mode in (TRIANGLE_STRIP, TRIANGLE_FAN):
        return max(count - 2, 0)
    return 0


def mesh_triangles(document: dict) -> list[tuple[str, int]]:
    """(name, triangles) for every mesh a node places. A mesh placed twice is listed twice."""
    meshes = document.get("meshes", [])
    accessors = document.get("accessors", [])
    placed = [node["mesh"] for node in document.get("nodes", []) if "mesh" in node]
    out = []
    for index in placed or range(len(meshes)):
        mesh = meshes[index]
        total = 0
        for primitive in mesh.get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            corners = accessors[position]["count"] if position is not None else 0
            if "indices" in primitive:
                corners = accessors[primitive["indices"]]["count"]
            total += _triangles(primitive.get("mode", TRIANGLES), corners)
        out.append((mesh.get("name") or f"mesh {index + 1}", total))
    return out


def bounds_metres(document: dict) -> tuple[float, float, float] | None:
    """Width, height, depth from the vertex min/max glTF stores. Node transforms are not
    applied: the lab's GLBs carry none on their meshes, and a skinned mesh ignores its own."""
    accessors = document.get("accessors", [])
    low, high = [float("inf")] * 3, [float("-inf")] * 3
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            index = primitive.get("attributes", {}).get("POSITION")
            accessor = accessors[index] if index is not None else {}
            if "min" not in accessor or "max" not in accessor:
                continue
            low = [min(a, b) for a, b in zip(low, accessor["min"], strict=True)]
            high = [max(a, b) for a, b in zip(high, accessor["max"], strict=True)]
    if low[0] == float("inf"):
        return None
    return tuple(h - lo for lo, h in zip(low, high, strict=True))  # type: ignore[return-value]


def _has_extra_influences(document: dict) -> bool:
    return any("JOINTS_1" in primitive.get("attributes", {})
               for mesh in document.get("meshes", []) for primitive in mesh.get("primitives", []))


# ---------------------------------------------------------------- the check

def check(document: dict, kind: str = "prop") -> list[dict[str, str]]:
    """Plain-language pass/warn/fail lines for one GLB, against Roblox's published limits."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    lines: list[dict[str, str]] = []

    def say(level: str, text: str) -> None:
        lines.append({"level": level, "text": text})

    used = set(document.get("extensionsUsed", []))
    packed = sorted(name for ext, name in PACKED.items() if ext in used)
    if packed:
        say("fail", "This is a compressed copy, which Roblox can't read. "
                    "Export the model itself (the file without .web in its name).")

    meshes = mesh_triangles(document)
    total = sum(count for _, count in meshes)
    too_big = [(name, count) for name, count in meshes if count > MESH_TRIANGLES]
    for name, count in too_big:
        say("fail", f"{name} has {count:,} triangles; Roblox takes at most {MESH_TRIANGLES:,} per mesh.")
    if not too_big:
        say("pass", f"{total:,} triangles: inside Roblox's {MESH_TRIANGLES:,} per mesh.")

    if kind == "accessory":
        if len(meshes) > 1:
            say("fail", f"A wearable item must be one mesh; this has {len(meshes)}.")
        if total > ACCESSORY_TRIANGLES:
            say("fail", f"{total:,} triangles is over the {ACCESSORY_TRIANGLES:,} a wearable item may have.")
    elif kind == "prop" and total <= ACCESSORY_TRIANGLES and len(meshes) == 1:
        # Good news only: a prop over the limit is still a fine prop, so it gets no line.
        say("pass", f"Light enough to sell as a wearable item too (Roblox allows {ACCESSORY_TRIANGLES:,} triangles).")

    if _has_extra_influences(document):
        say("warn", f"Some points follow more than {MAX_INFLUENCES} bones; Roblox keeps only {MAX_INFLUENCES}, "
                    "so bends may look slightly different.")
    return lines


# ---------------------------------------------------------------- pictures

def _view_bytes(document: dict, binary: bytes, view_index: int) -> bytes:
    view = document["bufferViews"][view_index]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def _decode(data: bytes, index: int) -> Image.Image:
    if data.startswith(KTX2_MAGIC):
        raise RobloxExportError("its pictures are compressed in a way Roblox can't read. "
                                "Export the model itself (the file without .web in its name).")
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _texture_image(document: dict, texture_index: int | None) -> int | None:
    """The image a texture shows, preferring the WebP source gltfpack points at."""
    if texture_index is None:
        return None
    texture = document["textures"][texture_index]
    webp = (texture.get("extensions") or {}).get("EXT_texture_webp", {})
    return webp.get("source", texture.get("source"))


def _repack(document: dict, binary: bytes, replaced: dict[int, bytes]) -> bytes:
    """New binary with some bufferViews' bytes swapped; every view re-laid out in order."""
    out = bytearray()
    for index, view in enumerate(document.get("bufferViews", [])):
        data = replaced.get(index, _view_bytes(document, binary, index))
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(data)
        view["buffer"] = 0
        out += data
    if document.get("buffers"):
        document["buffers"] = [{"byteLength": len(out)}]
    return bytes(out)


def _drop_extension(document: dict, name: str) -> None:
    for key in ("extensionsUsed", "extensionsRequired"):
        if name in document.get(key, []):
            document[key] = [e for e in document[key] if e != name]
            if not document[key]:
                del document[key]


def convert_pictures(document: dict, binary: bytes, max_texture: int) -> tuple[bytes, list[str]]:
    """Every embedded picture as PNG, longest side at most `max_texture`. Mutates `document`."""
    notes: list[str] = []
    replaced: dict[int, bytes] = {}
    for index, entry in enumerate(document.get("images", [])):
        view = entry.get("bufferView")
        if view is None:
            continue
        raw = _view_bytes(document, binary, view)
        image = _decode(raw, index)
        fitted = _fit(image, max_texture)
        is_png = entry.get("mimeType") == "image/png"
        if fitted is image and is_png:
            continue
        if fitted is not image:
            notes.append(f"picture {index + 1} shrunk from {image.width}x{image.height} to "
                         f"{fitted.width}x{fitted.height}")
        elif not is_png:
            notes.append(f"picture {index + 1} converted to PNG")
        see_through = "A" in fitted.getbands() or "transparency" in fitted.info or fitted.mode in ("P", "PA")
        replaced[view] = _png(fitted.convert("RGBA" if see_through else "RGB"))
        entry["mimeType"] = "image/png"
    for texture in document.get("textures", []):
        webp = (texture.get("extensions") or {}).pop("EXT_texture_webp", None)
        if webp is not None:
            texture["source"] = webp["source"]
        if texture.get("extensions") == {}:
            del texture["extensions"]
    _drop_extension(document, "EXT_texture_webp")
    return _repack(document, binary, replaced), notes


def _safe_name(name: str | None) -> str:
    """A material name fit for a file name on any system: letters, digits, _ and - only."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:40]
    return cleaned or "material"


def split_maps(document: dict, binary: bytes) -> dict[str, bytes]:
    """The loose pictures SurfaceAppearance takes, per material: colour, normal, roughness
    (glTF's green channel) and metalness (blue). Call after convert_pictures."""
    materials = document.get("materials", [])
    maps: dict[str, bytes] = {}
    for index, material in enumerate(materials):
        prefix = "" if len(materials) == 1 else f"{index + 1}_{_safe_name(material.get('name'))}_"
        pbr = material.get("pbrMetallicRoughness", {})
        slots = {
            "color": (pbr.get("baseColorTexture") or {}).get("index"),
            "normal": (material.get("normalTexture") or {}).get("index"),
        }
        for name, texture in slots.items():
            image = _texture_image(document, texture)
            if image is not None and "bufferView" in document["images"][image]:
                maps[f"{prefix}{name}.png"] = _view_bytes(document, binary,
                                                          document["images"][image]["bufferView"])
        packed = _texture_image(document, (pbr.get("metallicRoughnessTexture") or {}).get("index"))
        if packed is not None and "bufferView" in document["images"][packed]:
            raw = _view_bytes(document, binary, document["images"][packed]["bufferView"])
            rgb = Image.open(io.BytesIO(raw)).convert("RGB")
            _, green, blue = rgb.split()
            maps[f"{prefix}roughness.png"] = _png(green)
            maps[f"{prefix}metalness.png"] = _png(blue)
    return maps


# ---------------------------------------------------------------- scale

def _float_rows(document: dict, out: bytearray, index: int):
    """An accessor's values as a writable (count, width) view into `out`."""
    accessor = document["accessors"][index]
    if accessor.get("componentType") != FLOAT or "sparse" in accessor or "bufferView" not in accessor:
        raise RobloxExportError("part of this model is stored compressed, so it can't be resized. "
                                "Export the model itself (the file without .web in its name).")
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return np.ndarray((accessor["count"], width), dtype="<f4", buffer=out, offset=start,
                      strides=(view.get("byteStride", width * 4), 4))


def scale_to_studs(document: dict, binary: bytes, factor: float = STUDS_PER_METRE) -> bytes:
    """Scale the whole model by `factor` (metres -> studs by default), baked into the data.
    Mutates `document`; returns the new binary. Already scaled: returned unchanged.

    Uniform scale is the same as multiplying every distance: points (and blend-shape
    offsets), node and animated translations, and the translation part of skin
    inverse-bind matrices. Rotations and node scales stay as they are."""
    asset = document.setdefault("asset", {"version": "2.0"})
    if not isinstance(asset.get("extras"), dict):
        asset["extras"] = {}
    extras = asset["extras"]
    if SCALED_KEY in extras:
        return binary
    k = factor
    positions: set[int] = set()
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            for source in [primitive.get("attributes", {}), *primitive.get("targets", [])]:
                if "POSITION" in source:
                    positions.add(source["POSITION"])
    translations = {sampler["output"]
                    for animation in document.get("animations", [])
                    for channel in animation.get("channels", [])
                    if channel.get("target", {}).get("path") == "translation"
                    for sampler in [animation["samplers"][channel["sampler"]]]}
    bind = {skin["inverseBindMatrices"] for skin in document.get("skins", [])
            if "inverseBindMatrices" in skin}
    out = bytearray(binary)
    for index in sorted(positions | translations):
        _float_rows(document, out, index)[:] *= k
        accessor = document["accessors"][index]
        for key in ("min", "max"):
            if key in accessor:
                accessor[key] = [v * k for v in accessor[key]]
    for index in sorted(bind):
        _float_rows(document, out, index)[:, 12:15] *= k  # column-major: last column
    for node in document.get("nodes", []):
        if "translation" in node:
            node["translation"] = [v * k for v in node["translation"]]
        if "matrix" in node:
            node["matrix"] = [v * k if 12 <= i < 15 else v for i, v in enumerate(node["matrix"])]
    extras[SCALED_KEY] = k
    return bytes(out)


# ---------------------------------------------------------------- bones

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions; either side may be (4,) or (n, 4)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, float), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, float) * [-1, -1, -1, 1]


def _qmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _append(document: dict, out: bytearray, values: np.ndarray, accessor_type: str) -> int:
    """A new float accessor holding `values`; returns its index."""
    out.extend(b"\0" * (-len(out) % 4))
    data = np.ascontiguousarray(values, dtype="<f4").tobytes()
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(out), "byteLength": len(data)})
    out.extend(data)
    flat = np.asarray(values, float).reshape(len(values), -1)
    accessor = {"bufferView": len(document["bufferViews"]) - 1, "componentType": FLOAT,
                "count": len(values), "type": accessor_type}
    if accessor_type in ("SCALAR", "VEC3"):
        accessor["min"], accessor["max"] = flat.min(0).tolist(), flat.max(0).tolist()
    document["accessors"].append(accessor)
    return len(document["accessors"]) - 1


def straighten_bones(document: dict, binary: bytes) -> tuple[bytes, str | None]:
    """Rebuild the skeleton so every bone rests with no turn, keeping the model and its
    moves exactly as they were. Mutates `document`; returns the new binary and a note.

    Why: Roblox's importer plays a move's turns about the wrong axes on a bone that rests
    turned (tried 2026-11-16: a gnome whose legs rest turned 140 degrees twisted round at
    the hips while its head, resting nearly straight, was fine). With no rest turn there
    is nothing to get the wrong way round.

    Each bone j gets C_j = (its rest turn in the world)^-1, so its new pose is A_j @ C_j.
    Locally that is C_parent^-1 @ L @ C_j: a turn q becomes Q_parent q Q_j^-1 and a
    position t becomes Q_parent t. Inverse bind matrices become C_j^-1 @ IBM_j, so the
    skin follows exactly as before. Skipped (with a note) when the rig uses anything this
    can't carry: matrices, bones that scale, or curved (cubic) keys."""
    skins = document.get("skins") or []
    if not skins:
        return binary, None
    nodes = document["nodes"]
    joints = {j for skin in skins for j in skin["joints"]}
    parent = {c: i for i, node in enumerate(nodes) for c in node.get("children", [])}
    channels = [(a, c) for a in document.get("animations", []) for c in a.get("channels", [])]
    if any("matrix" in nodes[j] for j in joints) or any(
            not np.allclose(nodes[j].get("scale", [1, 1, 1]), 1, atol=1e-4) for j in joints):
        return binary, "bones left as they are: the rig uses scaled or matrix bones"
    if any(a["samplers"][c["sampler"]].get("interpolation") == "CUBICSPLINE" for a, c in channels):
        return binary, "bones left as they are: the moves use curved keys"
    if any(c.get("target", {}).get("path") == "scale" and c["target"].get("node") in joints
           for _, c in channels):
        return binary, "bones left as they are: a move changes a bone's size"

    world: dict[int, np.ndarray] = {}

    def turn(i: int) -> np.ndarray:  # rest turn of node i in the world
        if i not in world:
            own = np.array(nodes[i].get("rotation", [0, 0, 0, 1]), float)
            world[i] = own if i not in parent else _qmul(turn(parent[i]), own)
        return world[i]

    identity = np.array([0.0, 0.0, 0.0, 1.0])

    def fix(i: int) -> np.ndarray:  # the turn C_i^-1 for node i
        return turn(i) if i in joints else identity

    out = bytearray(binary)
    # Keys first: they need the old rest turns, which `turn` has cached from the old nodes.
    for node_index in list(range(len(nodes))):
        turn(node_index)
    for animation, channel in channels:
        node = channel.get("target", {}).get("node")
        if node is None or (node not in joints and parent.get(node) not in joints):
            continue
        sampler = animation["samplers"][channel["sampler"]]
        values = np.array(_float_rows(document, out, sampler["output"]), float)
        lead = fix(parent[node]) if node in parent else identity
        path = channel["target"]["path"]
        if path == "rotation":
            values = _qmul(_qmul(lead, values), _qconj(fix(node)))
            accessor_type = "VEC4"
        elif path == "translation":
            values = values @ _qmat(lead).T
            accessor_type = "VEC3"
        else:
            continue
        output = _append(document, out, values, accessor_type)
        if sum(c["sampler"] == channel["sampler"] for c in animation["channels"]) == 1:
            sampler["output"] = output
        else:  # shared with another track: give this one its own
            animation["samplers"].append(dict(sampler, output=output))
            channel["sampler"] = len(animation["samplers"]) - 1
    for i, node in enumerate(nodes):
        if i not in joints and parent.get(i) not in joints:
            continue
        lead = fix(parent[i]) if i in parent else identity
        rest = np.array(node.get("rotation", [0, 0, 0, 1]), float)
        rotation = _qmul(_qmul(lead, rest), _qconj(fix(i)))
        node["rotation"] = [float(v) for v in rotation / np.linalg.norm(rotation)]
        if "translation" in node:
            node["translation"] = [float(v) for v in _qmat(lead) @ np.array(node["translation"], float)]
    for skin in skins:
        if "inverseBindMatrices" not in skin:
            continue
        rows = np.array(_float_rows(document, out, skin["inverseBindMatrices"]), float)
        matrices = rows.reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major on disk
        for k, j in enumerate(skin["joints"]):
            lift = np.eye(4)
            lift[:3, :3] = _qmat(turn(j))
            matrices[k] = lift @ matrices[k]
        skin["inverseBindMatrices"] = _append(document, out, matrices.transpose(0, 2, 1).reshape(-1, 16), "MAT4")
    document["buffers"][0]["byteLength"] = len(out)
    return bytes(out), "bones rebuilt to rest straight"


# ---------------------------------------------------------------- the whole export

def export(data: bytes, *, kind: str | None = None, max_texture: int = MARKETPLACE_TEXTURE,
           height_studs: float | None = None, longest_studs: float | None = None) -> RobloxExport:
    """The Roblox-ready copy of a GLB. Raises RobloxExportError if it can't be made safely.

    kind: prop | character | accessory; left out, a model with a skeleton is a character.
    longest_studs: make the model's longest side this long (props: a rug or a sword lying down
    has almost no height, so sizing by height would make it hundreds of studs wide).
    height_studs: make the model this tall; wins over longest_studs. Without either, metres become studs, which suits
    rigged characters (the lab sizes those as people) but leaves props at whatever size
    the generator picked, often under a stud."""
    for name, value in (("height_studs", height_studs), ("longest_studs", longest_studs)):
        if value is not None and not 0.05 <= value <= 2048:
            raise ValueError(f"{name} must be 0.05..2048, got {value}")
    if not 64 <= max_texture <= MARKETPLACE_TEXTURE:
        raise ValueError(f"max_texture must be 64..{MARKETPLACE_TEXTURE}, got {max_texture}")
    document, binary = parse_glb(data)
    if SCALED_KEY in ((document.get("asset") or {}).get("extras") or {}):
        raise RobloxExportError("this is already a Roblox copy. Export the original model instead.")
    kind = kind or ("character" if document.get("skins") else "prop")
    checks = check(document, kind)
    if PACKED.keys() & set(document.get("extensionsUsed", [])):
        raise RobloxExportError(next(c["text"] for c in checks if c["level"] == "fail"))
    if any("bufferView" not in image for image in document.get("images", [])):
        checks.append({"level": "warn", "text": "Some pictures are linked from outside the file, so Roblox "
                                                "won't get them. Export the model from the studio instead."})
    size = bounds_metres(document)
    binary, notes = convert_pictures(document, binary, max_texture)
    maps = split_maps(document, binary)
    factor = STUDS_PER_METRE
    if height_studs is not None:
        if not size or size[1] <= 0:
            raise RobloxExportError("can't set a height: the model has no height to measure")
        factor = height_studs / size[1]
    elif longest_studs is not None:
        if not size or max(size) <= 0:
            raise RobloxExportError("can't set a size: the model has no size to measure")
        factor = longest_studs / max(size)
    if document.pop("animations", None):
        checks.append({"level": "note", "text": MOVES_LEFT_OUT})
    binary, bone_note = straighten_bones(document, binary)
    if bone_note:
        notes.append(bone_note)
    binary = scale_to_studs(document, binary, factor)
    studs = tuple(round(v * factor, 2) for v in size) if size else None
    if studs and max(studs) > 2048:
        checks.append({"level": "warn", "text": f"{max(studs):,.0f} studs across: Roblox parts stop at 2,048."})
    return RobloxExport(glb=build_glb(document, binary), maps=maps, checks=checks,
                        size_studs=studs, triangles=sum(n for _, n in mesh_triangles(document)),
                        notes=notes)


HOW_TO = """How to bring this into Roblox Studio

1. Studio: Home > Import 3D, pick model.glb. Leave "Scale Unit" on Studs: the file is already sized in studs.
2. Check "File Dimensions" reads about {size}.
3. Grey at first? Roblox checks uploaded pictures before showing them; it can take a while,
   and a new account may need to verify first (Settings > Account Info on roblox.com).
4. If the colours or shine look wrong, add a SurfaceAppearance to the MeshPart and upload
   color.png, normal.png, roughness.png and metalness.png into its four slots (with several
   materials, each set of four starts with the material's number and name).
5. Limits checked: see check.json.
"""


def write_folder(result: RobloxExport, out_dir: Path) -> Path:
    """Write model.glb, the loose maps, check.json and HOW-TO-IMPORT.txt into `out_dir`."""
    for name in result.maps:
        if Path(name).name != name or name.startswith("."):
            raise RobloxExportError(f"unsafe picture name {name!r}")
    # Built aside, then swapped in: a re-export leaves no pictures from the last one, and a
    # failure halfway leaves the old folder as it was.
    building = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    (building / "model.glb").write_bytes(result.glb)
    for name, png in result.maps.items():
        (building / name).write_bytes(png)
    size = " x ".join(f"{v:g}" for v in result.size_studs) + " studs" if result.size_studs else "the model's own size"
    record: dict[str, Any] = {"passed": result.passed, "triangles": result.triangles,
                              "size_studs": result.size_studs, "checks": result.checks,
                              "notes": result.notes, "maps": sorted(result.maps)}
    (building / "check.json").write_text(json.dumps(record, indent=2) + "\n")
    (building / "HOW-TO-IMPORT.txt").write_text(HOW_TO.format(size=size))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    building.rename(out_dir)
    return out_dir
