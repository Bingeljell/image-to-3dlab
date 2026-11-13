"""Engine bone names for a rigged humanoid, and at most four skin weights a vertex.

The auto-rig names its bones `bone_0` to `bone_N`, which a game engine's humanoid avatar cannot map on
its own (Unity's Humanoid goes by name). `name_bones` finds each body bone with the mapper the preset
moves use (`scripts/kimodo_retarget.py`: by the tree's shape, never by number) and gives it the engine's
name: Hips, Spine, Chest, ... LeftToes. Bones that are not part of the body (hair, a cape, a sword) keep
their names and stay where they hang, under the nearest body bone; so do the fingers of a mitten hand,
whose look-alike chains cannot be told apart.

Only joint names change, plus the skin weights of any vertex with more than four (engines take four).
Animations point at bones by index, so every move keeps working. Each preset is one table in `PRESETS`;
Unity's Humanoid is the first, other engines' names can follow the same way.
"""

from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
MAX_INFLUENCES = 4
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

_FINGERS = (("Index", "Index"), ("Middle", "Middle"), ("Ring", "Ring"), ("Pinky", "Little"))
_PHALANGES = ("Proximal", "Intermediate", "Distal")


def _unity() -> dict[str, str]:
    """SOMA joint (what `map_skintokens_to_soma` returns) -> Unity Humanoid bone name."""
    table = {"Hips": "Hips", "Spine1": "Spine", "Spine2": "Chest", "Chest": "UpperChest",
             "Neck1": "Neck", "Neck2": "Neck", "Head": "Head"}
    for side in ("Left", "Right"):
        table.update({f"{side}Shoulder": f"{side}Shoulder", f"{side}Arm": f"{side}UpperArm",
                      f"{side}ForeArm": f"{side}LowerArm", f"{side}Hand": f"{side}Hand",
                      f"{side}Leg": f"{side}UpperLeg", f"{side}Shin": f"{side}LowerLeg",
                      f"{side}Foot": f"{side}Foot", f"{side}ToeBase": f"{side}Toes"})
        for i, phalanx in enumerate(_PHALANGES):
            table[f"{side}HandThumb{i + 1}"] = f"{side}Thumb{phalanx}"
            # SOMA fingers start at the metacarpal (1); SkinTokens' start at the knuckle (2).
            for soma, unity in _FINGERS:
                table[f"{side}Hand{soma}{i + 2}"] = f"{side}{unity}{phalanx}"
    return table


PRESETS: dict[str, dict[str, str]] = {"unity": _unity()}
LABELS = {"unity": "Unity Humanoid"}
# The bones Unity's Humanoid avatar cannot do without; the rest (neck, chest, toes, fingers) are optional.
REQUIRED = {"unity": ("Hips", "Spine", "Head",
                      *(f"{s}{p}" for s in ("Left", "Right")
                        for p in ("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot")))}


def _preset(preset: str) -> dict[str, str]:
    if preset not in PRESETS:
        raise ValueError(f"bone names preset is one of: {', '.join(PRESETS)} (got {preset!r})")
    return PRESETS[preset]


def engine_names(parents: dict[str, str | None], mapping: dict[str, str], preset: str = "unity") -> dict[str, str]:
    """{bone: engine name} for the bones that get one; every other bone keeps its own name.

    A bone is named when it is the first of its chain with that engine role: the mapper hands extra
    bones (hair, a cape) their parent's role, and a two-bone neck both say Neck. A name that would still
    land on more than one bone (four mitten fingers, all "middle") goes to none of them, because guessing
    would put an index finger in the little finger's slot; and a name a kept bone already has is not
    given again, so no two bones share one.
    """
    table = _preset(preset)
    role = {bone: table.get(soma) for bone, soma in mapping.items()}
    first = {bone: name for bone, name in role.items() if name and role.get(parents.get(bone)) != name}
    counts = Counter(first.values())
    names = {bone: name for bone, name in first.items() if counts[name] == 1}
    taken = set(parents) - set(names)
    return {bone: name for bone, name in names.items() if name not in taken}


def _chunks(glb: bytes) -> tuple[int, dict[str, Any], bytes]:
    if len(glb) < 20 or struct.unpack_from("<I", glb, 0)[0] != GLB_MAGIC:
        raise ValueError("not a GLB file")
    version = struct.unpack_from("<I", glb, 4)[0]
    json_len, kind = struct.unpack_from("<II", glb, 12)
    if kind != JSON_CHUNK:
        raise ValueError("GLB has no JSON chunk first")
    doc = json.loads(glb[20:20 + json_len])
    rest = glb[20 + json_len:]
    binary = b""
    if len(rest) >= 8 and struct.unpack_from("<I", rest, 4)[0] == BIN_CHUNK:
        binary = rest[8:8 + struct.unpack_from("<I", rest, 0)[0]]
    return version, doc, binary


def _pack(version: int, doc: dict[str, Any], binary: bytes) -> bytes:
    chunk = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)  # chunks are 4-byte aligned; the spec pads JSON with spaces
    body = struct.pack("<II", len(chunk), JSON_CHUNK) + chunk
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        body += struct.pack("<II", len(binary), BIN_CHUNK) + binary
    return struct.pack("<III", GLB_MAGIC, version, 12 + len(body)) + body


def rename(glb: bytes, names: dict[str, str]) -> bytes:
    """`glb` with its skin joints renamed per `names`; the binary chunk is kept byte for byte."""
    version, doc, _ = _chunks(glb)
    json_len = struct.unpack_from("<I", glb, 12)[0]
    nodes = doc.get("nodes", [])
    for joint in {j for skin in doc.get("skins", []) for j in skin.get("joints", [])}:
        old = nodes[joint].get("name")
        if old in names:
            nodes[joint]["name"] = names[old]
    chunk = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)
    body = struct.pack("<II", len(chunk), JSON_CHUNK) + chunk + glb[20 + json_len:]
    return struct.pack("<III", GLB_MAGIC, version, 12 + len(body)) + body


_DTYPES = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}


def _view(doc: dict[str, Any], binary: bytearray, index: int) -> np.ndarray:
    """A writable (count, 4) view of a VEC4 accessor in the binary chunk, strides honoured."""
    accessor = doc["accessors"][index]
    if accessor.get("type") != "VEC4" or "sparse" in accessor or "bufferView" not in accessor:
        raise ValueError("skin attributes must be plain VEC4 accessors")
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = _DTYPES[accessor["componentType"]]
    stride = view.get("byteStride") or 4 * dtype.itemsize
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return np.ndarray((accessor["count"], 4), dtype=dtype, buffer=binary, offset=start,
                      strides=(stride, dtype.itemsize))


def _weights(doc: dict[str, Any], values: np.ndarray, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    if accessor["componentType"] == 5126:
        return values.astype(np.float64)
    return values.astype(np.float64) / np.iinfo(values.dtype).max  # normalised bytes or shorts


def limit_influences(glb: bytes, most: int = MAX_INFLUENCES) -> tuple[bytes, int]:
    """(`glb` with at most `most` skin weights a vertex, how many vertices had more).

    The largest weights stay and are scaled to sum to one again; they are written over JOINTS_0 and
    WEIGHTS_0 and the further sets (JOINTS_1, ...) are dropped from the primitive. A file that already
    has four or fewer comes back unchanged.
    """
    version, doc, original = _chunks(glb)
    binary = bytearray(original)
    capped, changed = 0, False
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            sets = sorted(int(k.split("_")[1]) for k in attributes
                          if k.startswith("JOINTS_") and f"WEIGHTS_{k.split('_')[1]}" in attributes)
            if len(sets) < 2 and most >= 4:
                continue
            joints = np.concatenate([_view(doc, binary, attributes[f"JOINTS_{n}"]) for n in sets], axis=1)
            weights = np.concatenate([_weights(doc, _view(doc, binary, attributes[f"WEIGHTS_{n}"]),
                                               attributes[f"WEIGHTS_{n}"]) for n in sets], axis=1)
            capped += int(((weights > 0).sum(axis=1) > most).sum())
            order = np.argsort(-weights, axis=1, kind="stable")[:, :min(most, 4)]
            top_j = np.take_along_axis(joints, order, axis=1)
            top_w = np.take_along_axis(weights, order, axis=1)
            total = top_w.sum(axis=1, keepdims=True)
            top_w = np.divide(top_w, total, out=np.zeros_like(top_w), where=total > 0)
            top_j = np.where(top_w > 0, top_j, 0)
            out_j = _view(doc, binary, attributes["JOINTS_0"])
            if top_j.max(initial=0) > np.iinfo(out_j.dtype).max:
                raise ValueError("JOINTS_0 is too narrow for the joints it would have to hold")
            out_j[:] = top_j
            out_w = _view(doc, binary, attributes["WEIGHTS_0"])
            if out_w.dtype.kind == "f":
                out_w[:] = top_w
            else:  # normalised integers: round, then give the rounding error to the largest weight
                scale = np.iinfo(out_w.dtype).max
                ints = np.rint(top_w * scale).astype(np.int64)
                ints[:, 0] += np.where(total[:, 0] > 0, scale - ints.sum(axis=1), 0)
                out_w[:] = ints
            for n in sets[1:]:
                del attributes[f"JOINTS_{n}"], attributes[f"WEIGHTS_{n}"]
            changed = True
    if not changed:
        return glb, 0
    return _pack(version, doc, bytes(binary)), capped


def name_bones(path: Path, preset: str = "unity") -> dict[str, Any]:
    """Name a rigged GLB's bones for an engine and cap its weights at four, in place.

    Returns a report: the preset, how many bones got a name out of how many, the ones that kept theirs,
    how many vertices had more than four weights, and the engine's required bones it could not find.
    Raises LookupError when the file has no skeleton or the mapper finds no humanoid in it.
    """
    _preset(preset)
    path = Path(path)
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from kimodo_retarget import map_skintokens_to_soma
    from rig_check import Y_UP, skeleton

    parents, heads = skeleton(path)
    try:
        mapping, _ = map_skintokens_to_soma(parents, heads, up=Y_UP)
    except (ValueError, KeyError, IndexError) as exc:
        raise LookupError(f"no humanoid body found in the skeleton ({exc})") from exc
    names = engine_names(parents, mapping, preset)
    data, capped = limit_influences(rename(path.read_bytes(), names))
    path.write_bytes(data)
    return {"preset": preset, "named": len(names), "bones": len(parents),
            "kept": sorted(set(parents) - set(names), key=lambda b: (len(b), b)), "capped": capped,
            "missing": [b for b in REQUIRED[preset] if b not in names.values()]}


def describe(report: dict[str, Any]) -> str:
    """The report in one line for a person."""
    label = LABELS.get(report["preset"], report["preset"])
    line = f"{label} names on {report['named']} of {report['bones']} bones"
    kept = len(report["kept"])
    if kept:
        line += f"; {kept} kept their own (hair, gear, mitten fingers)"
    if report["capped"]:
        line += f"; {report['capped']} vertices trimmed to {MAX_INFLUENCES} weights"
    if report["missing"]:
        line += f"; missing for the avatar: {', '.join(report['missing'])}"
    return line + "."
