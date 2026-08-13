#!/usr/bin/env python3
"""Re-attach the metallicRoughness map that `--material-mode matte` orphaned.

    python scripts/restore_pbr_material.py asset.glb --output asset_pbr.glb

**Why this can work at all.** `matte` mode only rewrote the GLB's JSON chunk — it deleted
the material's *reference* to the metallic-roughness texture and left the image itself
sitting in the binary buffer, untouched. So the map is still inside every asset we shipped;
it is merely unreachable. Restoring it is a JSON edit, not a 16-minute regeneration.

**Why it matters.** The Hugging Face reference assets ship `metallicFactor: 1.0` plus a
3072x3072 metallicRoughness map and `doubleSided: false`. Ours ship `metallicFactor: 0.0`,
no map, and `doubleSided: true` — mathematically flat under any light, and double-sided so
that a hollow mesh still looks solid in preview. See docs/hunyuan-eval-2026-08-13.md.

The orphaned texture is found by elimination: any texture no material references. If that
is ambiguous the script refuses and asks for `--mr-texture`, because guessing which image
is the metallic-roughness map would silently wire a base colour into the metalness channel.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.trellis_backend import read_glb, write_glb


def referenced_textures(gltf: dict) -> set[int]:
    """Every texture index reachable from a material.

    Covers the five glTF 2.0 core texture slots. A texture referenced from any of them is
    in use; anything left over is what `matte` orphaned.
    """
    used: set[int] = set()
    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        for slot in ("baseColorTexture", "metallicRoughnessTexture"):
            if slot in pbr:
                used.add(pbr[slot]["index"])
        for slot in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if slot in material:
                used.add(material[slot]["index"])
    return used


def find_orphan_texture(gltf: dict) -> int | None:
    """The single unreferenced texture, or None if there is not exactly one.

    Returning None for "several" rather than picking the first is deliberate: a wrong
    guess here is invisible in a viewport and wrong in an engine.
    """
    orphans = set(range(len(gltf.get("textures", [])))) - referenced_textures(gltf)
    return orphans.pop() if len(orphans) == 1 else None


def restore_material(
    gltf: dict,
    mr_texture: int | None = None,
    *,
    single_sided: bool = True,
) -> list[str]:
    """Re-attach the MR map and undo matte's flattening. Returns what changed.

    `metallicFactor` goes back to 1.0 because in glTF the factor *multiplies* the texture:
    left at 0.0 the restored map would be scaled to nothing and the change would appear to
    do nothing at all.
    """
    changes: list[str] = []
    materials = gltf.get("materials", [])
    if not materials:
        raise ValueError("GLB has no materials")

    index = mr_texture if mr_texture is not None else find_orphan_texture(gltf)
    if index is None:
        raise ValueError(
            "could not identify the metallicRoughness texture by elimination; "
            "pass --mr-texture N explicitly"
        )
    if not 0 <= index < len(gltf.get("textures", [])):
        raise ValueError(f"texture index {index} out of range")

    for position, material in enumerate(materials):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" not in pbr:
            pbr["metallicRoughnessTexture"] = {"index": index}
            changes.append(f"material[{position}]: attached metallicRoughnessTexture={index}")
        if pbr.get("metallicFactor") != 1.0:
            pbr["metallicFactor"] = 1.0
            changes.append(f"material[{position}]: metallicFactor -> 1.0")
        if pbr.get("roughnessFactor") != 1.0:
            pbr["roughnessFactor"] = 1.0
            changes.append(f"material[{position}]: roughnessFactor -> 1.0")
        if single_sided and material.get("doubleSided"):
            material["doubleSided"] = False
            changes.append(f"material[{position}]: doubleSided -> false")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path, help="write here instead of in place")
    parser.add_argument("--mr-texture", type=int, help="texture index of the MR map")
    parser.add_argument(
        "--keep-double-sided",
        action="store_true",
        help="leave doubleSided alone; by default it is turned off so culling tells the truth",
    )
    args = parser.parse_args()

    target = args.asset
    if args.output:
        shutil.copyfile(args.asset, args.output)
        target = args.output

    gltf, chunks, json_index, version = read_glb(target)
    changes = restore_material(
        gltf, args.mr_texture, single_sided=not args.keep_double_sided
    )
    write_glb(target, gltf, chunks, json_index, version)

    print(f"{target}: {len(changes)} change(s)")
    for change in changes:
        print(f"  {change}")
    if not changes:
        print("  (already PBR — nothing to do)")


if __name__ == "__main__":
    main()
