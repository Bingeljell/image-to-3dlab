"""Dump what a GLB actually contains, so a reference asset can be diffed against ours.

Written for one job: when a hosted demo produces an asset that looks better than ours,
answer *in what specific respect* rather than by eye. Every number here came from a
question we got wrong by assuming instead of measuring.

    python scripts/glb_forensics.py a.glb b.glb
    python scripts/glb_forensics.py --json a.glb        # machine-readable

The material section matters as much as the geometry: a missing metallicRoughness texture
is invisible in a viewport screenshot and decisive in an engine.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

_GLB_MAGIC = 0x46546C67
_GLB_JSON_CHUNK = 0x4E4F534A


def read_gltf_json(path: Path) -> dict[str, Any]:
    """Return the glTF JSON chunk of a GLB, unmodified by any mesh library.

    Deliberately not routed through trimesh: the loader normalises materials, which is
    exactly the information we are trying to inspect.
    """
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"{path} is not a binary glTF (GLB)")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != _GLB_JSON_CHUNK:
        raise ValueError(f"{path}: first chunk is not JSON")
    return json.loads(data[20 : 20 + chunk_len].decode("utf-8"))


def material_summary(gltf: dict[str, Any]) -> list[dict[str, Any]]:
    """Which PBR channels a GLB actually ships, per material.

    ``textures`` lists channels backed by an image. A material can declare
    ``metallicFactor: 1.0`` and still be flat if no map drives it, so both are reported.
    """
    out = []
    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        textures = []
        if "baseColorTexture" in pbr:
            textures.append("baseColor")
        if "metallicRoughnessTexture" in pbr:
            textures.append("metallicRoughness")
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if key in material:
                textures.append(key.removesuffix("Texture"))
        out.append(
            {
                "name": material.get("name"),
                "textures": textures,
                "doubleSided": material.get("doubleSided", False),
                "alphaMode": material.get("alphaMode", "OPAQUE"),
                "metallicFactor": pbr.get("metallicFactor"),
                "roughnessFactor": pbr.get("roughnessFactor"),
            }
        )
    return out


def geometry_summary(mesh) -> dict[str, Any]:
    """Topology and shape statistics for one trimesh geometry.

    ``boundary_edges`` is reported after merging by position, because splitting a vertex
    for a UV seam creates boundary edges that are not holes — the trap recorded in
    ``mesh-topology-measurement-trap``. It is a diagnostic, never a quality score: a
    reference asset we rate highly carries 224k of them.
    """
    import numpy as np

    merged = mesh.copy()
    # merge_tex/merge_norm are REQUIRED. Bare merge_vertices() preserves UV and normal
    # seams, so a textured mesh stays split along every atlas boundary: on the moss fox it
    # leaves 333,170 vertices where only 136,367 distinct positions exist. Every edge along
    # a seam then counts as a boundary edge, and the hole and non-manifold counts are
    # inflated by more than 2x. This is `mesh-topology-measurement-trap`, and it was in this
    # file's own docstring while the code did the wrong thing.
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    edges = merged.edges_sorted
    _uniq, counts = np.unique(edges, axis=0, return_counts=True)

    areas = mesh.area_faces
    verts = mesh.vertices
    lengths = np.linalg.norm(verts[mesh.edges[:, 0]] - verts[mesh.edges[:, 1]], axis=1)

    # Degenerate or open meshes can yield a nan/inf "volume" rather than raising.
    volume = float(mesh.volume)
    if not np.isfinite(volume):
        volume = None

    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "vertices_merged": len(merged.vertices),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "volume": volume,
        "face_area_ratio": float(areas.max() / np.median(areas)) if np.median(areas) else None,
        "edge_length_cv": float(lengths.std() / lengths.mean()),
    }


def inspect(path: Path) -> dict[str, Any]:
    import trimesh

    gltf = read_gltf_json(path)
    # process=False: trimesh's default merges vertices, which would erase the very
    # topology difference we are measuring.
    loaded = trimesh.load(path, process=False)
    geoms = loaded.geometry if hasattr(loaded, "geometry") else {"geometry_0": loaded}

    return {
        "file": str(path),
        "generator": gltf.get("asset", {}).get("generator"),
        "extensions": gltf.get("extensionsUsed", []),
        "materials": material_summary(gltf),
        "geometry": {name: geometry_summary(g) for name, g in geoms.items()},
        "textures": _texture_sizes(geoms),
    }


def _texture_sizes(geoms) -> dict[str, str]:
    sizes = {}
    for name, geom in geoms.items():
        material = getattr(geom.visual, "material", None)
        if material is None:
            continue
        for attr in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                     "occlusionTexture", "emissiveTexture"):
            image = getattr(material, attr, None)
            if image is not None:
                sizes[f"{name}.{attr}"] = f"{image.size[0]}x{image.size[1]}"
    return sizes


def _print(report: dict[str, Any]) -> None:
    print("=" * 78)
    print(report["file"])
    print("=" * 78)
    print(f"  generator      : {report['generator']!r}")
    if report["extensions"]:
        print(f"  extensions     : {report['extensions']}")
    for i, mat in enumerate(report["materials"]):
        print(f"  material[{i}]    : textures={mat['textures'] or ['NONE']}")
        print(f"                   doubleSided={mat['doubleSided']} "
              f"alphaMode={mat['alphaMode']} "
              f"metallic={mat['metallicFactor']} rough={mat['roughnessFactor']}")
    for name, size in report["textures"].items():
        print(f"  {name:34s}: {size}")
    for name, g in report["geometry"].items():
        print(f"\n  --- {name} ---")
        print(f"  faces / vertices     : {g['faces']:,} / {g['vertices']:,}")
        print(f"  boundary edges       : {g['boundary_edges']:,}  (diagnostic, not a score)")
        print(f"  non-manifold edges   : {g['nonmanifold_edges']:,}")
        print(f"  watertight           : {g['watertight']}")
        print(f"  winding consistent   : {g['winding_consistent']}")
        vol = g["volume"]
        print(f"  volume               : {vol:+.6f}" if vol is not None else
              "  volume               : n/a")
        if vol is not None and vol < 0:
            print("      ^^ NEGATIVE - mesh is inside-out")
        print(f"  face area max/median : {g['face_area_ratio']:.1f}"
              if g["face_area_ratio"] else "  face area max/median : n/a")
        print(f"  edge length cv       : {g['edge_length_cv']:.3f}  (low = uniform/remeshed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args()

    reports = [inspect(p) for p in args.paths]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            _print(report)
            print()


if __name__ == "__main__":
    main()
