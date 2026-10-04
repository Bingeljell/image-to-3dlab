#!/usr/bin/env python3
"""Finish every prop from a prop sheet: LODs re-baked from the original, then compressed.

    python scripts/finish_props.py SPLIT_DIR OUT_DIR [--lods 5000,2500,1000]

Takes the per-prop GLBs `scripts/blender_split_props.py` wrote and gives each prop levels
of detail (LODs): lighter copies a game swaps in as the prop gets further away. The
walkthrough is in `docs/prop-sheets.md`.

**Every LOD is baked from the original, not simplified from LOD0.** Measured on the
barrel of the test sheet: meshoptimizer's simplifier took LOD0 from 5,000 triangles to
1,000 by collapsing edges across texture seams, and the iron hoops came out blotched. A
normal map re-baked for that mesh did not help, because the texture coordinates
themselves had been stretched. Running `blender_retopo_bake.py` again at 1,000 faces,
from the original, with its own atlas, came out clean on all nine props. The cost is one
atlas per LOD rather than one shared between them, so each LOD after the first gets half
the atlas of the one before (down to 256): a far LOD is small on screen.

**Then the detail bake Finish uses.** `blender_bake_detail.py` bakes the original's
relief into a normal map and carries its metallic-roughness map across, so the iron
bands stay dark metal and the wood stays matte instead of sharing one flat sheen.
`--metallic`, `--roughness` and `--ior` only matter where the source has no such map.

**LOD sizes are targets.** They are what the bake is asked for. When QuadriFlow runs it
makes quads, two triangles each, so a LOD can hold up to twice as many triangles; when
it declines, as it did on every prop measured, the fallback lands on triangles. The
record lists the triangles each file really holds.

**Then gltfpack, for size.** meshoptimizer's `gltfpack` reorders each mesh for the GPU,
compresses it and re-encodes its textures as WebP, which is where the weight is: the
chest's LOD0 went from 3.7 MB to 495 KB. Those files need `KHR_mesh_quantization`,
`EXT_meshopt_compression` and `EXT_texture_webp`, which three.js reads; check your
engine before shipping them. The uncompressed GLBs are always kept beside them. Each
LOD's node is named after it (`chest_LOD0`), in both files.

gltfpack is optional and never downloaded by this script: put it on PATH or pass
`--gltfpack`. Use a native release build (github.com/zeux/meshoptimizer/releases); the
npm build cannot write WebP.

Writes `OUT_DIR/<prop>/<prop>_LOD<n>.glb`, the compressed `<prop>_LOD<n>.web.glb` when
gltfpack is found, and a record of the run in `OUT_DIR/finish_props.json`. The record is
written before the first bake, so `--resume` can refuse to mix LODs baked with other
settings into this run. Its paths are relative to OUT_DIR, so it stays right when the
folder moves. Progress goes out as `FINISH::{...}` lines, for the viewer's Props tab.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from compress_glb_textures import build_glb, parse_glb  # noqa: E402
from retopo_repaint import _run, bake_command, preflight, retopo_command, reuse  # noqa: E402

from image_to_3dlab.blender import find_blender  # noqa: E402

# 5,000 triangles kept the chest's rivets and lock readable with a normal map; 2,500 and
# 1,000 are the middle and far steps. `blender_retopo_bake.py` accepts 1,000 to 200,000.
DEFAULT_LODS = (5000, 2500, 1000)
FACE_RANGE = (1000, 200000)
# 1024 is a common prop texture size, and what every measurement here used.
DEFAULT_ATLAS = 1024
ATLAS_SIZES = (1024, 2048, 4096)
# --atlas is LOD0's. Each further LOD halves it down to this floor: a far LOD fills few
# pixels on screen, and one full-size atlas per LOD made LOD2 about as big as LOD0.
MIN_LOD_ATLAS = 256
# `blender_retopo_bake.py`'s own defaults for the unwrap angle and voxel size.
ANGLE = 89.0
VOXEL = 0.004
# -kn keeps the named node, so the prop's name survives compression.
GLTFPACK_FLAGS = ("-cc", "-tw", "-kn")
# What a LOD on disk was baked with. `--resume` refuses to keep LODs when any differ.
BAKE_SETTINGS = ("lods", "atlas", "lod_atlas", "surface")
# The line `blender_bake_detail.py` ends with: what it did, as JSON.
DETAIL_MARKER = "BAKE_DETAIL::"
# What this prints as it goes, for the viewer's progress bar: see progress_line.
PROGRESS_MARKER = "FINISH::"


def parse_lods(text: str) -> list[int]:
    """``"5000,2500,1000"`` to ``[5000, 2500, 1000]``: in range, and each smaller than the last."""
    try:
        lods = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--lods wants comma-separated face counts, got {text!r}") from None
    if not lods:
        raise SystemExit("--lods needs at least one face count")
    low, high = FACE_RANGE
    for faces in lods:
        if not low <= faces <= high:
            raise SystemExit(f"each LOD must be {low:,}..{high:,} faces, got {faces:,}")
    if any(later >= earlier for earlier, later in pairwise(lods)):
        raise SystemExit(f"LODs go from most detailed to least, got {lods}")
    return lods


def collect_props(source: Path) -> list[Path]:
    """The GLBs to finish: every ``*.glb`` in a directory, sorted, or the one file given."""
    if source.is_dir():
        found = sorted(source.glob("*.glb"))
    elif source.suffix.lower() == ".glb" and source.is_file():
        found = [source]
    else:
        raise SystemExit(f"not a GLB or a directory: {source}")
    if not found:
        raise SystemExit(f"no GLBs to finish in {source}")
    return found


def lod_path(out_dir: Path, name: str, index: int, web: bool = False) -> Path:
    suffix = ".web.glb" if web else ".glb"
    return out_dir / name / f"{name}_LOD{index}{suffix}"


def find_gltfpack(explicit: Path | None = None, which=shutil.which) -> Path | None:
    """``--gltfpack`` if given, else ``gltfpack`` on PATH, else ``vendor/gltfpack/``."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--gltfpack {explicit} does not exist")
        return explicit
    on_path = which("gltfpack")
    if on_path:
        return Path(on_path)
    for name in ("gltfpack", "gltfpack.exe"):  # .exe: what scripts/bootstrap_gltfpack.py
        vendored = REPO / "vendor" / "gltfpack" / name  # installs on Windows
        if vendored.is_file():
            return vendored
    return None


def gltfpack_command(binary: Path, source: Path, output: Path) -> list[str]:
    """Compress mesh and textures; no simplification, the LODs are already baked."""
    return [str(binary), "-i", str(source), "-o", str(output), *GLTFPACK_FLAGS]


def name_lod(glb: bytes, name: str) -> bytes:
    """Name every mesh node, and its mesh, after the LOD.

    The bake names its object RETOPO, which is what an engine would otherwise show.
    """
    document, binary = parse_glb(glb)
    for node in document.get("nodes", []):
        if "mesh" in node:
            node["name"] = name
    for mesh in document.get("meshes", []):
        mesh["name"] = name
    return build_glb(document, binary)


def write_lod(detailed: Path, lod: Path, name: str) -> None:
    """Name the detail bake's output after its LOD, then move it to the LOD's path.

    Named in logs/ and moved in whole, so no reader, and no --resume, ever finds a LOD
    that is half written or still called RETOPO.
    """
    detailed.write_bytes(name_lod(detailed.read_bytes(), name))
    detailed.replace(lod)


def lod_atlas(atlas: int, index: int) -> int:
    """The atlas size LOD `index` is baked at: LOD0's halved once per step, floored."""
    return max(MIN_LOD_ATLAS, atlas >> index)


def glb_triangles(glb: bytes) -> int:
    """How many triangles a GLB's meshes hold, from the accessor counts alone."""
    document, _binary = parse_glb(glb)
    accessors = document.get("accessors", [])
    total = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                continue
            source = primitive.get("indices", primitive.get("attributes", {}).get("POSITION"))
            if source is not None:
                total += accessors[source]["count"] // 3
    return total


def detail_record(log: str) -> dict:
    """What the detail bake reported, from its log: the metallic-roughness it moved, how
    well the two meshes lined up, and so on. Empty when it said nothing."""
    for line in reversed(log.splitlines()):
        if line.startswith(DETAIL_MARKER):
            try:
                report = json.loads(line[len(DETAIL_MARKER):])
            except ValueError:
                return {}
            keep = ("metallic_roughness", "spread", "normal_flipped_fraction",
                    "zero_tangents_repaired")
            return {key: report[key] for key in keep if key in report}
    return {}


def resume_mismatch(previous: dict | None, settings: dict) -> list[str]:
    """The bake settings that differ from the run being resumed.

    No record means nothing to compare against, as with LODs from before records were
    written first, and those are kept as they are.
    """
    if not isinstance(previous, dict):
        return []
    return [key for key in BAKE_SETTINGS if previous.get(key) != settings[key]]


def progress_line(prop: str, *, lod: int | None = None, done: str | None = None) -> str:
    """`FINISH::{"prop": ..., "lod": n}` as a LOD starts to bake, and `"done": <sizes>`
    once a prop is finished. The Props tab reads these rather than the log lines around
    them, which are for people and free to change."""
    fields: dict = {"prop": prop}
    if lod is not None:
        fields["lod"] = lod
    if done is not None:
        fields["done"] = done
    return PROGRESS_MARKER + json.dumps(fields)


def read_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="the directory blender_split_props.py wrote, or one prop GLB")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lods", default=",".join(map(str, DEFAULT_LODS)),
                        help="faces per LOD, most detailed first")
    parser.add_argument("--atlas", type=int, default=DEFAULT_ATLAS, choices=ATLAS_SIZES)
    surface = "only where the source has no metallic-roughness map"
    parser.add_argument("--metallic", type=float, default=0.25, help=surface)
    parser.add_argument("--roughness", type=float, default=0.65, help=surface)
    parser.add_argument("--ior", type=float, default=1.45)
    parser.add_argument("--gltfpack", type=Path, default=None)
    parser.add_argument("--no-compress", action="store_true",
                        help="skip gltfpack even when it is available")
    parser.add_argument("--resume", action="store_true",
                        help="keep LODs already written instead of baking them again; "
                             "refused if they were baked with other settings")
    parser.add_argument("--blender", type=Path, default=None,
                        help="Blender executable; found automatically when omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    props = collect_props(args.source)
    lods = parse_lods(args.lods)
    blender, problem = preflight(args.blender, skip_paint=True, find=find_blender)
    if problem:
        raise SystemExit(problem)
    gltfpack = None if args.no_compress else find_gltfpack(args.gltfpack)
    if gltfpack is None and not args.no_compress:
        print("gltfpack not found: writing uncompressed LODs only (see --help)")

    record = {"lods": lods, "atlas": args.atlas,
              "lod_atlas": [lod_atlas(args.atlas, i) for i in range(len(lods))],
              "gltfpack": str(gltfpack) if gltfpack else None,
              "surface": {"metallic": args.metallic, "roughness": args.roughness,
                          "ior": args.ior},
              "props": []}
    record_path = out_dir / "finish_props.json"
    if args.resume:
        changed = resume_mismatch(read_record(record_path), record)
        if changed:
            raise SystemExit(
                f"--resume: {', '.join(changed)} changed since the LODs in {out_dir} were "
                "baked. Drop --resume to bake them again, or pick another OUT_DIR.")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        record_path.write_text(json.dumps(record, indent=2))

    save()
    started = time.time()
    for source in props:
        name = source.stem
        logs = out_dir / name / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        entry = {"name": name, "source": os.path.relpath(source, out_dir), "lods": []}
        for index, faces in enumerate(lods):
            step = time.time()
            atlas = lod_atlas(args.atlas, index)
            baked = lod_path(out_dir, name, index)
            detail_log = logs / f"LOD{index}.detail.log"
            rebaked = not reuse(baked, args.resume)
            if rebaked:
                print(progress_line(name, lod=index), flush=True)
                # Both in-between meshes stay in logs/ and the LOD appears only once named:
                # --resume keeps any LOD file it finds, so it must never see a half-done one.
                retopo = logs / f"LOD{index}.retopo.glb"
                detailed = logs / f"LOD{index}.detail.glb"
                _run(retopo_command(source, retopo, faces, atlas, ANGLE, VOXEL,
                                    args.metallic, args.roughness, args.ior,
                                    blender=blender),
                     logs / f"LOD{index}.log", f"{name} LOD{index}")
                _run(bake_command(source, retopo, detailed, atlas, blender=blender),
                     detail_log, f"{name} LOD{index} detail")
                retopo.unlink()
                write_lod(detailed, baked, f"{name}_LOD{index}")
            lod = {"faces": faces, "atlas": atlas, "triangles": glb_triangles(baked.read_bytes()),
                   "glb": os.path.relpath(baked, out_dir), "bytes": baked.stat().st_size}
            if detail_log.is_file():
                lod["detail"] = detail_record(detail_log.read_text())
            if gltfpack is not None:
                packed = lod_path(out_dir, name, index, web=True)
                # A LOD baked again this run needs packing again, whatever is on disk.
                if rebaked or not reuse(packed, args.resume):
                    unpacked = logs / f"LOD{index}.web.glb"   # moved in whole, as the LOD is
                    _run(gltfpack_command(gltfpack, baked, unpacked),
                         logs / f"LOD{index}.gltfpack.log", f"{name} LOD{index} gltfpack")
                    unpacked.replace(packed)
                lod.update(web_glb=os.path.relpath(packed, out_dir),
                           web_bytes=packed.stat().st_size)
            lod["seconds"] = round(time.time() - step, 1)
            entry["lods"].append(lod)
        record["props"].append(entry)
        save()
        sizes = ", ".join(f"LOD{i} {lod.get('web_bytes', lod['bytes']) / 1024:,.0f} KB"
                          for i, lod in enumerate(entry["lods"]))
        print(f"{name}: {sizes}", flush=True)
        print(progress_line(name, done=f"{name}: {sizes}"), flush=True)

    record["total_seconds"] = round(time.time() - started, 1)
    save()
    print(f"finished {len(props)} props in {record['total_seconds']:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
