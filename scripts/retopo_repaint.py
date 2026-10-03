#!/usr/bin/env python3
"""Finish a generated asset: retopologise, repaint, bake detail, compress — one command.

    python scripts/retopo_repaint.py generated.glb source.png finished.glb

**Why this exists.** The route that works on this repo's assets is four separate commands
run by hand, each with its own flags, in an order that matters:

    blender_retopo_bake.py → run_paint_pbr.py → blender_bake_detail.py → compress_glb_textures.py

Run by hand across several assets, each gets slightly different treatment and the
comparison between them stops meaning anything. It also takes ~8 minutes of attention per
asset, most of it waiting. This runs the chain, keeps every intermediate, and writes a
JSON record of what was actually done.

**What the route is for.** TRELLIS.2's geometry is good and its 1024 texture stage bleaches
flat-illustration inputs. So: take the geometry, throw the texture away, and repaint from
the source image with Hunyuan 2.1 PBR. Retopology in the middle is what makes the repaint
affordable — xatlas takes 12 seconds on a 40k mesh against minutes at 500k, and the paint
stage's face-count ceiling stops being reachable.

**Stages, and why each is where it is:**

1. **Retopologise** (`blender_retopo_bake.py`, headless Blender). Voxel-remesh first, then
   decimate. The ordering is load-bearing: decimating the raw mesh shatters thin geometry,
   measured. Also bakes the original texture across as a fallback, so the intermediate is
   viewable even if the paint stage fails.
2. **Repaint** (`hunyuan_mlx/paint`, its own venv). Reuses `run_paint` from
   `hunyuan_mlx_xiong_generate.py` rather than re-deriving the env-var contract.
3. **Bake detail** (`blender_bake_detail.py`, headless Blender). A normal map from the
   original high-poly, so the retopo keeps its relief, plus the source's
   metallic-roughness map when there was no repaint (a repaint makes its own). After the
   repaint because the repaint re-unwraps: a map baked onto the retopo's UVs would land
   in the wrong place. Seconds, not minutes.
4. **Compress** (`compress_glb_textures.py`). The paint stage emits two uncompressed
   4096² PNGs, which is 30 MB of a 32 MB asset.

**Surface response is per-asset and this does not guess it.** The retopology stage bakes
only base colour, so `--metallic/--roughness/--ior` stand in for the source's material
until the bake-detail stage replaces them with the source's own map (and always, with
`--skip-bake`). The defaults are neutral
organic; the Snag wanted 0.648 / 0.686 / 1.4 by eye. Whatever is used is recorded in the
JSON so the next asset can be judged against a known setting rather than a remembered one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))

from image_to_3dlab.blender import MAC_APP, find_blender, missing_help  # noqa: E402
from image_to_3dlab.host import APPLE, host_platform  # noqa: E402

# Only a default for the command builders; main() finds the real one on every OS.
BLENDER = MAC_APP
REPAINT_OFF_MAC = ("The repaint runs on Apple Silicon only (it uses MLX). Pass --skip-paint: "
                   "Pixel Match still keeps the front exact, and the sides and back keep "
                   "the generator's own paint.")

# The viewer's job API parses these to drive its progress panel. Same contract as
# `scripts/blender_rebind.py`, so one parser serves both.
STAGE_MARKER = "I2L_STAGE"


def emit_stage(phase: str, message: str) -> None:
    """Announce a stage boundary on stdout, for whatever is watching."""
    print(f"{STAGE_MARKER}::{phase}::{message}", flush=True)


def stage_plan(skip_paint: bool, skip_compress: bool, skip_bake: bool = False,
               photo: bool = False) -> list[str]:
    """The stages this run will execute, in order.

    Named separately from the running of them so a caller — the viewer's job API, say —
    can show the sequence before anything starts, and so the order is testable.
    """
    stages = ["retopologise"]
    if not skip_paint:
        stages.append("repaint")
    if photo:
        stages.append("photo")
    if not skip_bake:
        stages.append("bake")
    if not skip_compress:
        stages.append("compress")
    return stages


def headless(blender: Path, script: str) -> list[str]:
    """Blender, headless, running one of this repo's scripts; its arguments follow.

    Blender exits 0 when a ``--python`` script raises unless ``--python-exit-code`` comes
    first, and a stage that crashed would then read as done.
    """
    return [str(blender), "--background", "--python-exit-code", "1",
            "--python", str(SCRIPTS / script), "--"]


def retopo_command(
    source: Path, output: Path, faces: int, atlas: int, angle: float,
    voxel: float, metallic: float, roughness: float, ior: float,
    blender: Path = BLENDER,
) -> list[str]:
    """The headless Blender invocation, as a list.

    `blender_retopo_bake.py` takes positional arguments after `--`, in a fixed order, and
    getting that order wrong silently produces a differently-tuned asset rather than an
    error. Built here so it can be asserted in a test.
    """
    return [
        *headless(blender, "blender_retopo_bake.py"),
        str(source), str(output), str(faces), str(atlas), str(angle),
        str(voxel), str(metallic), str(roughness), str(ior),
    ]


def bake_command(
    source: Path, target: Path, output: Path, size: int, blender: Path = BLENDER,
) -> list[str]:
    """The headless bake-detail invocation: original high-poly onto the current mesh."""
    return [
        *headless(blender, "blender_bake_detail.py"),
        str(source), str(target), str(output), str(size),
    ]


def reuse(path: Path, resume: bool) -> bool:
    """Whether a stage can be skipped because its artifact is already there.

    A zero-byte file is a stage that died mid-write, not one that finished; treating it
    as complete would hand the next stage a truncated GLB instead of re-running the stage
    that actually failed.
    """
    return resume and path.is_file() and path.stat().st_size > 0


def shown(command: list[str]) -> str:
    """The start of a stage's command, for its progress line: through the script it runs,
    which is what tells one Blender stage from another."""
    end = command.index("--python") + 2 if "--python" in command else 4
    return " ".join(command[:end])


def _run(command: list[str], log: Path, label: str) -> None:
    """Run one stage, tee its output to a log, and fail loudly."""
    print(f"[{label}] {shown(command)} ...", flush=True)
    with log.open("w") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise SystemExit(f"[{label}] failed (exit {process.returncode}); last lines:\n{tail}")


def photo_stage(model: Path, views_dir: Path, output: Path, weights_png: Path) -> dict:
    """Paint `model` with the real pixels of the photos in `views_dir`; see
    image_to_3dlab/photo_paint.py. In-process: numpy only, seconds on a Finish mesh."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    from PIL import Image

    from image_to_3dlab import photo_paint as pp

    positions, uvs, faces, texture = pp.read_glb(model)
    views, mesh_scale = pp.load_views(views_dir)
    painted, weight = pp.paint_texture(texture, positions, uvs, faces, views, mesh_scale)
    output.write_bytes(pp.replace_base_colour(model.read_bytes(), pp.encode_png(painted)))
    Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(weights_png)
    share = round(float((weight > 0.5).mean()), 3)
    print(f"[photo] {len(views)} view(s); {share:.0%} of the texture from the photo", flush=True)
    return {"count": len(views), "texture_share": share}


def step_paths(output: Path, steps_dir: Path | None = None) -> dict[str, Path]:
    """Where each in-between file goes.

    Without `steps_dir`, beside the output with a suffix (`out_retopo.glb`), as always.
    With it, inside that folder and numbered by the order the stages run
    (`1_retopo.glb`), so a finished run's folder shows the final model and nothing else
    that looks like one. Numbers are fixed per stage, so a skipped stage leaves a gap
    rather than renumbering what a resume looks for.
    """
    if steps_dir is None:
        stem = str(output.with_suffix(""))
        names = {"retopo": f"{stem}_retopo.glb", "retopo_log": f"{stem}_retopo.log",
                 "painted": f"{stem}_painted.glb", "photo": f"{stem}_photo.glb",
                 "photo_weights": f"{stem}_photo_weights.png",
                 "baked": f"{stem}_baked.glb", "bake_log": f"{stem}_bake.log"}
        return {key: Path(value) for key, value in names.items()}
    return {
        "retopo": steps_dir / "1_retopo.glb", "retopo_log": steps_dir / "1_retopo.log",
        "painted": steps_dir / "2_painted.glb", "photo": steps_dir / "3_photo.glb",
        "photo_weights": steps_dir / "3_photo_weights.png",
        "baked": steps_dir / "4_baked.glb", "bake_log": steps_dir / "4_bake.log",
    }


def preflight(blender: Path | None, skip_paint: bool, platform: str | None = None,
              find=find_blender) -> tuple[Path | None, str | None]:
    """(Blender to use, why the run cannot start). Checked before any work: a missing
    Blender or an off-Mac repaint used to fail minutes in, or with a bare exit code."""
    blender = blender or find()
    if blender is None or not blender.is_file():
        return blender, missing_help()
    if not skip_paint and (platform or host_platform()) != APPLE:
        return blender, REPAINT_OFF_MAC
    return blender, None


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, separately from running it.

    Extracted so a test can assert the flags and their defaults without a Blender run:
    every one of these reaches a stage that costs minutes, and a default that silently
    drifts produces a differently-tuned asset rather than an error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="generated GLB to finish")
    parser.add_argument("image", type=Path, help="source concept art, for the repaint")
    parser.add_argument("output", type=Path, help="finished GLB to write")
    parser.add_argument("--faces", type=int, default=40000, help="retopology target")
    parser.add_argument("--atlas", type=int, default=2048)
    parser.add_argument("--angle", type=float, default=89.0)
    parser.add_argument(
        "--voxel", type=float, default=0.004,
        help="voxel size as a fraction of the asset; 0 skips the remesh, which shatters "
             "thin geometry on everything tested so far",
    )
    parser.add_argument("--metallic", type=float, default=0.25)
    parser.add_argument("--roughness", type=float, default=0.65)
    parser.add_argument("--ior", type=float, default=1.45)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    parser.add_argument("--texture-size", type=int, default=2048,
                        help="cap for colour textures in the compress stage")
    parser.add_argument("--skip-paint", action="store_true",
                        help="stop after retopology, keeping the transferred texture")
    parser.add_argument("--skip-bake", action="store_true",
                        help="skip the normal-map and metallic-roughness bake from the "
                             "original; the asset keeps flat surface settings")
    parser.add_argument("--skip-compress", action="store_true")
    parser.add_argument(
        "--views", type=Path, default=None,
        help="transforms.json directory holding the source photo(s) and their cameras "
             "(a Pixal3D run's .svviews). Given, every surface a photo can see takes the "
             "photo's real pixels after the repaint, which keeps text and logos exact",
    )
    parser.add_argument(
        "--steps-dir", type=Path, default=None,
        help="put the in-between files in this folder, numbered by stage "
             "(1_retopo.glb, 2_painted.glb, ...), instead of beside the output",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse any stage artifact already sitting beside the output instead of "
             "recomputing it. The repaint is five to six minutes of a six-minute run, so "
             "a run that died in compression must not pay for it twice. Only safe with "
             "the settings the intermediates were made with -- change one and start over",
    )
    parser.add_argument("--blender", type=Path, default=None,
                        help="Blender executable; found automatically when omitted")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for path in (args.source, args.image):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    args.blender, problem = preflight(args.blender, args.skip_paint)
    if problem:
        raise SystemExit(problem)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    steps = step_paths(args.output, args.steps_dir)
    if args.steps_dir is not None:
        args.steps_dir.mkdir(parents=True, exist_ok=True)
    if args.views is not None and not (args.views / "transforms.json").is_file():
        raise SystemExit(f"--views has no transforms.json: {args.views}")
    stages = stage_plan(args.skip_paint, args.skip_compress, args.skip_bake,
                        photo=args.views is not None)
    print(f"[retopo-repaint] {args.source.name} -> {args.output.name}: "
          f"{' -> '.join(stages)}", flush=True)

    started = time.time()
    timings: dict[str, float] = {}

    retopo_glb = steps["retopo"]
    step = time.time()
    if reuse(retopo_glb, args.resume):
        emit_stage("retopologise", f"Reusing {retopo_glb.name}")
    else:
        emit_stage("retopologise", f"Retopologising to {args.faces:,} faces")
        _run(
            retopo_command(
                args.source, retopo_glb, args.faces, args.atlas, args.angle,
                args.voxel, args.metallic, args.roughness, args.ior, args.blender,
            ),
            steps["retopo_log"], "retopologise",
        )
    timings["retopologise"] = round(time.time() - step, 1)

    current = retopo_glb
    if not args.skip_paint:
        painted = steps["painted"]
        step = time.time()
        if reuse(painted, args.resume):
            emit_stage("repaint", f"Reusing {painted.name}")
        else:
            # Imported here, not at the top: it pulls in MLX and the paint venv, which a
            # resume that already has the painted GLB has no reason to wait for.
            sys.path.insert(0, str(SCRIPTS))
            from hunyuan_mlx_xiong_generate import run_paint

            emit_stage("repaint", f"Repainting from {args.image.name} at {args.paint_res}px")
            run_paint(
                current, args.image, painted, args.paint_seed, args.paint_res,
                args.paint_steps, args.paint_tex, started,
            )
        timings["repaint"] = round(time.time() - step, 1)
        current = painted

    photo_record = None
    if args.views is not None:
        photo_glb = steps["photo"]
        step = time.time()
        if reuse(photo_glb, args.resume):
            emit_stage("photo", f"Reusing {photo_glb.name}")
        else:
            emit_stage("photo", "Pixel Match: keeping the photo's real pixels where it can see")
            photo_record = photo_stage(current, args.views, photo_glb,
                                       steps["photo_weights"])
        timings["photo"] = round(time.time() - step, 1)
        current = photo_glb

    if not args.skip_bake:
        baked = steps["baked"]
        step = time.time()
        if reuse(baked, args.resume):
            emit_stage("bake", f"Reusing {baked.name}")
        else:
            emit_stage("bake", "Baking normal map and surface from the original")
            _run(bake_command(args.source, current, baked, args.atlas, args.blender),
                 steps["bake_log"], "bake")
        timings["bake"] = round(time.time() - step, 1)
        current = baked

    if not args.skip_compress:
        sys.path.insert(0, str(SCRIPTS))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compress_glb_textures", SCRIPTS / "compress_glb_textures.py"
        )
        compress_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compress_module)

        step = time.time()
        emit_stage("compress", f"Compressing textures to {args.texture_size}px")
        data = current.read_bytes()
        out_bytes, report = compress_module.compress(data, max_size=args.texture_size)
        args.output.write_bytes(out_bytes)
        timings["compress"] = round(time.time() - step, 1)
        print(f"[compress] {len(data) / 1048576:.1f} -> "
              f"{len(out_bytes) / 1048576:.1f} MB", flush=True)
    else:
        args.output.write_bytes(current.read_bytes())
        report = []

    record = {
        "resumed": args.resume,
        "source": str(args.source),
        "image": str(args.image),
        "output": str(args.output),
        "stages": stages,
        "seconds": timings,
        "total_seconds": round(time.time() - started, 1),
        "retopology": {"faces": args.faces, "atlas": args.atlas, "voxel": args.voxel,
                       "angle": args.angle},
        "surface": {"metallic": args.metallic, "roughness": args.roughness, "ior": args.ior},
        "bake": None if args.skip_bake else {"size": args.atlas},
        "paint": None if args.skip_paint else {
            "seed": args.paint_seed, "res": args.paint_res,
            "steps": args.paint_steps, "texture": args.paint_tex,
        },
        "photo": None if args.views is None else {
            "views": str(args.views), **(photo_record or {"reused": True}),
        },
        "textures": report,
        "size_bytes": args.output.stat().st_size,
    }
    record_path = Path(f"{stem}.retopo-repaint.json")
    record_path.write_text(json.dumps(record, indent=2))

    emit_stage("done", f"Finished at {record['size_bytes'] / 1048576:.1f} MB")
    print(f"[retopo-repaint] done in {record['total_seconds']:.0f}s -> {args.output} "
          f"({record['size_bytes'] / 1048576:.1f} MB); record {record_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
