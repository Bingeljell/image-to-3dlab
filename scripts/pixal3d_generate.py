#!/usr/bin/env python3
"""End-to-end Pixal3D generation: image -> textured GLB, on a Mac, NVIDIA or AMD.

    python scripts/pixal3d_generate.py input.png output.glb [--res 1024] [--seed 42]

Wraps `trellis-cli` from `vendor/pixal3d-cpp` (raven38/pixal3d.cpp), a C++/GGML runtime
with Metal kernels on a Mac, CUDA on NVIDIA and HIP on AMD. Install it with
`scripts/bootstrap_pixal3d.py`.

**Why this port and not the PyTorch one.** `pawel-mazurkiewicz/Pixal3D-mac` loads ~22 GB of
bf16 weights before sampling and its low-VRAM mode moves models between CPU and GPU, which
frees nothing on unified memory. This one runs the same model from 8 GB of Q8_0 weights,
with real Metal flash-attention, and finishes in about 6 minutes where the PyTorch port
could not finish on a 32 GB machine at all.

**Single-view needs a camera.** Pixal3D conditions on pixel-aligned features projected
through an explicit camera, so `--sv-image` synthesizes a front gauge camera at `--fov`
(20 degrees by default). A pre-matted RGBA image goes straight in; anything else is cut out
with u2net first (~5s), because a background left in becomes *geometry* -- a grey studio
backdrop came back as two enormous white sheets either side of a fox's head. `--no-matte`
skips it, `--matte` forces it.

Note that an alpha *channel* is not a matte: Qwen-Image writes RGBA whose alpha is opaque
noise, and both this wrapper and trellis-cli used to read that as "already cut out".

Deliberately not `--pipeline-type 512`: the single-view weight family has no res-512
texture flow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.host import executable
from image_to_3dlab.matte import cut_out, fallback_note, is_matted, matte_model
from image_to_3dlab.glb_turn import turn_cameras, turn_glb
from image_to_3dlab.provenance import LICENSES, sha256_file

PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
CLI = executable(PIXAL3D_ROOT / "build", "trellis-cli")
MODELS = PIXAL3D_ROOT / "models" / "pixal3d-sv"

# The gauge camera the single-view path is designed around: 20 degrees, as radians.
DEFAULT_FOV = 0.3490658503988659

# Structure guidance strength. `trellis-cli` defaults to 7.5, which can drop thin parts
# (a sword blade vanished entirely); 10 keeps them. 13 is worse: thin parts detach.
DEFAULT_GSS = 10.0

# Sampling steps. `trellis-cli` hard-codes 12 for every flow; `--steps` lowers it through
# `PIXAL3D_STEPS`, which only a build patched by `scripts/patch_pixal3d_steps.py` reads.
DEFAULT_STEPS = 12
# What "auto" picks when the build can honour it: same shape and front as 12 on the robot
# and murloc tests (2026-09-28), 16-31% faster; small painted marks on unseen sides soften.
FAST_STEPS = 8
STEPS_ENV = "PIXAL3D_STEPS"
STEPS_MARKER = "i2l_steps"
FLOW_SOURCE = PIXAL3D_ROOT / "src" / "flow_runner.cpp"

STAGES = ["stage", "views", "ss", "shape", "decode", "texture", "write"]
STAGE_LABELS = {
    "views": "Preparing view",
    "ss": "Sparse structure",
    "shape": "Shape SLAT (512 -> 1024 cascade)",
    "decode": "Shape decode",
    "texture": "Texture SLAT + PBR decode",
    "write": "Writing GLB",
}
# `trellis-cli` announces progress as `[n/6] ...`; this maps n to a stage id.
BANNER_STAGES = {1: "views", 2: "ss", 3: "shape", 4: "decode", 5: "texture", 6: "write"}


# The remover that did the cut, recorded in the manifest. Chosen by image_to_3dlab/matte.py:
# BiRefNet-lite when installed, u2net otherwise. Never BRIA RMBG -- a licence guardrail.
MATTE_LICENSES = {
    "birefnet-general-lite": "MIT (BiRefNet)",
    "u2net": "MIT code / Apache-2.0 U-2-Net",
}

# Must match viewer/backend_catalog.py's "pixal3d" entry; a test holds them together.
PROFILE = LICENSES["pixal3d"]
LICENSE_NAME = PROFILE.license_name
LICENSE_URL = PROFILE.license_url


def manifest(image: Path, output: Path, *, res: int, seed: int, fov: float, gss: float,
             gsh: float | None, matted: bool, matted_here: bool,
             seconds: float, steps: int = DEFAULT_STEPS,
             matte_model: str | None = None) -> dict[str, object]:
    """The run's provenance record, written beside the GLB as `<output>.json`.

    Same place and shape as the Hunyuan route's manifest, so the viewer serves it as the
    job's manifest without special-casing, plus the licence and file hashes.
    """
    return {
        "schema_version": 1,
        "backend": "pixal3d",
        "input": {"path": str(image), "sha256": sha256_file(image)},
        "output": {"path": str(output), "sha256": sha256_file(output),
                   "classification": PROFILE.classification},
        "parameters": {"res": res, "seed": seed, "fov": fov, "gss": gss, "gsh": gsh,
                       "matted": matted, "steps": steps},
        "license": {"name": LICENSE_NAME, "url": LICENSE_URL},
        "components": [{
            "component": f"rembg/{matte_model}",
            "purpose": "background removal",
            "license": MATTE_LICENSES.get(matte_model, "see rembg"),
        }] if matted_here and matte_model else [],
        "timings_seconds": {"total": round(seconds, 1)},
    }


def face_front(output: Path) -> None:
    """Turn the GLB trellis-cli wrote to face +Z, glTF's front, and its saved camera with it.

    pixal3d.cpp writes models facing -Z, so every viewer showed the back first. The camera
    in `<output>.svviews/` turns too, or Pixel Match would paint the photo on the back.
    The .ply beside it is left as written: nothing here reads it.
    """
    turned = output.with_name(output.name + ".turning")
    turned.write_bytes(turn_glb(output.read_bytes()))
    turned.replace(output)
    cameras = output.with_suffix(".svviews") / "transforms.json"
    if cameras.is_file():
        cameras.write_text(json.dumps(turn_cameras(json.loads(cameras.read_text())),
                                      indent=2) + "\n")


def has_alpha(image: Path) -> bool:
    """Whether the image carries a matte already -- a real one, not just a fourth channel.

    A pre-matted RGBA image skips BiRefNet entirely, which is both faster and a better
    comparison: the cutout is then identical to whatever else was run on that image.

    **The mode is not the question.** Qwen-Image through `stable-diffusion.cpp` writes RGBA
    whose alpha is noise in the 219-255 range with nothing transparent in it. Trusting the
    mode meant skipping BiRefNet on an image that had never been cut out, so Pixal3D
    reconstructed the backdrop as geometry: the grey studio background of a low-poly fox
    came back as two enormous white sheets either side of its head (2026-09-22).

    So the contents decide. An image counts as matted only when a meaningful share of it is
    actually transparent, which a stray antialiased pixel or a soft vignette will not reach.
    """
    from PIL import Image

    with Image.open(image) as opened:
        return is_matted(opened)


def _rembg_remove(image):
    """Indirection so a test can stand in for a model download. Returns (RGBA, model)."""
    return cut_out(image)


def matte_path(image: Path) -> Path:
    return image.with_name(f"{image.stem}__matted.png")


def matte(image: Path, destination: Path | None = None) -> tuple[Path, str]:
    """Cut the subject out ourselves and write an RGBA beside the source.

    Returns the cutout's path and the remover that made it (see image_to_3dlab/matte.py).

    We do this rather than passing `--bg-removal birefnet` because that was measured and
    does nothing: trellis-cli decides "already matted" from the alpha channel's presence,
    the same mistake `has_alpha` used to make, so a Qwen image with its junk alpha sails
    straight through uncut. Handing over a real cutout removes the guess entirely.

    **Never BRIA RMBG**, which this repo's generation pipeline must not load.
    """
    from PIL import Image

    destination = destination or matte_path(image)
    with Image.open(image) as opened:
        cut, model = _rembg_remove(opened.convert("RGB"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cut.save(destination)
    return destination, model


def build_command(
    image: Path, output: Path, res: int, seed: int, fov: float,
    models: Path = MODELS, cli: Path = CLI, matted: bool = True,
    gss: float = DEFAULT_GSS, gsh: float | None = None,
) -> list[str]:
    """The `trellis-cli` invocation.

    Everything takes `--sv-image`, which crops to the alpha bounding box the way the
    reference preprocess does and synthesizes the gauge camera. An image that is not
    already cut out additionally asks the CLI for BiRefNet matting.

    `--gss` is always passed rather than left to the CLI default, because that default
    (7.5) is the setting that dropped the warrior girl's sword blade.
    """
    # Always `--sv-image`, matted or not: `--pixal3d-weights` is refused without it
    # ("--pixal3d-weights requires --views DIR or --sv-image PATH"), and those weights are
    # the entire reason to use this backend. An unmatted image is handed to BiRefNet by the
    # CLI instead of being pre-cut by us.
    head = [str(cli), "--sv-image", str(image)]
    if not matted:
        head += ["--bg-removal", "birefnet"]
    command = head + [
        "--fov", str(fov),
        "--models", str(models),
        "--seed", str(seed),
        "--res", str(res),
        "--pixal3d-weights", "sv",
        "--gss", str(gss),
    ]
    if gsh is not None:
        command += ["--gsh", str(gsh)]
    return command + [str(output)]


def steps_problem(steps: int | None, source: Path = FLOW_SOURCE,
                  cli: Path = CLI) -> str | None:
    """Why `--steps` cannot be honoured, or None when it can.

    An unpatched build ignores `PIXAL3D_STEPS` and quietly runs 12 steps, which would give
    a 27-minute run a label that is a lie. Checked before starting, in seconds.
    """
    if steps is None or steps == DEFAULT_STEPS:
        return None
    if not 1 <= steps <= 50:
        return "--steps must be 1 to 50"
    if not source.is_file() or STEPS_MARKER not in source.read_text():
        return ("trellis-cli is not patched for --steps: run "
                "scripts/patch_pixal3d_steps.py, then rebuild trellis-cli")
    if not cli.is_file() or cli.stat().st_mtime < source.stat().st_mtime:
        return ("trellis-cli is older than the patched flow_runner.cpp: rebuild it with "
                "cmake --build vendor/pixal3d-cpp/build --target trellis-cli")
    return None


def resolve_steps(requested: int | None, source: Path = FLOW_SOURCE,
                  cli: Path = CLI) -> tuple[int, str]:
    """The step count a run will use, and a log line saying why.

    Unset means auto: FAST_STEPS on a patched, rebuilt trellis-cli, otherwise the stock 12
    -- never a refusal, because every install that predates the patch lands here. An
    explicit `--steps` the build cannot honour is refused instead (see steps_problem).
    """
    if requested is None:
        if steps_problem(FAST_STEPS, source, cli) is None:
            return FAST_STEPS, f"steps={FAST_STEPS} (auto)"
        return DEFAULT_STEPS, (f"steps={DEFAULT_STEPS} (auto: this trellis-cli has no steps "
                               "patch; run scripts/patch_pixal3d_steps.py and rebuild for "
                               f"{FAST_STEPS})")
    problem = steps_problem(requested, source, cli)
    if problem:
        raise SystemExit(problem)
    return requested, f"steps={requested}"


def rocm_library_path(base: dict[str, str], root: Path | None = None) -> str | None:
    """ROCm's library directory, to put on the loader's path when it is not already there.

    A HIP `trellis-cli` links against ROCm's own `libamdhip64.so` and friends. CMake records
    that path in the binary at build time on most systems, so this is only the safety net for
    one where it did not — without it the process dies at load time with a message about a
    missing library rather than anything about the GPU.
    """
    found = root or _rocm_root()
    if found is None:
        return None
    directory = found / "lib"
    if not directory.is_dir():
        return None
    return None if str(directory) in base.get("LD_LIBRARY_PATH", "").split(":") else directory


def _rocm_root() -> Path | None:
    from image_to_3dlab import host

    return host.rocm_root()


def run_env(steps: int | None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The CLI's environment. `PIXAL3D_STEPS` is set only when asked for, and a stray one
    in the caller's shell is dropped, so the manifest's step count is always the truth."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k != STEPS_ENV}
    if steps is not None and steps != DEFAULT_STEPS:
        env[STEPS_ENV] = str(steps)
    rocm = rocm_library_path(env)
    if rocm is not None:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{rocm}:{existing}" if existing else str(rocm)
    return env


def steps_not_applied(line: str, steps: int | None, seen_override: bool) -> bool:
    """True when a flow has started sampling without announcing the override: the build
    ignored `PIXAL3D_STEPS`, and the run should stop now rather than 20 minutes later."""
    if steps is None or steps == DEFAULT_STEPS or seen_override:
        return False
    return "[flow] [" in line


def kernel_noise(line: str) -> bool:
    """True for ggml's own kernel chatter, on every backend it has.

    Metal names its pipelines `ggml_metal_*`, HIP its kernels `ggml_hip_*`, and CUDA says
    `ggml_cuda_init` and friends. A 6-minute run emits hundreds of these lines, and they
    bury the six `[n/6]` stage banners the progress panel reads, so they are dropped rather
    than printed. Anything else, including a line that merely mentions a kernel, is kept.
    """
    return line.startswith(("ggml_metal", "ggml_hip", "ggml_cuda")) or "loaded kernel" in line


def stage_from_banner(line: str) -> tuple[str, int] | None:
    """Turn a `[n/6] ...` banner into (stage_id, percent), or None.

    Percent is the banner index rather than anything measured: the stages are wildly
    uneven (shape is ~2 minutes, decode ~13 seconds), but a monotonic bar beats none.
    """
    if not line.startswith("[") or "]" not in line:
        return None
    marker = line[1:line.index("]")]
    if "/6" not in marker:
        return None
    try:
        index = int(marker.split("/")[0])
    except ValueError:
        return None
    stage = BANNER_STAGES.get(index)
    if stage is None:
        return None
    return stage, min(99, round(index / 6 * 100))


def readiness(cli: Path = CLI, models: Path = MODELS) -> dict[str, object]:
    """What is missing before a run can start, for the viewer's setup panel."""
    weights = sorted(models.glob("*.gguf")) if models.is_dir() else []
    return {
        "cli_built": cli.is_file(),
        "cli_path": str(cli),
        "models_dir": str(models),
        "weights_present": len(weights),
        "ready": cli.is_file() and len(weights) >= 9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--res", type=int, choices=(1024,), default=1024)  # --sv-image is 1024-only
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fov", type=float, default=DEFAULT_FOV,
        help="gauge camera FOV in radians for the single-view rig; 0.349 is 20 degrees",
    )
    parser.add_argument(
        "--gss", type=float, default=DEFAULT_GSS,
        help="structure guidance strength; 10 recovers thin props the CLI default of 7.5 drops",
    )
    parser.add_argument(
        "--gsh", type=float, default=None,
        help="shape guidance strength; left to the runtime default when unset",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help=f"sampling steps per flow. Default: {FAST_STEPS} when trellis-cli is patched "
             f"(scripts/patch_pixal3d_steps.py + rebuild), else {DEFAULT_STEPS}. An explicit "
             "value the build cannot honour is refused",
    )
    parser.add_argument(
        "--matte", dest="matte", action="store_true", default=None,
        help="force background removal even if the image looks cut out already",
    )
    parser.add_argument(
        "--no-matte", dest="matte", action="store_false",
        help="never matte; use the image exactly as given (u2net can eat thin structures)",
    )
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--cli", type=Path, default=CLI)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"not found: {args.image}")
    state = readiness(args.cli, args.models)
    if not state["ready"]:
        raise SystemExit(
            f"pixal3d.cpp is not ready: {state}. Run scripts/bootstrap_pixal3d.py"
        )

    steps, steps_note = resolve_steps(args.steps, cli=args.cli)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # `trellis-cli` is launched from its own tree so it can find its Metal library, which
    # means a relative input or output path would resolve against *that* directory and the
    # run dies at once with "can't fopen". Absolute paths are the only safe thing to pass.
    # Resolved *before* matting, so the cutout is written beside the real source and the
    # path handed to the CLI is absolute whichever branch produced it.
    args.image = args.image.resolve()
    args.output = args.output.resolve()

    image = args.image
    matted = has_alpha(image)
    matted_here = False
    used_matte = None
    # `--matte` / `--no-matte` override the detection; by default it decides. Left to
    # itself, an image that is already cut out keeps its own matte, and anything else --
    # every picture the Generate Image tab makes included -- gets one.
    if args.matte is False:
        print("[pixal3d] --no-matte: using the image exactly as given", flush=True)
    elif args.matte or not matted:
        print(f"[pixal3d] matting with {matte_model()}", flush=True)
        image, used_matte = matte(image)
        if fallback_note(used_matte):
            print(f"[pixal3d] note: {fallback_note(used_matte)}", flush=True)
        matted = True
        matted_here = True
        print(f"[pixal3d] matted image: {image}", flush=True)

    started = time.time()
    command = build_command(
        image, args.output, args.res, args.seed, args.fov,
        args.models, args.cli, matted, args.gss, args.gsh,
    )
    print(f"[pixal3d] res={args.res} seed={args.seed} gss={args.gss} matted={matted}",
          flush=True)
    print(f"[pixal3d] {steps_note}", flush=True)

    process = subprocess.Popen(
        command, cwd=str(PIXAL3D_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env(steps),
    )
    assert process.stdout is not None
    seen_override = False
    for raw in process.stdout:
        line = raw.rstrip("\n")
        seen_override = seen_override or f"{STEPS_ENV}=" in line
        if steps_not_applied(line, steps, seen_override):
            process.kill()
            raise SystemExit(f"trellis-cli ignored {STEPS_ENV}; stopped before wasting "
                             "the run. Re-apply scripts/patch_pixal3d_steps.py and rebuild")
        if not kernel_noise(line):
            print(line, flush=True)
    code = process.wait()
    if code != 0:
        raise SystemExit(f"trellis-cli exited with code {code}")
    if not args.output.is_file():
        raise SystemExit(f"trellis-cli exited 0 without writing {args.output}")
    face_front(args.output)

    seconds = time.time() - started
    record = manifest(args.image, args.output, res=args.res, seed=args.seed, fov=args.fov,
                      gss=args.gss, gsh=args.gsh, matted=matted, matted_here=matted_here,
                      matte_model=used_matte,
                      seconds=seconds, steps=steps)
    record_path = args.output.with_name(f"{args.output.stem}.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    size = args.output.stat().st_size / 1048576
    print(f"[pixal3d] done in {seconds:.0f}s -> {args.output} "
          f"({size:.1f} MB); manifest {record_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
