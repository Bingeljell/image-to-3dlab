#!/usr/bin/env python3
"""Install Hunyuan3D-2.1 for Linux with an NVIDIA card: Tencent's own code, built for CUDA.

The Mac routes run MLX ports of Hunyuan (`hunyuan_mlx/`). On NVIDIA there is no port:
this clones `Tencent-Hunyuan/Hunyuan3D-2.1` at a pinned commit into
`vendor/hunyuan-cuda/`, makes a Python 3.11 venv there and installs what upstream's
README installs (upstream tests 3.10 with `bpy` 4.0, but bpy 4.0 is gone from PyPI; the
oldest left is 4.2.0, for 3.11 only):

- PyTorch. Upstream tests 2.5.1 + CUDA 12.4, and that is what most cards get. RTX 50-series
  / RTX PRO 6000 cards (compute capability 12.0) are too new for it and get 2.7.1 + CUDA
  12.8 instead; that pairing is untested upstream.
- `requirements.txt`, minus its two extra package mirrors (we install from PyPI only) and
  minus what only the Gradio demo or training uses.
- Two pieces compiled for this card: the paint stage's `custom_rasterizer` (CUDA, needs
  `nvcc` 12.x) and its `mesh_inpaint_processor` (plain C++).

Then the **weights**, about 19.5 GB: the shape model and VAE, the PBR paint model, the
DINOv2-giant image encoder the paint model conditions on, and RealESRGAN for upscaling.
Upstream fetches most of these on first use; here they are named and fetched up front.

Shape needs ~10 GB of GPU memory and paint ~21 GB. The generator unloads shape before
loading paint, so a 24 GB card is enough.

**Licence:** the Hunyuan weights are under the Tencent Hunyuan Community License, which
does **not** cover the EU, the UK or South Korea. Do not install them there.

Linux only. Windows is not supported for Hunyuan3D-2.1 in this lab yet.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_hunyuan_cuda.py            # says what it wants, then asks
    python scripts/bootstrap_hunyuan_cuda.py --yes      # for the viewer and for agents
    python scripts/bootstrap_hunyuan_cuda.py --code-only
    python scripts/bootstrap_hunyuan_cuda.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host, matte  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "hunyuan-cuda"
CHECKOUT = VENDOR / "Hunyuan3D-2.1"
# Where upstream's shape loader looks before it downloads (its HY3DGEN_MODELS). Pointed
# inside vendor/ so 8 GB of shape weights land next to the code, not in a hidden cache.
MODELS = VENDOR / "models"
# Written last, so a build that died halfway never looks installed.
BUILT_MARKER = VENDOR / ".i2l-build-complete"
UPSTREAM = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
# The commit this route was written against (upstream main, 2025-10-17).
COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
PYTHON_VERSION = "3.11"
# Upstream pins bpy==4.0, which PyPI no longer has. The paint stage uses bpy only to turn
# its OBJ into a GLB (wm.obj_import, shade_smooth_by_angle, export_scene.gltf), all in 4.2.
BPY_PIN = "bpy==4.2.0"
# setuptools 81+ dropped pkg_resources, which lightning_fabric (upstream's pytorch-lightning
# 1.9.5) still imports: the paint stage died on start with setuptools 84.
SETUPTOOLS_PIN = "setuptools<81"

# (route, torch, torchvision, index tag, minimum driver CUDA). "tested" is upstream's own
# pairing; "blackwell" exists only because 2.5.1 has no kernels for compute capability 12.0.
# Everything but PyTorch, pinned (scripts/lock_nvidia_routes.py regenerates it).
PACKAGE_LOCK = REPO / "scripts" / "locks" / "hunyuan-cuda.txt"
TORCH = {
    "tested": ("2.5.1", "0.20.1", "cu124", (12, 4)),
    "blackwell": ("2.7.1", "0.22.1", "cu128", (12, 8)),
}
BLACKWELL_ARCH = "120"

# Dropped from upstream's requirements.txt: the Gradio demo's web stack, training-only
# packages, and cupy, which nothing on the inference path imports. open3d stays: no
# Hunyuan file imports it, but paint's remesh reaches it through trimesh.
SKIP_REQUIREMENTS = {"gradio", "fastapi", "uvicorn", "deepspeed", "pythreejs",
                     "cupy-cuda12x"}

REALESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                  "RealESRGAN_x4plus.pth")
# Beside the checkout, not in it (upstream's README uses hy3dpaint/ckpt): --weights-only
# may run before the clone, and git refuses to clone into a folder that already has files.
REALESRGAN = VENDOR / "ckpt" / "RealESRGAN_x4plus.pth"

HUNYUAN_REPO = "tencent/Hunyuan3D-2.1"
DINO_REPO = "facebook/dinov2-giant"
# (label, source, allow_patterns, local_dir or None for the Hugging Face cache, GB).
# Must match the hunyuan-cuda entry in viewer/backend_catalog.py; a test holds them together.
WEIGHTS = [
    ("Hunyuan3D-2.1 shape + VAE", HUNYUAN_REPO,
     ["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"], MODELS / HUNYUAN_REPO, 8.03),
    ("Hunyuan3D-2.1 PBR paint", HUNYUAN_REPO, ["hunyuan3d-paintpbr-v2-1/*"], None, 6.89),
    ("DINOv2-giant image encoder", DINO_REPO,
     ["config.json", "preprocessor_config.json", "model.safetensors"], None, 4.55),
    ("RealESRGAN x4plus upscaler", REALESRGAN_URL, None, REALESRGAN, 0.067),
]

LICENCE = (
    "Hunyuan3D-2.1: Tencent Hunyuan Community License (code + weights). NOT licensed in\n"
    "  the EU, the UK or South Korea: do not install it there.\n"
    "  DINOv2: Apache-2.0. RealESRGAN: BSD-3-Clause."
)


def total_gb() -> float:
    return sum(size for *_, size in WEIGHTS)


def choose_route(driver: tuple[int, int] | None, capability: str | None,
                 nvcc: tuple[int, int] | None) -> tuple[str | None, str]:
    """("tested" | "blackwell" | None, why). None means there is no way to install here."""
    if driver is None:
        return None, "no NVIDIA driver answered (nvidia-smi)"
    route = "blackwell" if capability == BLACKWELL_ARCH else "tested"
    _, _, tag, minimum = TORCH[route]
    if driver < minimum:
        return None, (f"the driver supports CUDA {driver[0]}.{driver[1]}; this card needs "
                      f"{minimum[0]}.{minimum[1]} or newer")
    if nvcc is None:
        return None, ("the CUDA toolkit (nvcc) is not installed; the paint stage's "
                      "rasterizer is compiled for your card and needs it")
    # torch refuses to build an extension with an nvcc of another major version.
    if nvcc[0] != 12 or nvcc < minimum:
        return None, (f"the CUDA toolkit is {nvcc[0]}.{nvcc[1]}; this route needs a 12.x "
                      f"toolkit, {minimum[0]}.{minimum[1]} or newer")
    torch_version = TORCH[route][0]
    note = "" if route == "tested" else " (newer than upstream tests, for this card)"
    return route, f"PyTorch {torch_version} + {tag}{note}"


def detect() -> tuple[str | None, str]:
    driver = host.driver_cuda_version()
    nvcc = host.nvcc_cuda_version(host.find_nvcc())
    return choose_route(driver, host.compute_capability(), nvcc)


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: Hunyuan3D-2.1 (Tencent)",
             f"  route:   NVIDIA / CUDA on Linux, {route or 'none'}: {why}"]
    if code:
        lines.append(f"  code:    Tencent-Hunyuan/Hunyuan3D-2.1 @ {COMMIT[:7]} -> "
                     "vendor/hunyuan-cuda/, plus its own venv (several GB of packages)")
        lines.append("           the paint rasterizer is compiled for this card "
                     "(a few minutes)")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total")
        for label, source, _, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {label} ({source})")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def install_background_remover(target: Path | None = None, download=None) -> None:
    """The cut-out every upload without alpha goes through; without it the first
    generation fetched u2net (176 MB) unannounced."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def filter_requirements(text: str) -> list[str]:
    """Upstream's requirements.txt, as lines we are willing to install.

    Drops index options (upstream adds two extra package mirrors; we install from PyPI
    only), comments, and everything in SKIP_REQUIREMENTS.
    """
    kept = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name in SKIP_REQUIREMENTS:
            continue
        kept.append(BPY_PIN if name == "bpy" else line)
    return kept


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def venv_python(root: Path = VENDOR) -> Path:
    return root / ".venv" / "bin" / "python"


def arch_list(capability: str | None) -> str | None:
    """`89` -> `8.9`, the spelling TORCH_CUDA_ARCH_LIST wants."""
    if not capability or not capability.isdigit() or len(capability) < 2:
        return None
    return f"{capability[:-1]}.{capability[-1]}"


def install_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Compile settings plus patient downloads: torch and its CUDA libraries are
    gigabytes, and uv's default timeout gave up on them over a slow connection."""
    return host.patient_downloads(
        build_env(host.compute_capability(), host.find_nvcc(), base))


def build_env(capability: str | None, nvcc: str | None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for compiling the rasterizer on this machine."""
    env = dict(os.environ if base is None else base)
    arch = arch_list(capability)
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    env["MAX_JOBS"] = str(host.build_jobs())
    if nvcc:
        bin_dir = Path(nvcc).parent
        env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
        env.setdefault("CUDA_HOME", str(bin_dir.parent))
    return env


def requirement_args(lock: Path = PACKAGE_LOCK) -> list[str]:
    """What the main install gets: the lock (scripts/lock_nvidia_routes.py), so a fresh
    install matches the tested one. Upstream's own list is only the fallback."""
    if lock.is_file():
        return ["-r", str(lock)]
    return filter_requirements((CHECKOUT / "requirements.txt").read_text())


def pip_commands(route: str, uv: str, python: Path, requirements: list[str],
                 checkout: Path = CHECKOUT) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    if route not in TORCH:
        raise ValueError(f"no install plan for route {route!r}")
    torch_version, vision_version, tag, _ = TORCH[route]
    pip = [uv, "pip", "install", "--python", str(python)]
    return [
        [*pip, SETUPTOOLS_PIN, "wheel", "ninja", "pybind11"],
        [*pip, f"torch=={torch_version}", f"torchvision=={vision_version}",
         "--index-url", host.TORCH_INDEX + tag],
        # --no-build-isolation: basicsr's setup.py imports torch, and the rasterizer must
        # compile against the torch just installed. rembg is already in the list; our own
        # cut-out (image_to_3dlab/matte.py) uses it.
        [*pip, "--no-build-isolation", SETUPTOOLS_PIN, *requirements],
        [*pip, "--no-build-isolation", str(checkout / "hy3dpaint" / "custom_rasterizer")],
    ]


def mesh_painter_command(includes: str, suffix: str) -> list[str]:
    """upstream's compile_mesh_painter.sh, with our venv's Python answering for pybind11.

    The script shells out to `python3-config`, which inside a venv can name a different
    Python than the one that will import the module.
    """
    return ["c++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC", *includes.split(),
            "mesh_inpaint_processor.cpp", "-o", f"mesh_inpaint_processor{suffix}"]


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def fetch_checkout(commit: str = COMMIT) -> None:
    if not (CHECKOUT / ".git").is_dir():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", UPSTREAM, str(CHECKOUT)])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT,
                                   text=True).strip()
    if head != commit:
        run(["git", "fetch", "origin", commit], cwd=CHECKOUT)
        run(["git", "checkout", "--detach", commit], cwd=CHECKOUT)


def compile_mesh_painter(python: Path) -> None:
    includes = subprocess.check_output([str(python), "-m", "pybind11", "--includes"],
                                       text=True).strip()
    suffix = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        text=True).strip()
    run(mesh_painter_command(includes, suffix),
        cwd=CHECKOUT / "hy3dpaint" / "DifferentiableRenderer")


def venv_needs_rebuild(venv: Path) -> bool:
    """True when there is no venv, or one on another Python: a setup that failed on the
    old bpy 4.0 pin left a 3.10 venv behind, and this route now needs PYTHON_VERSION."""
    try:
        cfg = (venv / "pyvenv.cfg").read_text()
    except OSError:
        return True
    match = re.search(r"^version_info\s*=\s*(\d+\.\d+)", cfg, re.MULTILINE)
    return match is None or match.group(1) != PYTHON_VERSION


def install_code(route: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout()
    if venv_needs_rebuild(VENDOR / ".venv"):
        shutil.rmtree(VENDOR / ".venv", ignore_errors=True)
        run([uv, "venv", str(VENDOR / ".venv"), "--python", PYTHON_VERSION])
    requirements = requirement_args()
    env = install_env()
    for command in pip_commands(route, uv, venv_python(), requirements):
        run(command, env=env)
    compile_mesh_painter(venv_python())
    BUILT_MARKER.write_text(json.dumps({"route": route, "torch": TORCH[route][0],
                                        "commit": COMMIT}) + "\n")


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. pip install -r requirements.txt"
                         ) from exc
    for label, source, patterns, local, size in WEIGHTS:
        print(f"\nFetching {label} ({size:.2f} GB)...", flush=True)
        if patterns is None:  # a plain file URL
            if local.is_file():
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            partial = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(source, partial)
            partial.replace(local)
            continue
        kwargs = {"local_dir": str(local)} if local else {}
        snapshot_download(source, allow_patterns=patterns, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and compile its extensions, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    args = parser.parse_args(argv)

    if host.build_target() != "linux-nvidia":
        print("This installs Hunyuan3D-2.1 on Linux with an NVIDIA card. Windows is not "
              "supported for it yet; on a Mac use the Hunyuan3D-MLX routes. "
              "Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    route, why = detect()
    if code and route is None:
        print(f"Cannot install Hunyuan3D-2.1 here: {why}. Nothing downloaded.")
        return 1
    print(announcement(route, why, code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if code:
        install_code(route)
    if weights:
        install_weights()
        install_background_remover()
    print("\nDone. Pick Hunyuan3D-2.1 (NVIDIA) in the studio (Create, Advanced: 3D engine), or run:\n"
          "    vendor/hunyuan-cuda/.venv/bin/python scripts/hunyuan_cuda_generate.py "
          "input.png output.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
