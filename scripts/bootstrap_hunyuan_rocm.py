#!/usr/bin/env python3
"""Install Hunyuan3D-2.1 for Linux with an AMD card: Tencent's own code, built with HIP.

This is the AMD twin of `bootstrap_hunyuan_cuda.py`. No port: it clones
`Tencent-Hunyuan/Hunyuan3D-2.1` at the same pinned commit into `vendor/hunyuan-rocm/`,
applies `patch_hunyuan_rocm.py`, makes a Python 3.11 venv there and installs what
upstream's README installs.

**What had to change, which is less than you would expect.** The shape stage is plain
PyTorch with no compiled code at all. The paint stage has exactly one CUDA extension, and
ROCm's torch compiles it with `hipcc` on its own -- its `setup.py` needs no edit, because
`CUDAExtension` dispatches to hipcc when `torch.version.hip` is set. So there is no hipify
step and no source port. What the patches fix is three Python-level things: a scheduler
that crashes on a float comparison, attention that outgrows a 20 GB card, and a hardcoded
device.

PyTorch itself is not installed from PyPI. PyPI's Linux torch is a CUDA build, so on an
AMD card it reports no device at all; this picks the ROCm build that matches the installed
runtime, from the table below, which is read off PyTorch's own index rather than guessed.

Shape needs about 10 GB of GPU memory. Paint is the one that does not fit a 20 GB card at
the default settings: set `HY3D_SDPA_QUERY_CHUNK` (the generate script does, and lowers
`resolution` and `max_num_view` first) and it does.

Then the **weights**, about 19.5 GB: the shape model and VAE, the PBR paint model, the
DINOv2-giant image encoder the paint model conditions on, and RealESRGAN for upscaling.

**Licence:** the Hunyuan weights are under the Tencent Hunyuan Community License, which
does **not** cover the EU, the UK or South Korea. Do not install them there.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_hunyuan_rocm.py            # says what it wants, then asks
    python scripts/bootstrap_hunyuan_rocm.py --yes      # for the viewer and for agents
    python scripts/bootstrap_hunyuan_rocm.py --code-only
    python scripts/bootstrap_hunyuan_rocm.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_hunyuan_cuda as cuda
import patch_hunyuan_rocm as hunyuan_patch

from image_to_3dlab import host, matte

VENDOR = REPO / "vendor" / "hunyuan-rocm"
CHECKOUT = VENDOR / "Hunyuan3D-2.1"
# Where upstream's shape loader looks before it downloads (its HY3DGEN_MODELS). Pointed
# inside vendor/ so 8 GB of shape weights land next to the code, not in a hidden cache.
MODELS = VENDOR / "models"
# Written last, so a build that died halfway never looks installed.
BUILT_MARKER = VENDOR / ".i2l-build-complete"

# Same upstream commit as the NVIDIA route, so the two installs differ only in how they
# build, not in what they build.
UPSTREAM = cuda.UPSTREAM
COMMIT = cuda.COMMIT
PYTHON_VERSION = cuda.PYTHON_VERSION
BPY_PIN = cuda.BPY_PIN
SETUPTOOLS_PIN = cuda.SETUPTOOLS_PIN
LICENCE = cuda.LICENCE

# (ROCm index tag, torch, torchvision, minimum ROCm). Read off PyTorch's own index rather
# than guessed: each tag publishes its own builds and a mismatched pair fails at the first
# kernel rather than at load time. `host.TORCH_ROCM_BUILDS` says which tag a runtime gets.
TORCH = {
    "rocm7.2": ("2.14.1", "0.29.1", (7, 2)),
    "rocm7.1": ("2.13.0", "0.28.0", (7, 1)),
    "rocm7.0": ("2.10.0", "0.25.0", (7, 0)),
    "rocm6.4": ("2.9.1", "0.24.1", (6, 4)),
    "rocm6.3": ("2.9.1", "0.24.1", (6, 3)),
    "rocm6.2": ("2.5.1", "0.20.1", (6, 2)),
}

# cupy-cuda12x is already in the shared skip list: nothing on the inference path imports
# it, and it is a CUDA wheel.
SKIP_REQUIREMENTS = cuda.SKIP_REQUIREMENTS

# Upstream pins onnxruntime==1.16.3, whose shared object asks the loader for an executable
# stack. A kernel that refuses one answers "cannot enable executable stack as shared object
# requires" — and it did, on the first import diffusers made of its lazy module table, taking
# the paint stage down on an import nothing in Hunyuan had asked for.
#
# It was "fixed" here by skipping the package. That was the wrong remedy: the module that
# actually imports it is real. `hy3dshape.rembg` does, `image_to_3dlab/matte.py` does, and
# rembg imports onnxruntime at module import — so a machine that took the skip would have
# swapped an ImportError for a ModuleNotFoundError the first time it cut out a background.
# So the pin is replaced instead, with the same floor requirements.txt already asks for. The
# current wheel is an ordinary shared object and loads.
REPLACED = {"onnxruntime": "onnxruntime>=1.18"}

HUNYUAN_REPO = cuda.HUNYUAN_REPO
DINO_REPO = cuda.DINO_REPO
REALESGAN_URL = cuda.REALESRGAN_URL
REALESRGAN = VENDOR / "ckpt" / "RealESRGAN_x4plus.pth"

# (label, source, allow_patterns, local_dir or None for the Hugging Face cache, GB).
# Must match the hunyuan-rocm entry in viewer/backend_catalog.py; a test holds them together.
WEIGHTS = [
    ("Hunyuan3D-2.1 shape + VAE", HUNYUAN_REPO,
     ["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"], MODELS / HUNYUAN_REPO, 8.03),
    ("Hunyuan3D-2.1 PBR paint", HUNYUAN_REPO, ["hunyuan3d-paintpbr-v2-1/*"], None, 6.89),
    ("DINOv2-giant image encoder", DINO_REPO,
     ["config.json", "preprocessor_config.json", "model.safetensors"], None, 4.55),
    ("RealESRGAN x4plus upscaler", REALESGAN_URL, None, REALESRGAN, 0.067),
]


def total_gb() -> float:
    return sum(size for *_, size in WEIGHTS)


def choose_route(version: tuple[int, int] | None, hipcc: str | None,
                 gfx: list[str]) -> tuple[str | None, str]:
    """("rocm7.2" | ..., why). None means there is no way to install here."""
    if version is None:
        return None, ("no ROCm install answered (no /opt/rocm version and no `hipconfig`); "
                      "install ROCm first, https://rocm.docs.amd.com")
    if hipcc is None:
        return None, ("the HIP compiler (hipcc) is not installed; the paint stage's "
                      "rasterizer is compiled for your card and needs it")
    tag = host.torch_rocm_index(version)
    if tag is None:
        return None, (f"ROCm {version[0]}.{version[1]} is older than PyTorch's ROCm builds "
                      f"(they need {TORCH['rocm6.2'][2][0]}.{TORCH['rocm6.2'][2][1]}+)")
    # `torch_rocm_index` names a tag from host.TORCH_ROCM_BUILDS; TORCH above is this
    # route's copy of that table, so a tag in one and not the other means someone added a
    # ROCm build without picking a torch for it.
    name = tag.rsplit("/", 1)[-1]
    if name not in TORCH:
        return None, (f"ROCm {version[0]}.{version[1]} maps to PyTorch's {name} index, "
                      f"which this route has no tested torch for")
    if not gfx:
        return None, ("rocminfo did not report a gfx target, so there is nothing to "
                      "compile the rasterizer for")
    return name, f"PyTorch {TORCH[name][0]} + {tag}, kernels for {', '.join(gfx)}"


def detect() -> tuple[str | None, str]:
    return choose_route(host.rocm_version(), host.find_hipcc(), host.rocm_gfx_targets())


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: Hunyuan3D-2.1 (Tencent)",
             f"  route:   AMD / ROCm on Linux, {route or 'none'}: {why}"]
    if code:
        lines.append(f"  code:    Tencent-Hunyuan/Hunyuan3D-2.1 @ {COMMIT[:7]} -> "
                     "vendor/hunyuan-rocm/, plus its own venv (several GB of packages)")
        lines.append("           the paint rasterizer is compiled with hipcc for your card "
                     "(a few minutes)")
        lines.append("           the shape stage is plain PyTorch and needs no compiling")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total")
        for label, source, _, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {label} ({source})")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def arch_list(gfx: list[str]) -> str | None:
    """`PYTORCH_ROCM_ARCH` as a comma-separated list of this machine's gfx targets.

    `rocm_gfx_targets` already leaves out an integrated GPU when a discrete card is
    present, because compiling for both would double the build for nothing.
    """
    return ",".join(gfx) if gfx else None


def build_env(gfx: list[str], hipcc: str | None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for compiling the rasterizer with hipcc on this machine."""
    env = dict(os.environ if base is None else base)
    arch = arch_list(gfx)
    if arch:
        env["PYTORCH_ROCM_ARCH"] = arch
    env["MAX_JOBS"] = str(host.build_jobs())
    if hipcc:
        root = host.rocm_root(hipcc)
        if root:
            env["ROCM_HOME"] = str(root)
        env["PATH"] = os.pathsep.join([str(Path(hipcc).parent), env.get("PATH", "")])
    return env


def install_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Compile settings plus patient downloads: torch's ROCm wheel is gigabytes."""
    return host.patient_downloads(build_env(host.rocm_gfx_targets(), host.find_hipcc(), base))


def filter_requirements(text: str) -> list[str]:
    """Upstream's list with *this route's* skips and replacements applied.

    Handing ours over is the whole point: a wrapper that just called the shared filter let it
    read the CUDA module's globals, so this route's extra skip and its extra replacement were
    both ignored and every upstream pin went in exactly as written.
    """
    return cuda.filter_requirements(text, SKIP_REQUIREMENTS, REPLACED)


def requirement_args(checkout: Path = CHECKOUT) -> list[str]:
    """Upstream's own list: the NVIDIA route has a generated lock, and every pin in it was
    resolved against a CUDA torch, so reusing it here would drag those pins in."""
    return filter_requirements((checkout / "requirements.txt").read_text())


def pip_commands(route: str, uv: str, python: Path, requirements: list[str],
                 checkout: Path = CHECKOUT) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    if route not in TORCH:
        raise ValueError(f"no install plan for route {route!r}")
    torch_version, vision_version, _ = TORCH[route]
    pip = [uv, "pip", "install", "--python", str(python)]
    return [
        [*pip, SETUPTOOLS_PIN, "wheel", "ninja", "pybind11"],
        # --index-url, not --extra-index-url: the ROCm build has to be the one that wins,
        # and PyPI's Linux torch is the CUDA build that reports no device on an AMD card.
        [*pip, f"torch=={torch_version}", f"torchvision=={vision_version}",
         "--index-url", host.TORCH_INDEX + route],
        # --no-build-isolation: the rasterizer must compile against the torch just
        # installed, and basicsr's setup.py imports torch.
        [*pip, "--no-build-isolation", SETUPTOOLS_PIN, *requirements],
        [*pip, "--no-build-isolation", str(checkout / "hy3dpaint" / "custom_rasterizer")],
    ]


def run(command: list[str], **kwargs) -> None:
    cuda.run(command, **kwargs)


def fetch_checkout(commit: str = COMMIT) -> None:
    if not (CHECKOUT / ".git").is_dir():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", UPSTREAM, str(CHECKOUT)])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT,
                                   text=True).strip()
    if head != commit:
        run(["git", "fetch", "origin", commit], cwd=CHECKOUT)
        run(["git", "checkout", "--detach", commit], cwd=CHECKOUT)


def compile_mesh_painter(python: Path, checkout: Path = CHECKOUT) -> None:
    """upstream's compile_mesh_painter.sh, with our venv's Python answering for pybind11.

    The script shells out to `python3-config`, which inside a venv can name a different
    Python than the one that will import the module.
    """
    includes = subprocess.check_output([str(python), "-m", "pybind11", "--includes"],
                                       text=True).strip()
    suffix = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        text=True).strip()
    run(cuda.mesh_painter_command(includes, suffix),
        cwd=checkout / "hy3dpaint" / "DifferentiableRenderer")


def venv_python() -> Path:
    return VENDOR / ".venv" / "bin" / "python"


def venv_needs_rebuild(venv: Path) -> bool:
    return cuda.venv_needs_rebuild(venv)


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def install_code(route: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout()
    hunyuan_patch.patch(CHECKOUT)
    if venv_needs_rebuild(VENDOR / ".venv"):
        shutil.rmtree(VENDOR / ".venv", ignore_errors=True)
        run([uv, "venv", str(VENDOR / ".venv"), "--python", PYTHON_VERSION])
    env = install_env()
    for command in pip_commands(route, uv, venv_python(), requirement_args()):
        run(command, env=env)
    compile_mesh_painter(venv_python())
    BUILT_MARKER.write_text(json.dumps({"route": route, "torch": TORCH[route][0],
                                        "commit": COMMIT}) + "\n")


def install_weights() -> None:
    """Fetch every weight set, into vendor/hunyuan-rocm rather than the NVIDIA route's
    folder, so a machine with both cards never overwrites the other's."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. pip install -r requirements.txt"
                         ) from exc
    import urllib.request

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

    if host.build_target() != "linux-amd":
        print("This installs Hunyuan3D-2.1 on Linux with an AMD card and ROCm. On NVIDIA "
              "use bootstrap_hunyuan_cuda.py; on a Mac use the Hunyuan3D-MLX routes. "
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
        cuda.install_background_remover()
    print("\nDone. Pick Hunyuan3D-2.1 (AMD) in the studio (Create, Advanced: 3D engine), "
          "or run:\n    vendor/hunyuan-rocm/.venv/bin/python scripts/hunyuan_rocm_generate.py "
          "input.png output.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())