#!/usr/bin/env python3
"""Install TRELLIS.2 for Linux with an NVIDIA card: Microsoft's own code, built for CUDA.

This is the NVIDIA twin of `bootstrap_trellis_space_macos.py`. No port: it clones
`microsoft/TRELLIS.2` at a pinned commit into `vendor/trellis-cuda/`, applies our BRIA
guardrail (`patch_trellis_cuda_no_bria.py`), makes a venv there and installs the CUDA
pieces upstream's `setup.sh` installs. Two routes, picked from the card:

- **RTX 50-series / RTX PRO 6000 (compute capability 12.0):** a pinned set of prebuilt
  CUDA 13 wheels that ran end to end on RunPod (`runpod_trellis2_cuda_requirements.txt`).
  Needs a driver new enough for CUDA 13. Nothing to compile.
- **Any other recent card:** PyTorch for the driver's CUDA, then flash-attn, nvdiffrast,
  nvdiffrec, CuMesh, FlexGEMM and o-voxel compiled for this card. Needs the CUDA toolkit
  (`nvcc`) at the same major version as PyTorch's CUDA. The first compile is long:
  count on 30-60 minutes, more if flash-attn has no prebuilt wheel for this setup.

Then the **weights**, about 15 GB: TRELLIS.2-4B, one decoder from TRELLIS-image-large, and
the DINOv3 image encoder, which is **gated** (request access to
facebook/dinov3-vitl16-pretrain-lvd1689m on Hugging Face, Meta approves by hand, then run
`hf auth login`). Access is checked before anything is built or downloaded.

Linux only. Windows is not supported for TRELLIS.2 yet.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_trellis_cuda.py            # says what it wants, then asks
    python scripts/bootstrap_trellis_cuda.py --yes      # for the viewer and for agents
    python scripts/bootstrap_trellis_cuda.py --code-only
    python scripts/bootstrap_trellis_cuda.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import patch_trellis_cuda_no_bria as bria_patch  # noqa: E402, RUF100

from image_to_3dlab import host, matte  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "trellis-cuda"
CHECKOUT = VENDOR / "TRELLIS.2"
EXTENSIONS = VENDOR / "extensions"
# Written last, so a build that died halfway never looks installed.
BUILT_MARKER = VENDOR / ".i2l-build-complete"
UPSTREAM = "https://github.com/microsoft/TRELLIS.2.git"
LOCK = REPO / "audit" / "trellis-port" / "upstreams.lock.json"
PINNED_REQUIREMENTS = REPO / "scripts" / "runpod_trellis2_cuda_requirements.txt"

# The same utils3d commit upstream's setup.sh installs.
UTILS3D = ("git+https://github.com/EasternJournalist/utils3d.git"
           "@9a4eb15e4021b67b12c460c7057d642626897ec8")
# Upstream's --basic list, minus what only training or the Gradio demo uses, plus rembg
# for our own cut-out (image_to_3dlab/matte.py). transformers is pinned to the version
# that ran DINOv3 on RunPod; older ones do not know the model.
# numpy and plyfile are o-voxel's, which installs with --no-deps (see pip_commands).
BASIC = ["imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
         "ninja", "trimesh", "transformers==4.57.3", "zstandard", "kornia", "timm",
         "pillow", "huggingface_hub", "rembg", "onnxruntime", "numpy", "plyfile"]
# Everything but PyTorch, pinned (scripts/lock_nvidia_routes.py regenerates it).
PACKAGE_LOCK = REPO / "scripts" / "locks" / "trellis-cuda.txt"
FLASH_ATTN = "flash-attn"
# flash-attn publishes Linux wheels for CUDA 12 up to torch 2.8 only. An unpinned torch
# outran them, and pip then compiled flash-attn: one to two hours on an 8-CPU machine.
# Pinning torch to the newest version it has a wheel for makes that step a download.
# CUDA 13 has no flash-attn wheel yet, so that route still compiles it.
TORCH_FOR_FLASH_WHEEL = "2.8.0"
TORCHVISION_FOR_FLASH_WHEEL = "0.23.0"
SOURCE_PYTHON = "3.11"
FLASH_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
               "flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
# (name, git url, tag or commit, recursive), as upstream's setup.sh clones them. The
# commits are what built and ran on an RTX 3090 on 2026-10-01; upstream's own script takes
# whatever HEAD is that day.
SOURCE_EXTENSIONS = (
    ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0", False),
    ("nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git",
     "b296927cc7fd01c2ac1087c8065c4d7248f72da4", False),  # the renderutils branch
    ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git",
     "12289e1062f0603f2f0d0771b02e1395d247f26f", True),
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git",
     "6dd94a859c26ee8246888502eada3dd8ad85532e", True),
)

GATED = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# (repo, allow_patterns or None for the whole repo, approximate gigabytes). Must match the
# TRELLIS entry in viewer/backend_catalog.py; a test holds them together.
WEIGHTS = [
    ("microsoft/TRELLIS.2-4B", None, 15.15),
    ("microsoft/TRELLIS-image-large", ["ckpts/ss_dec_conv3d_16l8_fp16.*"], 0.145),
    (GATED, None, 1.1),
    # The Generate tab's input advisor. Advisory only, but the Setup card counts it, so a
    # setup that skipped it read as unfinished.
    ("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", None, 0.09),
]

LICENCE = (
    "TRELLIS.2: MIT (code + weights). DINOv3: DINOv3 License, gated: request access at\n"
    f"  https://huggingface.co/{GATED} and run `hf auth login`.\n"
    "  BRIA RMBG-2.0, which upstream loads by default, is patched out and never fetched."
)

PINNED_ARCH = "120"


class GatedAccess(Exception):
    """Hugging Face refused a gated repository."""


def pinned_commit(lock: Path = LOCK) -> str:
    return json.loads(lock.read_text())["upstreams"]["microsoft_trellis2"]["commit"]


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def torch_tag_version(tag: str) -> tuple[int, int]:
    """`cu128` -> (12, 8)."""
    digits = tag.removeprefix("cu")
    return int(digits[:-1]), int(digits[-1])


def torch_build_for(driver: tuple[int, int] | None,
                    nvcc: tuple[int, int] | None) -> str | None:
    """The newest PyTorch CUDA build this driver runs *and* this toolkit can compile for.

    torch refuses to build an extension with an nvcc whose major version differs from its
    own CUDA, so a CUDA 13 driver with a 12.8 toolkit must get cu128, not cu130.
    """
    if driver is None or nvcc is None:
        return None
    for minimum, tag in host.TORCH_CUDA_BUILDS:
        if driver >= minimum and nvcc[0] == minimum[0] and nvcc >= minimum:
            return tag
    return None


def arch_list(capability: str | None) -> str | None:
    """`89` -> `8.9`, the spelling TORCH_CUDA_ARCH_LIST wants."""
    if not capability or not capability.isdigit() or len(capability) < 2:
        return None
    return f"{capability[:-1]}.{capability[-1]}"


def choose_route(driver: tuple[int, int] | None, capability: str | None,
                 nvcc: tuple[int, int] | None,
                 machine: str | None = None) -> tuple[str | None, str]:
    """("pinned" | "source" | None, why). None means there is no way to install here."""
    machine = machine or platform.machine()
    if driver is None:
        return None, "no NVIDIA driver answered (nvidia-smi)"
    if capability == PINNED_ARCH and driver >= (13, 0) and machine == "x86_64":
        return "pinned", "prebuilt CUDA 13 wheels for compute capability 12.0"
    if host.torch_cuda_index(driver) is None:
        return None, (f"the driver supports CUDA {driver[0]}.{driver[1]}; PyTorch needs a "
                      "driver for CUDA 12.8 or newer")
    if nvcc is None:
        return None, ("the CUDA toolkit (nvcc) is not installed; TRELLIS.2's extensions "
                      "are compiled for your card and need it")
    tag = torch_build_for(driver, nvcc)
    if tag is None:
        return None, (f"the CUDA toolkit is {nvcc[0]}.{nvcc[1]}, which matches no PyTorch "
                      "CUDA build this driver can run; install CUDA 12.8+ or 13.0+")
    return "source", f"PyTorch {tag}, extensions compiled for this card"


def detect() -> tuple[str | None, str, str | None]:
    """(route, why, torch tag) for this machine."""
    driver = host.driver_cuda_version()
    nvcc = host.nvcc_cuda_version(host.find_nvcc())
    route, why = choose_route(driver, host.compute_capability(), nvcc)
    return route, why, torch_build_for(driver, nvcc) if route == "source" else None


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: TRELLIS.2 (Microsoft)",
             f"  route:   NVIDIA / CUDA on Linux, {route or 'none'}: {why}"]
    if code:
        lines.append(f"  code:    microsoft/TRELLIS.2 @ {pinned_commit()[:7]} -> "
                     "vendor/trellis-cuda/, plus its own venv (several GB of packages)")
        if route == "source":
            lines.append("           CUDA extensions compiled for this card: the first "
                         "compile takes 30-60 minutes, longer if flash-attn has to build")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total -> Hugging Face cache")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {repo}")
        lines.append(f"           {GATED} is gated (see licence below)")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def install_background_remover(target: Path | None = None, download=None) -> None:
    """The cut-out every upload without alpha goes through; without it the first
    generation fetched u2net (176 MB) unannounced."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def has_flash_wheel(torch_tag: str) -> bool:
    return torch_tag_version(torch_tag)[0] == 12


def flash_attn_target(torch_tag: str) -> str:
    """The prebuilt flash-attn wheel where one exists for this CUDA, else the source package."""
    return FLASH_WHEEL if has_flash_wheel(torch_tag) else FLASH_ATTN


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def venv_python(root: Path = VENDOR) -> Path:
    return root / ".venv" / "bin" / "python"


def build_env(capability: str | None, nvcc: str | None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for compiling the CUDA extensions on this machine."""
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


def install_env(route: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the package installs: compile settings on the source route, and
    patient downloads on both, since torch alone is gigabytes."""
    base = dict(os.environ if base is None else base)
    if route == "source":
        base = build_env(host.compute_capability(), host.find_nvcc(), base)
    return host.patient_downloads(base)


def basic_args(lock: Path = PACKAGE_LOCK) -> list[str]:
    """Everything but PyTorch and the compiled extensions, from the lock when it exists."""
    return ["-r", str(lock)] if lock.is_file() else list(BASIC)


def pip_commands(route: str, uv: str, python: Path, torch_tag: str | None = None,
                 checkout: Path = CHECKOUT,
                 extensions: Path = EXTENSIONS) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    pip = [uv, "pip", "install", "--python", str(python)]
    commands = [[*pip, "setuptools", "wheel", "ninja"]]
    if route == "pinned":
        commands.append([*pip, "-r", str(PINNED_REQUIREMENTS)])
        commands.append([*pip, UTILS3D, "rembg", "onnxruntime", "huggingface_hub"])
        return commands
    if route != "source" or torch_tag is None:
        raise ValueError(f"no install plan for route {route!r}")
    torch = ([f"torch=={TORCH_FOR_FLASH_WHEEL}", f"torchvision=={TORCHVISION_FOR_FLASH_WHEEL}"]
             if has_flash_wheel(torch_tag) else ["torch", "torchvision"])
    commands.append([*pip, *torch, "--index-url", host.TORCH_INDEX + torch_tag])
    commands.append([*pip, *basic_args()])
    commands.append([*pip, UTILS3D])
    # --no-build-isolation throughout: the extensions must compile against the torch just
    # installed, not a fresh one pip would fetch into a throwaway build environment.
    if has_flash_wheel(torch_tag):
        commands.append([*pip, flash_attn_target(torch_tag)])
    else:
        commands.append([*pip, "--no-build-isolation", FLASH_ATTN])
    for name, *_ in SOURCE_EXTENSIONS:
        commands.append([*pip, "--no-build-isolation", str(extensions / name)])
    # --no-deps: o-voxel names CuMesh and FlexGEMM as git dependencies, and would fetch
    # and compile both again over the ones just built.
    commands.append([*pip, "--no-build-isolation", "--no-deps", str(checkout / "o-voxel")])
    return commands


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def is_commit(ref: str | None) -> bool:
    return bool(ref) and len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)


def clone(url: str, target: Path, ref: str | None = None, recursive: bool = False) -> None:
    """Clone at a tag or branch (`--branch`) or at an exact commit (checkout after)."""
    if (target / ".git").is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    named = ref if ref and not is_commit(ref) else None
    run(["git", "clone", *(["--recursive"] if recursive else []),
         *(["--branch", named] if named else []), url, str(target)])
    if is_commit(ref):
        run(["git", "checkout", "--detach", ref], cwd=target)
        if recursive:
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=target)


def fetch_checkout(commit: str) -> None:
    clone(UPSTREAM, CHECKOUT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT,
                                   text=True).strip()
    if head != commit:
        run(["git", "fetch", "origin", commit], cwd=CHECKOUT)
        run(["git", "checkout", "--detach", commit], cwd=CHECKOUT)
    # o-voxel vendors Eigen as a submodule.
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=CHECKOUT)


def install_code(route: str, torch_tag: str | None) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout(pinned_commit())
    changed = bria_patch.apply(CHECKOUT)
    print(f"BRIA guardrail: {'applied' if changed else 'already present'}", flush=True)
    python_version = "3.12" if route == "pinned" else SOURCE_PYTHON
    if not venv_python().is_file():
        run([uv, "venv", str(VENDOR / ".venv"), "--python", python_version])
    if route == "source":
        for name, url, ref, recursive in SOURCE_EXTENSIONS:
            clone(url, EXTENSIONS / name, ref, recursive)
    env = install_env(route)
    for command in pip_commands(route, uv, venv_python(), torch_tag):
        run(command, env=env)
    BUILT_MARKER.write_text(json.dumps({"route": route, "torch": torch_tag,
                                        "commit": pinned_commit()}) + "\n")


def check_access(repo: str = GATED) -> None:
    """Raise GatedAccess when Hugging Face will refuse the gated encoder. Seconds, no
    download: better than finding out after a 14 GB fetch and an hour of compiling."""
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.utils import GatedRepoError
    except ImportError:
        print("(huggingface_hub has no auth_check; skipping the access check)")
        return
    try:
        auth_check(repo)
    except GatedRepoError as exc:
        raise GatedAccess(repo) from exc


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. pip install -r requirements.txt"
                         ) from exc
    for repo, patterns, size in WEIGHTS:
        print(f"\nFetching {repo} ({size:.2f} GB)...", flush=True)
        try:
            snapshot_download(repo, allow_patterns=patterns)
        except GatedRepoError as exc:
            raise GatedAccess(repo) from exc


def gated_help(repo: str) -> str:
    return (f"\nHugging Face refused {repo}: it is gated.\n"
            f"  1. Request access at https://huggingface.co/{repo} (Meta approves by hand)\n"
            "  2. Run `hf auth login` with a token from that account\n"
            "  3. Run this again. Nothing was downloaded for it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and CUDA extensions, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    args = parser.parse_args(argv)

    if host.build_target() != "linux-nvidia":
        print("This installs TRELLIS.2 on Linux with an NVIDIA card. Windows is not "
              "supported for TRELLIS.2 yet; on a Mac use "
              "scripts/bootstrap_trellis_space_macos.py. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    route, why, torch_tag = detect()
    if code and route is None:
        print(f"Cannot install TRELLIS.2 here: {why}. Nothing downloaded.")
        return 1
    print(announcement(route, why, code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    try:
        check_access()
        if code:
            install_code(route, torch_tag)
        if weights:
            install_weights()
            install_background_remover()
    except GatedAccess as exc:
        print(gated_help(str(exc)))
        return 1
    print("\nDone. Pick TRELLIS.2 in the studio (Create, Advanced: 3D engine), or run:\n"
          "    vendor/trellis-cuda/.venv/bin/python scripts/trellis_cuda_generate.py "
          "input.png output.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
