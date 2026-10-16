#!/usr/bin/env python3
"""Install TRELLIS.2 for Linux with an AMD card: Microsoft's own code, built with HIP.

The AMD twin of `bootstrap_trellis_cuda.py`. It runs the same pinned
`microsoft/TRELLIS.2` checkout, the same pipeline and the same weights; only the four
compiled pieces differ, and three of those need no port at all.

- **CuMesh, FlexGEMM and o-voxel are already HIP-capable upstream.** Their `setup.py` reads
  `IS_HIP_EXTENSION` from torch and passes `--offload-arch`, which is all a ROCm torch
  needs. The only thing missing is `-std=c++20`, because torch 2.14's headers require it
  and those commits say C++17. So this route builds the *same pinned commits* as the NVIDIA
  route, with a three-word patch, and both cards run identical source. Verified by building
  them from upstream on an RX 7900 XT, not assumed.
- **nvdiffrast is the exception.** Upstream is CUDA-only with no HIP path to patch into, so
  this takes `dmonkman/nvdiffrast-ROCm`, and applies `patch_trellis_rocm.py` to fix the
  device-bitcode path it hardcodes for ROCm 10.
- **Attention needs a backend that has no CUDA extension at all.** TRELLIS.2's three are
  xformers, flash-attn and flash-attn-3, all compiled against CUDA. This installs
  AuleAttention, a Triton flash attention that needs no compilation, and the same patch
  adds it as a fourth backend.

The BRIA guardrail (`patch_trellis_cuda_no_bria.py`) applies unchanged: upstream loads
BRIA RMBG-2.0 by default and this repo must never load it.

Then the **weights**, about 16 GB: TRELLIS.2-4B, one decoder from TRELLIS-image-large, and
the DINOv3 image encoder, which is **gated** (request access to
facebook/dinov3-vitl16-pretrain-lvd1689m on Hugging Face, Meta approves by hand, then run
`hf auth login`). Access is checked before anything is built or downloaded.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_trellis_rocm.py            # says what it wants, then asks
    python scripts/bootstrap_trellis_rocm.py --yes      # for the viewer and for agents
    python scripts/bootstrap_trellis_rocm.py --code-only
    python scripts/bootstrap_trellis_rocm.py --weights-only
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

import bootstrap_trellis_cuda as cuda  # noqa: E402, RUF100
import patch_trellis_cuda_no_bria as bria_patch  # noqa: E402, RUF100
import patch_trellis_rocm as rocm_patch  # noqa: E402, RUF100

from image_to_3dlab import host, matte  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "trellis-rocm"
CHECKOUT = VENDOR / "TRELLIS.2"
EXTENSIONS = VENDOR / "extensions"
# Written last, so a build that died halfway never looks installed.
BUILT_MARKER = VENDOR / ".i2l-build-complete"

UPSTREAM = cuda.UPSTREAM
LOCK = cuda.LOCK
SOURCE_PYTHON = cuda.SOURCE_PYTHON
BASIC = cuda.BASIC
GATED = cuda.GATED
WEIGHTS = cuda.WEIGHTS

UTILS3D = cuda.UTILS3D
PACKAGE_LOCK = cuda.PACKAGE_LOCK

# The one extension upstream cannot build for HIP, and the one community port of it.
# dmonkman/nvdiffrast-ROCm is NVIDIA's v0.4.0 plus a HIP path and the device-bitcode fix
# scripts/patch_trellis_rocm.py still has to make.
NVDIFFRAST = ("nvdiffrast", "https://github.com/dmonkman/nvdiffrast-ROCm.git",
              "dd6f84d115d6c4c2a31d5cbcb7b225d5f7a46c5a", False)
# These two are NVIDIA's own, at the same commits the CUDA route uses. Upstream already reads
# IS_HIP_EXTENSION and passes --offload-arch, so they need no fork: the only change this
# route makes is the C++20 bump the ROCm forks all make too. Checked, not assumed — built
# from upstream at these commits on an RX 7900 XT (gfx1100, ROCm 7.2, torch 2.14.1).
SOURCE_EXTENSIONS = (
    NVDIFFRAST,
    ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git",
     "12289e1062f0603f2f0d0771b02e1395d247f26f", True),
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git",
     "6dd94a859c26ee8246888502eada3dd8ad85532e", True),
)

# Triton flash attention with no compiled extension, so it is the one backend that also
# runs on ROCm. See scripts/patch_trellis_rocm.py for the dispatch that selects it.
AULE_ATTENTION = "aule-attention"

# Per ROCm tag, read off PyTorch's own index. The third element is unused here: this route
# has no prebuilt flash-attn wheel to match the way the CUDA route pins torch for one.
TORCH = {
    (7, 2): ("2.14.1", "0.29.1"),
    (7, 1): ("2.13.0", "0.28.0"),
    (7, 0): ("2.10.0", "0.25.0"),
    (6, 4): ("2.9.1", "0.24.1"),
    (6, 3): ("2.9.1", "0.24.1"),
    (6, 2): ("2.5.1", "0.20.1"),
}

# These forks split GPU_ARCHS on a semicolon, not the comma PYTORCH_ROCM_ARCH uses. One
# arch on this machine, and a mistake here compiles for the wrong card.
ARCH_SEPARATOR = ";"

# Upstream's --basic list minus what only training or the Gradio demo uses. onnxruntime is
# dropped: its 1.16.3 wheel asks for an executable stack the kernel here refuses, and nothing
# on the generation path needs it. transformers is pinned to the version that knows
# DINOv3. numpy and plyfile are o-voxel's, which installs with --no-deps.
BASIC_ROCM = [p for p in BASIC if p != "onnxruntime"]

LICENCE = (
    "TRELLIS.2: MIT (code + weights). DINOv3: DINOv3 License, gated: request access at\n"
    f"  https://huggingface.co/{GATED} and run `hf auth login`.\n"
    "  nvdiffrast: NVIDIA Source Code License, from the dmonkman ROCm port.\n"
    "  BRIA RMBG-2.0, which upstream loads by default, is patched out and never fetched."
)


class GatedAccess(Exception):
    """Hugging Face refused a gated repository."""


def pinned_commit(lock: Path = LOCK) -> str:
    return json.loads(lock.read_text())["upstreams"]["microsoft_trellis2"]["commit"]


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def choose_route(version: tuple[int, int] | None, hipcc: str | None,
                 gfx: list[str]) -> tuple[str | None, str]:
    """("source" | None, why). None means there is no way to install here."""
    if not gfx:
        return None, ("no AMD GPU target found. Needs rocminfo, or a readable "
                      "/sys/class/kfd/kfd/topology; check the driver")
    if hipcc is None:
        return None, ("hipcc was not found. These extensions are compiled for your card and "
                      "need the ROCm toolkit")
    if version is None:
        return None, ("the ROCm version could not be read. PyTorch ROCm builds are published "
                      "per ROCm release, so the route needs to know which one is installed")
    if version not in TORCH:
        supported = ", ".join(f"{v[0]}.{v[1]}" for v in sorted(TORCH))
        return None, (f"ROCm {version[0]}.{version[1]} is not one this route installs. "
                      f"PyTorch publishes ROCm builds for: {supported}")
    return "source", (f"PyTorch {TORCH[version][0]} for ROCm {version[0]}.{version[1]}, "
                      f"extensions compiled for {ARCH_SEPARATOR.join(gfx)}")


def detect() -> tuple[str | None, str]:
    return choose_route(host.rocm_version(), host.find_hipcc(), host.rocm_gfx_targets())


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    gfx = host.rocm_gfx_targets()
    lines = ["", "About to install:", "", "  backend: TRELLIS.2 (Microsoft)",
             f"  route:   AMD / ROCm on Linux, {route or 'none'}: {why}"]
    if code:
        lines.append(f"  code:    microsoft/TRELLIS.2 @ {pinned_commit()[:7]} -> "
                     "vendor/trellis-rocm/, plus its own venv (several GB of packages)")
        lines.append("           nvdiffrast from the dmonkman ROCm port; CuMesh, FlexGEMM and "
                     "o-voxel from")
        lines.append("           upstream at the same commits the NVIDIA route uses")
        lines.append(f"           all four compiled with hipcc for "
                     f"{ARCH_SEPARATOR.join(gfx) or 'your card'}: count on 20-40 minutes")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total -> Hugging Face cache")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {repo}")
        lines.append(f"           {GATED} is gated (see licence below)")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def arch_list(gfx: list[str]) -> str | None:
    """`GPU_ARCHS` for the extensions, which split it on a semicolon."""
    return ARCH_SEPARATOR.join(gfx) if gfx else None


def build_env(gfx: list[str], hipcc: str | None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for compiling the extensions with hipcc on this machine."""
    env = dict(os.environ if base is None else base)
    arch = arch_list(gfx)
    if arch:
        env["GPU_ARCHS"] = arch
    env["MAX_JOBS"] = str(host.build_jobs())
    if hipcc:
        root = host.rocm_root(hipcc)
        if root:
            env["ROCM_PATH"] = str(root)
        env["PATH"] = os.pathsep.join([str(Path(hipcc).parent), env.get("PATH", "")])
    return env


def install_env(gfx: list[str] | None = None, hipcc: str | None = None,
                base: dict[str, str] | None = None) -> dict[str, str]:
    """Compile settings plus patient downloads: torch's ROCm wheel is gigabytes."""
    gfx = host.rocm_gfx_targets() if gfx is None else gfx
    hipcc = host.find_hipcc() if hipcc is None else hipcc
    return host.patient_downloads(build_env(gfx, hipcc, base))


def basic_args(lock: Path = PACKAGE_LOCK) -> list[str]:
    """Everything but PyTorch and the compiled extensions.

    The CUDA route's lock is not reused: every pin in it was resolved against a CUDA torch,
    and those pins would drag a CUDA torch back in.
    """
    if lock.is_file() and lock.name.endswith("rocm.txt"):
        return ["-r", str(lock)]
    return list(BASIC_ROCM)


def pip_commands(route: str, uv: str, python: Path, version: tuple[int, int] | None = None,
                 checkout: Path = CHECKOUT,
                 extensions: Path = EXTENSIONS) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    if route != "source":
        raise ValueError(f"no install plan for route {route!r}")
    if version is None or version not in TORCH:
        raise ValueError(f"no PyTorch build for ROCm {version}")
    torch_version, vision_version = TORCH[version]
    pip = [uv, "pip", "install", "--python", str(python)]
    commands = [[*pip, "setuptools", "wheel", "ninja"]]
    commands.append([*pip, f"torch=={torch_version}", f"torchvision=={vision_version}",
                     "--index-url", host.torch_rocm_index(version)])
    commands.append([*pip, *basic_args()])
    commands.append([*pip, UTILS3D])
    # Pure Python and GPU-agnostic: the attention backend that replaces flash-attn here.
    commands.append([*pip, AULE_ATTENTION])
    # --no-build-isolation: the extensions must compile against the torch just installed,
    # not a fresh one pip would fetch into a throwaway build environment.
    for name, *_ in SOURCE_EXTENSIONS:
        commands.append([*pip, "--no-build-isolation", str(extensions / name)])
    # --no-deps: o-voxel names CuMesh and FlexGEMM as git dependencies, and would fetch and
    # compile both again over the ones just built.
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


def venv_python(root: Path = VENDOR) -> Path:
    return root / ".venv" / "bin" / "python"


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def install_code(version: tuple[int, int], route: str = "source") -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    gfx = host.rocm_gfx_targets()
    fetch_checkout(pinned_commit())
    for name, url, ref, recursive in SOURCE_EXTENSIONS:
        clone(url, EXTENSIONS / name, ref, recursive)

    # The build patch first: the extensions cannot compile without it on ROCm 7.x.
    changed = rocm_patch.apply(VENDOR)
    print(f"ROCm build patches: {sum(1 for _, was in changed if was)} of {len(changed)} "
          f"changed", flush=True)
    changed = bria_patch.apply(CHECKOUT)
    print(f"BRIA guardrail: {'applied' if changed else 'already present'}", flush=True)

    if not venv_python().is_file():
        run([uv, "venv", str(VENDOR / ".venv"), "--python", SOURCE_PYTHON])
    env = install_env(gfx)
    for command in pip_commands(route, uv, venv_python(), version):
        run(command, env=env)

    # Checked after the installs, not before: a patch that silently did not take would
    # otherwise be found on the first generation, minutes in.
    if not rocm_patch.is_patched(VENDOR):
        raise SystemExit("the ROCm patches did not all apply. Do not run a generation on "
                         "this checkout.")
    BUILT_MARKER.write_text(json.dumps(
        {"route": route, "torch": TORCH[version][0], "commit": pinned_commit(),
         "attention": AULE_ATTENTION, "gfx": gfx}) + "\n")


def check_access(repo: str = GATED) -> None:
    """Raise GatedAccess when Hugging Face will refuse the gated encoder. Seconds, no
    download: better than finding out after a 16 GB fetch and half an hour of compiling."""
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


def install_background_remover(target: Path | None = None, download=None) -> None:
    """The cut-out every upload without alpha goes through; without it the first
    generation fetched u2net (176 MB) unannounced."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and ROCm extensions, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    args = parser.parse_args(argv)

    if host.build_target() != "linux-amd":
        print("This installs TRELLIS.2 on Linux with an AMD card. On an NVIDIA card use "
              "scripts/bootstrap_trellis_cuda.py, and on a Mac "
              "scripts/bootstrap_trellis_space_macos.py. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    route, why = detect()
    version = host.rocm_version()
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
            install_code(version or (0, 0), route or "source")
        if weights:
            install_weights()
            install_background_remover()
    except GatedAccess as exc:
        print(gated_help(str(exc)))
        return 1
    print("\nDone. Pick TRELLIS.2 (AMD) in the studio, or run:\n"
          "    vendor/trellis-rocm/.venv/bin/python scripts/trellis_rocm_generate.py "
          "input.png output.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())