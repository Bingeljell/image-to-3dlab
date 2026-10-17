#!/usr/bin/env python3
"""Install TRELLIS.2 for Linux with an AMD card: Microsoft's own code, built with HIP.

The AMD twin of `bootstrap_trellis_cuda.py`. It runs the same pinned
`microsoft/TRELLIS.2` checkout, the same pipeline and the same weights. Of the four compiled
extensions, one is NVIDIA's own at the same pinned commit as the CUDA route, and three need
more:

- **FlexGEMM and o-voxel are upstream**, at the same commits the NVIDIA route uses. Both
  already dispatch on `IS_HIP_EXTENSION` and pass `--offload-arch`, and both only call the raw
  CUDA runtime, which hipcc aliases. Their one genuine problem is `-std=c++17` against
  torch 2.14's C++20 headers, which is three words in `patch_trellis_rocm.py`. So these two
  builds run identical source on both cards.
- **CuMesh and nvdiffrast come from dmonkman's ROCm ports.** nvdiffrast because upstream is
  CUDA-only with no HIP path to patch into. CuMesh because its eight `.cu` sources need a
  hipify — `cuda::` to hip, `cub::` to `hipcub::` — which is machine generated rather than
  something to hand-edit. Both ports still hardcode the ROCm 10 device-bitcode path and
  declare `requires-python >=3.12`; the same patch fixes both.
- **Attention needs a backend that has no CUDA extension at all.** TRELLIS.2's three are
  xformers, flash-attn and flash-attn-3, all compiled against CUDA. This installs
  AuleAttention, a Triton flash attention that needs no compilation, and the same patch
  adds it as a fourth backend.

The BRIA guardrail (`patch_trellis_cuda_no_bria.py`) applies unchanged: upstream loads
BRIA RMBG-2.0 by default and this repo must never load it.

**The torch is chosen by asking the card, not by looking at a table.** PyTorch's own index
carries device code for the ROCm releases it publishes and stops at rocm7.2; AMD's own index
publishes for newer releases, and its rocm10.1 wheel ships with an *empty* `.hip_fatbin` — no
code object at all — so on a ROCm 10.2 card it cannot launch a kernel and the failure lands
two minutes into the first generation as `invalid kernel file`. So `torch_candidates` offers
both, best first, and `install_torch` proves one on the card before a single extension is
compiled against it. Forty minutes of compiling is too late to find out.

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
import tempfile
from dataclasses import dataclass
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

# Two extensions come from dmonkman's ROCm ports, for two different reasons, and it matters
# which is which.
#
# nvdiffrast: upstream is CUDA-only with no HIP path to patch into, so there is no choice.
#   dmonkman/nvdiffrast-ROCm is NVIDIA's v0.4.0 plus that path.
# CuMesh: upstream dispatches on HIP, but its eight .cu sources need a hipify -- cuda:: to
#   hip, cub:: to hipcub:: -- which is machine generated and not something to hand-edit. The
#   fork is that hipify, applied to the commit pinned below. Checked by diff, not assumed:
#   building upstream CuMesh here fails on `no member named 'cuda' in the global namespace`.
NVDIFFRAST = ("nvdiffrast", "https://github.com/dmonkman/nvdiffrast-ROCm.git",
              "dd6f84d115d6c4c2a31d5cbcb7b225d5f7a46c5a", False)
CUMESH = ("CuMesh", "https://github.com/dmonkman/CuMesh-ROCm.git",
          "e28fe419201ce134deeccf0947bb97b8228f2e82", False)
# This one is NVIDIA's own, at the same commit the CUDA route uses. It needs no fork: it only
# calls the raw CUDA runtime, which hipcc aliases, and its one genuine problem was the C++
# standard. Verified by building it from upstream on this card, not assumed.
SOURCE_EXTENSIONS = (
    NVDIFFRAST,
    CUMESH,
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git",
     "6dd94a859c26ee8246888502eada3dd8ad85532e", True),
)

# Triton flash attention with no compiled extension, so it is the one backend that also
# runs on ROCm. See scripts/patch_trellis_rocm.py for the dispatch that selects it.
# Pinned: 0.5.0 is the release this route was tested with (newest on PyPI, 2025-12-20).
AULE_ATTENTION = "aule-attention==0.5.0"

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

# One candidate torch: where it comes from, what it is, and whether that index is the only
# source for it (`--index-url`) or a second one to search alongside PyPI (`--extra-index-url`).
@dataclass(frozen=True)
class Torch:
    index: str
    torch: str
    torchvision: str
    own_index: bool
    label: str


def torch_command(uv: str, python: Path, candidate: Torch) -> list[str]:
    """The one pip line that installs this candidate torch."""
    flag = "--index-url" if candidate.own_index else "--extra-index-url"
    return [uv, "pip", "install", "--python", str(python),
            f"torch=={candidate.torch}", f"torchvision=={candidate.torchvision}",
            flag, candidate.index]


# ROCm's HIP headers define __noinline__ to nothing
# (/opt/rocm/include/hip/amd_detail/host_defines.h). libstdc++ 16 writes
# [[__gnu__::__noinline__]] in <format>, so once that macro is defined the header parses as
# [[__gnu__::]] and every .cpp that reaches <chrono> fails with "expected an identifier for
# the attribute name". It is a collision between ROCm's headers and a 2026-era libstdc++,
# not a fault in the extensions.
#
# Reading <format> before the HIP headers define the macro fixes it: CPPFLAGS is honoured by
# setuptools and reaches exactly the host-compiler invocations, which are the ones that
# break. It is only added when the probe below says it is needed, so a machine with an older
# libstdc++ gets nothing.
FORMAT_WORKAROUND_FLAGS = "-include format"
NOINLINE_PROBE_SOURCE = (
    "#define __noinline__\n"      # what ROCm's host_defines.h does
    "#include <format>\n"         # what libstdc++ 16 then cannot parse
    "int main() { return 0; }\n"
)

# Upstream's --basic list minus what only training or the Gradio demo uses. transformers is
# pinned to the version that knows DINOv3. numpy and plyfile are o-voxel's, which installs
# with --no-deps. onnxruntime stays: rembg needs it to cut out pictures that arrive without
# alpha, and the current release installs and runs here. An old 1.16.3 wheel asked for an
# executable stack this kernel refused, and dropping it was the right answer at the time; it
# is simply not the wheel that gets resolved now.
BASIC_ROCM = list(BASIC)

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


def newest_known() -> tuple[tuple[int, int], tuple[int, int]]:
    """The newest ROCm each of the two indexes publishes a torch for, read off the tables
    rather than written out, so the refusal message cannot go stale when they grow."""
    pytorch = max(TORCH)
    amd = max(minimum for minimum, _, _ in host.AMD_TORCH_BUILDS)
    return pytorch, amd


def choose_route(version: tuple[int, int] | None, hipcc: str | None,
                 gfx: list[str]) -> tuple[str | None, str]:
    """("source" | None, why). None means there is no way to install here.

    Two indexes, not one. PyTorch's stops at rocm7.2; a newer machine gets a torch from AMD's
    own index, and falls back to PyTorch's newest if that one turns out not to run on the card.
    Refusing a ROCm 10.x card outright was wrong: the route does build on whatever machine it
    is run on, and "newer than PyTorch's index" is a different index, not a different job.
    """
    if not gfx:
        return None, ("no AMD GPU target found. Needs rocminfo, or a readable "
                      "/sys/class/kfd/kfd/topology; check the driver")
    if hipcc is None:
        return None, ("hipcc was not found. These extensions are compiled for your card and "
                      "need the ROCm toolkit")
    if version is None:
        return None, ("the ROCm version could not be read. PyTorch ROCm builds are published "
                      "per ROCm release, so the route needs to know which one is installed")
    try:
        candidates = torch_candidates(version)
    except ValueError:
        pytorch_newest, amd_newest = newest_known()
        known = sorted({*(v for v in TORCH),
                        *(minimum for minimum, _, _ in host.AMD_TORCH_BUILDS)})
        supported = ", ".join(f"{v[0]}.{v[1]}" for v in known)
        return None, (f"ROCm {version[0]}.{version[1]} is not one this route installs. "
                      f"PyTorch's index stops at {pytorch_newest[0]}.{pytorch_newest[1]} and "
                      f"AMD's stops at {amd_newest[0]}.{amd_newest[1]}; the route covers: "
                      f"{supported}")
    first = candidates[0]
    where = (f"from {first.label}" if len(candidates) == 1 else
             f"from {first.label}, or {candidates[-1].torch} from {candidates[-1].label} if "
             f"that cannot launch a kernel here")
    return "source", (f"PyTorch {first.torch} for ROCm {version[0]}.{version[1]} {where}, "
                      f"extensions compiled for {ARCH_SEPARATOR.join(gfx)}, the torch proved "
                      f"on the card before anything is compiled")


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
        lines.append("           nvdiffrast and CuMesh from the dmonkman ROCm ports (nvdiffrast has no "
                     "HIP path upstream;")
        lines.append("           CuMesh's sources need a hipify); FlexGEMM and o-voxel from "
                     "upstream at the")
        lines.append("           same commits the NVIDIA route uses")
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


def noinline_collides(compiler: str = "c++", tmp_dir: Path | None = None) -> bool:
    """Does this machine's libstdc++ collide with ROCm's empty __noinline__?

    Compiles one line with the macro defined, exactly as ROCm's headers define it. Seconds,
    and it is the difference between a build that starts and one that dies twenty minutes in
    with an error in a system header nobody asked for.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as work:
        probe = Path(work) / "noinline_probe.cpp"
        probe.write_text(NOINLINE_PROBE_SOURCE)
        result = subprocess.run(
            [compiler, "-std=c++20", "-c", str(probe), "-o", os.devnull],
            capture_output=True, text=True, check=False)
        return result.returncode != 0


def build_env(gfx: list[str], hipcc: str | None,
              base: dict[str, str] | None = None,
              format_workaround: bool | None = None) -> dict[str, str]:
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
    needs = noinline_collides() if format_workaround is None else format_workaround
    if needs:
        env["CPPFLAGS"] = FORMAT_WORKAROUND_FLAGS
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


def torch_index_for(version: tuple[int, int]) -> str:
    """Where torch comes from for this ROCm: PyTorch's index, or AMD's.

    The first candidate's index, so the announcement and the install cannot disagree. A
    machine whose first candidate turns out not to run on its card gets the next one, and the
    install says so.
    """
    return torch_candidates(version)[0].index


# A GPU op, a reduction and a synchronise. Deliberately tiny: it asks "does this torch launch a
# kernel on this card", which is the one question no table can answer and the one a generation
# run answers two minutes too late.
#
# Ones, not random numbers: a check that a random draw can fail rejects a working torch about
# one run in twenty (64 normal values all under 2 in size is not rare), and on a machine with
# one candidate that stops the whole setup.
GPU_PROBE_SOURCE = (
    "import torch\n"
    "x = torch.ones(8, 8, device='cuda')\n"
    "assert (x @ x).sum().item() == 512\n"
    "torch.cuda.synchronize()\n"
)

# Everything a candidate torch brings with it. Two indexes ship Triton under two names that
# write into one `triton/` folder, so a rejected candidate left in place is a later uninstall
# away from breaking the attention backend. Cleared before the next one goes in.
TORCH_FAMILY = ("torch", "torchvision", "triton", "triton-rocm", "pytorch-triton-rocm",
                "rocm", "rocm-bootstrap", "rocm-sdk-core", "rocm-sdk-libraries")


def clear_torch_command(uv: str, python: Path) -> list[str]:
    """Take a rejected candidate out again. uv skips names that are not installed."""
    return [uv, "pip", "uninstall", "--python", str(python), *TORCH_FAMILY]


def gpu_probe(python: Path | None = None, run=subprocess.run, timeout: int = 600) -> tuple[bool, str]:
    """Does the torch in this venv launch a kernel on this card? (works, why not).

    Seconds, and the difference between a build that starts and one that dies minutes into the
    first generation with `hipErrorInvalidKernelFile` and nothing to act on.
    """
    command = [str(python or venv_python()), "-c", GPU_PROBE_SOURCE]
    try:
        result = run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if result.returncode == 0:
        return True, ""
    lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
    return False, next((line for line in reversed(lines) if "Error" in line or "error" in line),
                       lines[-1] if lines else f"exit {result.returncode}")


def torch_pair_for(version: tuple[int, int]) -> tuple[str, str]:
    """The torch and torchvision versions to install for this ROCm: the first candidate's."""
    first = torch_candidates(version)[0]
    return first.torch, first.torchvision


def torch_candidates(version: tuple[int, int]) -> list[Torch]:
    """Every torch worth trying on this ROCm, best first.

    Two indexes, and which one a machine can actually run is a property of the machine, not of
    either table. PyTorch's own index carries device code for each card it is built for; AMD's
    multi-arch build does not always -- the rocm10.1 wheel installed here declared a 322 MB
    `.hip_fatbin` that was empty (NOBITS, with no bundle anywhere beside it) and failed on the
    first kernel with `hipErrorInvalidKernelFile`, while PyTorch's rocm7.2 wheel ran on the
    same card under the same ROCm 10.2 runtime without a word of complaint.

    So both are offered, in that order of preference, and `install_torch` proves one on the
    card before anything is compiled against it. Picking by table alone is how a 40-minute
    build ends in `invalid kernel file` two minutes into the first generation.
    """
    if version in TORCH:
        torch_version, vision_version = TORCH[version]
        return [Torch(host.torch_rocm_index(version), torch_version, vision_version, True,
                      f"PyTorch's ROCm {version[0]}.{version[1]} index")]
    candidates: list[Torch] = []
    amd = host.amd_torch_build(version)
    if amd is None:
        # Outside the range both tables cover. Refusing here rather than falling back to an
        # arbitrarily old wheel is the older, deliberate answer: a machine this old is not a
        # ROCm 10.x box that needs rescuing, it is a machine nobody has checked.
        raise ValueError(f"no torch for ROCm {version}")
    candidates.append(Torch(host.AMD_ROCM_INDEX, *amd, False, "AMD's own index"))
    # PyTorch stopped publishing at its newest tag, but a newer runtime runs an older wheel:
    # that is the whole point of ROCm's compatibility promise, and it is what this machine
    # needed once its first choice turned out to carry no device code.
    newest = max(TORCH)
    index = host.torch_rocm_index(newest)
    if index:
        torch_version, vision_version = TORCH[newest]
        candidates.append(Torch(index, torch_version, vision_version, True,
                                f"PyTorch's ROCm {newest[0]}.{newest[1]} index"))
    return candidates


def pip_commands(route: str, uv: str, python: Path, version: tuple[int, int] | None = None,
                 torch: Torch = None,
                 checkout: Path = CHECKOUT,
                 extensions: Path = EXTENSIONS) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    if route != "source":
        raise ValueError(f"no install plan for route {route!r}")
    if version is None:
        raise ValueError("no install plan without a ROCm version")
    torch = torch or torch_candidates(version)[0]
    pip = [uv, "pip", "install", "--python", str(python)]
    commands = [[*pip, "setuptools", "wheel", "ninja"]]
    # The torch line is here as well as in install_torch: it was already installed (and
    # proved) by then, so re-running it costs an audit rather than a download.
    commands.append(torch_command(uv, python, torch))
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


def install_torch(uv: str, version: tuple[int, int], env: dict[str, str],
                  python: Path | None = None, install=run, probe=gpu_probe,
                  prefer: str | None = None) -> Torch:
    """Install the first candidate torch that actually launches a kernel here, and return it.

    Forty minutes of compiling four extensions against a torch whose first kernel will not load
    is the worst way to find out, so this asks the card first. A candidate that fails is
    reported with its own error, taken out again, and the next one is tried; if none can,
    nothing is built.

    `prefer` is the index a previous build proved here. Tried first, so a re-run keeps the
    torch it has rather than swapping in a known-bad one only to swap it back out.
    """
    target = python or venv_python()
    rejected: list[str] = []
    candidates = sorted(torch_candidates(version), key=lambda c: c.index != prefer)
    for candidate in candidates:
        if rejected:
            install(clear_torch_command(uv, target), env=env)
        install(torch_command(uv, target, candidate), env=env)
        works, why = probe(target)
        if works:
            if rejected:
                print(f"torch {candidate.torch} from {candidate.label} runs on this card, "
                      f"after {', '.join(rejected)} did not", flush=True)
            else:
                print(f"torch {candidate.torch} from {candidate.label} runs on this card",
                      flush=True)
            return candidate
        print(f"torch {candidate.torch} from {candidate.label} does not run here: {why}",
              flush=True)
        rejected.append(f"{candidate.torch} ({candidate.label})")
    raise SystemExit(
        "No torch on the list launches a kernel on this card, so nothing built against one "
        f"would run either: {'; '.join(rejected)}. Install the torch your ROCm release "
        "publishes, or report this with the errors above.")


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


def recorded_torch_index(marker: Path | None = None) -> str | None:
    """The torch index the last finished build proved on this card, or None."""
    try:
        return json.loads((marker or BUILT_MARKER).read_text()).get("torch_index")
    except (OSError, ValueError, AttributeError):
        return None


def build_record(version: tuple[int, int], route: str, gfx: list[str],
                 torch: Torch = None) -> dict[str, object]:
    """What the build marker records. Pure, so it can be checked without a 40-minute build.

    `torch` is the candidate that was proved on the card, not the one the tables preferred:
    a machine that had to fall back should say so in the one file a later run reads.
    """
    torch = torch or torch_candidates(version)[0]
    return {"route": route, "torch": torch.torch, "torchvision": torch.torchvision,
            "torch_index": torch.index, "rocm": f"{version[0]}.{version[1]}",
            "commit": pinned_commit(), "attention": AULE_ATTENTION, "gfx": list(gfx)}


def install_code(version: tuple[int, int], route: str = "source") -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    gfx = host.rocm_gfx_targets()
    # The plan is settled, and the torch candidates resolved, before anything is cloned or
    # compiled: a machine this route cannot install for should hear so in a second, not after
    # a gigabyte of git. `torch_candidates` raises for a ROCm release nothing publishes for.
    torch_candidates(version)
    record = build_record(version, route, gfx)
    # The old marker goes before anything changes: a re-run that stops halfway leaves a venv
    # that is no longer the one it describes, and the card would still call it installed.
    proved = recorded_torch_index()
    BUILT_MARKER.unlink(missing_ok=True)
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
    # Torch first, and proved, because everything after this point is compiled against it.
    torch = install_torch(uv, version, env, prefer=proved)
    record = build_record(version, route, gfx, torch)
    for command in pip_commands(route, uv, venv_python(), version, torch=torch):
        run(command, env=env)

    # Checked after the installs, not before: a patch that silently did not take would
    # otherwise be found on the first generation, minutes in.
    if not rocm_patch.is_patched(VENDOR):
        raise SystemExit("the ROCm patches did not all apply. Do not run a generation on "
                         "this checkout.")
    BUILT_MARKER.write_text(json.dumps(record) + "\n")


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