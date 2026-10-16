#!/usr/bin/env python3
"""Install Stable Fast 3D for Linux with an AMD card: its own venv, built with HIP.

This is the AMD twin of `bootstrap_sf3d.py`. Less work than the old note suggested.

**Why SF3D was left out of Linux entirely.** Its pins -- numpy 1.26, transformers 4.42, an
old huggingface-hub -- broke the lab's shared environment, so rather than give it its own
2 GB of packages to fight over them it was marked Apple-only. That reasoning was about the
shared interpreter, not about the card, and it does not hold once the route gets a venv of
its own: nothing here can reach the viewer's packages.

**What actually needed changing, upstream.** One line, in `texture_baker/setup.py`. It
decides whether to build the GPU kernels with `use_cuda = use_cuda and CUDA_HOME is not
None`, but `CUDA_HOME` is None on a ROCm build of PyTorch, so the `.cu` file was dropped
from the build and it produced a CPU-only baker without saying so. Everything else was
already there: `setup.py` has an `is_hip_extension` branch, and `CUDAExtension` dispatches
to `hipcc` on its own. `scripts/patch_sf3d_linux.py` makes that change.

PyTorch comes from ROCm's own index. PyPI's Linux torch is a CUDA build, so on an AMD card
it reports no device at all.

Then the **weights**, about 4.9 GB: SF3D itself and DINOv2-large, which SF3D would otherwise
fetch unannounced on its first run.

**Licence:** the SF3D weights are gated. Accept Stability's licence at
https://huggingface.co/stabilityai/stable-fast-3d and run `hf auth login` first. They are
free under $1M annual revenue and need a commercial licence above that.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_sf3d_rocm.py            # says what it wants, then asks
    python scripts/bootstrap_sf3d_rocm.py --yes      # for the viewer and for agents
    python scripts/bootstrap_sf3d_rocm.py --code-only
    python scripts/bootstrap_sf3d_rocm.py --weights-only
    python scripts/bootstrap_sf3d_rocm.py --rebuild  # build again over a finished build
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

import bootstrap_sf3d as apple  # noqa: E402, RUF100
import patch_sf3d_linux as sf3d_patch  # noqa: E402, RUF100

from image_to_3dlab import host, matte  # noqa: E402, RUF100

# The same checkout the Apple route uses: one clone, and each machine's venv beside it.
VENDOR = apple.VENDOR
CHECKOUT = VENDOR
UPSTREAM = apple.UPSTREAM
COMMIT = apple.COMMIT
PYTHON_VERSION = "3.11"
# Written last, so a build that died halfway never looks installed.
BUILT_MARKER = VENDOR / ".i2l-build-complete"

# The weights and the licence are SF3D's, unchanged. Sharing them means the two routes
# cannot drift apart in what they promise to fetch.
WEIGHTS = apple.WEIGHTS
LICENCE = apple.LICENCE

# (ROCm index tag, torch, torchvision, minimum ROCm). Read off PyTorch's own index.
TORCH = {
    "rocm7.2": ("2.14.1", "0.29.1", (7, 2)),
    "rocm7.1": ("2.13.0", "0.28.0", (7, 1)),
    "rocm7.0": ("2.10.0", "0.25.0", (7, 0)),
    "rocm6.4": ("2.9.1", "0.24.1", (6, 4)),
    "rocm6.3": ("2.9.1", "0.24.1", (6, 3)),
    "rocm6.2": ("2.5.1", "0.20.1", (6, 2)),
}


def total_gb() -> float:
    return apple.total_gb()


def choose_route(version: tuple[int, int] | None, hipcc: str | None,
                 gfx: list[str]) -> tuple[str | None, str]:
    """("rocm7.2" | ..., why). None means there is no way to install here."""
    if version is None:
        return None, ("no ROCm install answered (no /opt/rocm version and no `hipconfig`); "
                      "install ROCm first, https://rocm.docs.amd.com")
    if hipcc is None:
        return None, ("the HIP compiler (hipcc) is not installed; the texture baker is "
                      "compiled for your card and needs it")
    tag = host.torch_rocm_index(version)
    if tag is None:
        return None, (f"ROCm {version[0]}.{version[1]} is older than PyTorch's ROCm builds "
                      f"(they need {TORCH['rocm6.2'][2][0]}.{TORCH['rocm6.2'][2][1]}+)")
    name = tag.rsplit("/", 1)[-1]
    if name not in TORCH:
        return None, (f"ROCm {version[0]}.{version[1]} maps to PyTorch's {name} index, "
                      f"which this route has no tested torch for")
    if not gfx:
        return None, ("rocminfo did not report a gfx target, so there is nothing to "
                      "compile the texture baker for")
    return name, f"PyTorch {TORCH[name][0]} + {tag}, kernels for {', '.join(gfx)}"


def detect() -> tuple[str | None, str]:
    return choose_route(host.rocm_version(), host.find_hipcc(), host.rocm_gfx_targets())


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: Stable Fast 3D (Stability AI)",
             f"  route:   AMD / ROCm on Linux, {route or 'none'}: {why}"]
    if code:
        lines.append(f"  code:    Stability-AI/stable-fast-3d @ {COMMIT[:7]} -> "
                     "vendor/stable-fast-3d/, plus its own venv")
        lines.append("           in its own venv so its pins cannot reach the viewer's "
                     "packages")
        lines.append("           the texture baker is compiled with hipcc for your card")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>5.2f} GB  {repo}")
        lines.append("           stabilityai/stable-fast-3d is gated (see licence below)")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def build_env(gfx: list[str], hipcc: str | None,
              base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for compiling the texture baker with hipcc on this machine.

    USE_CUDA=1 is what tells `texture_baker/setup.py` to include the .cu files. It
    defaults from `torch.cuda.is_available()`, which is true under ROCm, but setting it
    means the build does not depend on the driver being visible to the installer.
    """
    env = dict(os.environ if base is None else base)
    if gfx:
        env["PYTORCH_ROCM_ARCH"] = ",".join(gfx)
    env["MAX_JOBS"] = str(host.build_jobs())
    env["USE_CUDA"] = "1"
    env["USE_METAL"] = "0"
    if hipcc:
        root = host.rocm_root(hipcc)
        if root:
            env["ROCM_HOME"] = str(root)
        env["PATH"] = os.pathsep.join([str(Path(hipcc).parent), env.get("PATH", "")])
    return env


def install_env(base: dict[str, str] | None = None) -> dict[str, str]:
    return host.patient_downloads(build_env(host.rocm_gfx_targets(), host.find_hipcc(), base))


def pip_commands(route: str, uv: str, python: Path) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested.

    One requirements install, because upstream's own list builds texture_baker and
    uv_unwrapper as local paths (`./texture_baker/`) and there is no reason to split them.
    It needs --no-build-isolation: both extensions import torch in their setup.py, so they
    must compile against the torch just installed rather than a fresh one.
    """
    if route not in TORCH:
        raise ValueError(f"no install plan for route {route!r}")
    torch_version, vision_version, _ = TORCH[route]
    pip = [uv, "pip", "install", "--python", str(python)]
    return [
        # --index-url, not --extra-index-url: the ROCm build has to win over PyPI's CUDA one.
        [*pip, f"torch=={torch_version}", f"torchvision=={vision_version}",
         "--index-url", host.TORCH_INDEX + route],
        # -r, not a bare path: uv reads a positional path as an archive to install and
        # rejects requirements.txt for not ending in .whl or .tar.gz.
        [*pip, "--no-build-isolation", "-r", str(CHECKOUT / "requirements.txt")],
    ]


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def venv_python() -> Path:
    return VENDOR / ".venv" / "bin" / "python"


def build_present() -> bool:
    """Finished, and still there: a marker beside a deleted venv is not a build to keep."""
    return BUILT_MARKER.is_file() and venv_python().is_file()


def fetch_checkout(commit: str = COMMIT) -> None:
    if not (CHECKOUT / ".git").is_dir():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", UPSTREAM, str(CHECKOUT)])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT,
                                   text=True).strip()
    if head != commit:
        run(["git", "fetch", "origin", commit], cwd=CHECKOUT)
        run(["git", "checkout", "--detach", commit], cwd=CHECKOUT)


def install_code(route: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout()
    # Before anything builds: an unpatched setup.py compiles a CPU-only baker and says
    # nothing, so this has to run first rather than be noticed afterwards.
    sf3d_patch.patch(CHECKOUT)
    # The venv folder is read off venv_python(), never spelled out again: a second spelling
    # is how a test that pointed venv_python() elsewhere deleted the real venv.
    venv = venv_python().parents[1]
    if not venv_python().is_file():
        shutil.rmtree(venv, ignore_errors=True)
        run([uv, "venv", str(venv), "--python", PYTHON_VERSION])
    env = install_env()
    torch, *rest = pip_commands(route, uv, venv_python())
    run(torch, env=env, cwd=str(CHECKOUT))
    prove_torch(venv_python())
    for command in rest:
        run(command, env=env, cwd=str(CHECKOUT))
    BUILT_MARKER.write_text(json.dumps({"route": route, "torch": TORCH[route][0],
                                        "commit": COMMIT}) + "\n")


def prove_torch(python: Path, probe=None) -> None:
    """Launch one kernel with the torch just installed, before the baker compiles on it.

    The same probe TRELLIS uses: a ROCm torch can install cleanly and carry no code for
    this card, and that otherwise surfaces as `hipErrorInvalidKernelFile` mid-generation.
    """
    if probe is None:
        from bootstrap_trellis_rocm import gpu_probe as probe
    ok, why = probe(python)
    if not ok:
        raise SystemExit(f"The ROCm torch just installed does not run on this card: {why}. "
                         "The texture baker was not compiled.")


def code_plan(code: bool, built: bool, rebuild: bool) -> str:
    """"build", "keep" or "skip". A finished build is kept unless --rebuild says otherwise,
    so a re-run after a gated-licence failure goes straight to the weights."""
    if not code:
        return "skip"
    return "build" if rebuild or not built else "keep"


def install_weights() -> None:
    """Fetch the weights into the Hugging Face cache, refusing a gated repo we cannot read."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. pip install -r requirements.txt"
                         ) from exc
    for repo, patterns, size in WEIGHTS:
        print(f"\nFetching {repo} ({size:.2f} GB)...", flush=True)
        try:
            snapshot_download(repo, allow_patterns=patterns)
        except Exception as error:
            if "gated" in str(error).lower() or "403" in str(error):
                raise SystemExit(
                    f"{repo} is gated and this machine cannot read it. Accept the licence "
                    f"at https://huggingface.co/{repo} and run `hf auth login`, then again. "
                    f"Nothing else was changed.") from error
            raise
    del hf_hub_download
    install_background_remover()


def install_background_remover(target: Path | None = None, download=None) -> None:
    """The cut-out sf3d_backend.py runs on every upload without alpha. Without it the first
    generation fell back to u2net and fetched it unannounced, as every other AMD route
    already guards against."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and compile its extensions, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the code even if a finished build is already there.")
    args = parser.parse_args(argv)

    if host.build_target() != "linux-amd":
        print("This installs Stable Fast 3D on Linux with an AMD card and ROCm. On a Mac "
              "use bootstrap_sf3d.py. Nothing downloaded.")
        return 1

    plan = code_plan(not args.weights_only, build_present(), args.rebuild)
    code = plan == "build"
    weights = not args.code_only
    route, why = detect()
    if code and route is None:
        print(f"Cannot install Stable Fast 3D here: {why}. Nothing downloaded.")
        return 1
    if plan == "keep":
        print(f"Stable Fast 3D is already built in {VENDOR}, leaving it alone "
              "(--rebuild builds it again).")
    if not code and not weights:
        return 0
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
    print("\nDone. Pick Stable Fast 3D in the studio, or run:\n"
          "    vendor/stable-fast-3d/.venv/bin/python scripts/sf3d_generate.py "
          "input.png output.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())