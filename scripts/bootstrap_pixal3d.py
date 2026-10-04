#!/usr/bin/env python3
"""Install Pixal3D (raven38/pixal3d.cpp): a `trellis-cli` build plus its Q8_0 weights.

Two halves, the same way the viewer tracks every backend. The **build** depends on the
machine:

- **Apple Silicon:** cloned and compiled from source, because Metal kernels need the
  local Xcode toolchain. That needs full Xcode, not just the Command Line Tools.
- **Linux or Windows with an NVIDIA card:** upstream's prebuilt CUDA 12 build, runtime
  included, when the driver is new enough for it (575+, i.e. CUDA 12.9). About a minute,
  no compiler. It runs upstream's 12 steps (our 8-step patch needs a source build); on
  NVIDIA that costs seconds, while a compile costs 15 minutes on a fast pod and far more
  on an 8-CPU machine.
- **Linux, driver too old for the prebuilt, CUDA toolkit present:** compiled locally for
  this card instead. `--compile` asks for that even when the prebuilt would run. With
  neither, it says which driver to install and stops.

The **weights** are the single-view Q8_0 set plus the BiRefNet matting model, 8.4 GB, and
BiRefNet-lite (224 MB), the background remover Pixal3D's cut-out uses. Without lite the
cut-out falls back to u2net, which ate a white robot's upper arms on a fresh install.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_pixal3d.py            # says what it wants, then asks
    python scripts/bootstrap_pixal3d.py --yes      # for the viewer and for agents
    python scripts/bootstrap_pixal3d.py --build-only
    python scripts/bootstrap_pixal3d.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host, matte

VENDOR = REPO / "vendor" / "pixal3d-cpp"
BUILD = VENDOR / "build"
MODELS = VENDOR / "models" / "pixal3d-sv"
UPSTREAM = "https://github.com/raven38/pixal3d.cpp.git"
WEIGHTS_REPO = "raven38/pixal3d-sv-q8_0-v1"
MATTE_REPO = "ilintar/trellis2-gguf"
WEIGHTS_GB = 8.4
# Exact Hugging Face revisions, so a fresh install gets the files that were tested
# (fresh NVIDIA pod, 2026-10-01), not whatever was pushed since.
WEIGHTS_REVISION = "46d399ac986f45a0d7f5b1ca5058614d8729a131"
MATTE_REVISION = "a57397bd3d351599d9729fc144b3f87c3f87d65b"

# Pinned, not "latest": every upstream release so far is a pre-release, and a prebuilt
# that has been run end to end is worth more than a newer one that has not.
PREBUILT_RELEASE = "v0.10.1-desktop-alpha"
# The commit that tag points at. A tag can be moved; a commit cannot, so the CUDA source
# build fetches this.
PREBUILT_COMMIT = "1f432fd3f0689c504fa1e9b15b038c33584174d1"
RELEASE_API = "https://api.github.com/repos/raven38/pixal3d.cpp/releases/tags/{tag}"

# CUDA 12 rather than upstream's unversioned CUDA build (which is newer): CUDA 12 runs on
# older drivers, and the runtime is bundled either way. "Older" has a floor, though: the
# CUDA 12 build is compiled with 12.9, and on a 12.8 driver it died at its first kernel
# ("the provided PTX was compiled with an unsupported toolchain", RunPod 4090, 2026-09-23).
PREBUILT_MIN_CUDA = (12, 9)
PREBUILT_MIN_DRIVER = "575"
PREBUILTS = {
    "linux-nvidia": ("trellis-cuda12-linux-x64.tar.gz", "~640 MB"),
    "windows-nvidia": ("trellis-cuda12-windows-x64.zip", "~610 MB"),
}

LICENCE = (
    "MIT (code and flow weights); the bundled image encoder is under the\n"
    "  DINOv3 License. https://huggingface.co/raven38/pixal3d-sv-q8_0-v1"
)

# Looked up through the module so a test can pretend to be another machine.
target = host.build_target
driver_cuda = host.driver_cuda_version
find_nvcc = host.find_nvcc
nvcc_cuda = host.nvcc_cuda_version


def build_kind(key: str | None, cuda: tuple[int, int] | None, nvcc: str | None,
               nvcc_cuda: tuple[int, int] | None = None,
               prefer_compile: bool = False) -> str | None:
    """`prebuilt`, `cuda-source`, `metal-source`, or None when nothing will run here.

    The prebuilt wins on NVIDIA whenever the driver can run it: a minute to install, where
    a compile took 15 minutes on a 9-vCPU A40 pod (2026-10-02). The compile is faster per
    model (about 190 s against 390 s on a 4090) but that was never worth the wait. It is
    the fallback for an old driver, and cannot run when the toolkit is newer than the
    driver; an unreadable toolkit version is tried, not refused.
    """
    if key == "macos-arm64":
        return "metal-source"
    prebuilt_runs = key in PREBUILTS and cuda is not None and cuda >= PREBUILT_MIN_CUDA
    # A Windows source build is a Visual Studio project of its own; not offered.
    compile_runs = (key == "linux-nvidia" and bool(nvcc)
                    and (nvcc_cuda is None or cuda is None or nvcc_cuda <= cuda))
    if prefer_compile and compile_runs:
        return "cuda-source"
    if prebuilt_runs:
        return "prebuilt"
    return "cuda-source" if compile_runs else None


def current_kind(key: str | None, prefer_compile: bool = False) -> str | None:
    nvcc = find_nvcc() if key == "linux-nvidia" else None
    return build_kind(key, driver_cuda() if key in PREBUILTS else None, nvcc,
                      nvcc_cuda(nvcc) if nvcc else None, prefer_compile)


def route_and_size(key: str | None,
                   prefer_compile: bool = False) -> tuple[str, str] | None:
    kind = current_kind(key, prefer_compile)
    if kind == "metal-source":
        return "built from source with Metal", "compiled locally, needs full Xcode"
    if kind == "cuda-source":
        return ("compiled locally with CUDA for this card",
                "10+ minutes of compiling, once, with the CUDA toolkit")
    if kind == "prebuilt":
        name, size = PREBUILTS[key]
        return f"CUDA 12 prebuilt ({name}, {PREBUILT_RELEASE})", size
    return None


def no_route_message(key: str | None) -> str:
    if key not in PREBUILTS:
        return ("Pixal3D needs an Apple Silicon Mac, or Linux/Windows with an NVIDIA card "
                "(nvidia-smi must list it). Nothing downloaded.")
    cuda = driver_cuda()
    have = f"{cuda[0]}.{cuda[1]}" if cuda else "unknown"
    need = f"{PREBUILT_MIN_CUDA[0]}.{PREBUILT_MIN_CUDA[1]}"
    lines = [(f"Your NVIDIA driver supports CUDA {have}; Pixal3D's prebuilt needs {need} "
              f"(driver {PREBUILT_MIN_DRIVER} or newer)."),
             f"Update the NVIDIA driver to {PREBUILT_MIN_DRIVER}+ and run this again."]
    if key == "linux-nvidia":
        lines.append("Or install the CUDA toolkit (nvcc) and this compiles Pixal3D locally.")
    lines.append("Nothing downloaded.")
    return "\n".join(lines)


def announcement(build: bool = True, weights: bool = True,
                 prefer_compile: bool = False) -> str:
    """Exactly what is about to be fetched, before anything is."""
    found = route_and_size(target(), prefer_compile)
    route = found[0] if found else "none for this machine"
    lines = ["", "About to install:", "", "  backend: Pixal3D (raven38/pixal3d.cpp)",
             f"  route:   {route}"]
    if build and found:
        lines.append(f"  build:   {found[1]} -> vendor/pixal3d-cpp/build/")
    if weights:
        lines.append(f"  weights: {WEIGHTS_GB:.1f} GB -> vendor/pixal3d-cpp/models/pixal3d-sv/")
        lines.append(f"             {WEIGHTS_REPO}, plus BiRefNet matting ({MATTE_REPO})")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    note = old_driver_note(target(), prefer_compile)
    if note:
        lines += ["", "  " + note]
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def old_driver_note(key: str | None, prefer_compile: bool = False) -> str | None:
    """Why this NVIDIA machine compiles, when the reason is only an old driver.

    Without it the installer just starts a 10+ minute compile, and nobody learns that a
    driver update would have made it a one-minute download.
    """
    if prefer_compile or key not in PREBUILTS or current_kind(key) != "cuda-source":
        return None
    cuda = driver_cuda()
    have = f"CUDA {cuda[0]}.{cuda[1]}" if cuda else "an unknown CUDA version"
    return (f"note:    your NVIDIA driver supports {have}, so Pixal3D compiles here. "
            f"With driver {PREBUILT_MIN_DRIVER}+ it installs ready-made in about a minute.")


def cli_path() -> Path:
    return host.executable(BUILD, "trellis-cli")


def build_present() -> bool:
    """A function so the idempotence check is testable without touching the disk."""
    return cli_path().exists()


def pick_prebuilt(assets: list[dict], key: str) -> dict | None:
    wanted = PREBUILTS[key][0]
    return next((a for a in assets if a.get("name") == wanted), None)


def unpack_prebuilt(archive: Path, destination: Path) -> Path:
    """Unpack a prebuilt into `destination` and return the runnable `trellis-cli`.

    The Linux tarball's library symlinks (`libcudart.so.12 -> libcudart.so.12.9.79`) must
    survive; the loader looks for the short names.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            # The "data" filter refuses paths that escape `destination`; it exists on
            # 3.11.4+ and 3.12+, and older interpreters get the plain extract.
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(destination, filter="data")
            else:
                bundle.extractall(destination)
    cli = host.executable(destination, "trellis-cli")
    if not cli.exists():
        raise SystemExit(f"{archive.name} contained no trellis-cli.")
    for name in ("trellis-cli", "trellis-server"):
        path = host.executable(destination, name)
        if path.exists() and not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def install_prebuilt(key: str) -> Path:
    print(f"Finding pixal3d.cpp release {PREBUILT_RELEASE}...", flush=True)
    try:
        url = RELEASE_API.format(tag=PREBUILT_RELEASE)
        with urllib.request.urlopen(url, timeout=30) as response:
            release = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach GitHub: {exc}") from exc
    asset = pick_prebuilt(release.get("assets", []), key)
    if asset is None:
        raise SystemExit(f"Release {PREBUILT_RELEASE} has no {PREBUILTS[key][0]}.")
    VENDOR.mkdir(parents=True, exist_ok=True)
    archive = VENDOR / asset["name"]
    print(f"Downloading {asset['name']} ({asset.get('size', 0) / 1e6:.0f} MB)...", flush=True)
    urllib.request.urlretrieve(asset["browser_download_url"], archive)
    try:
        cli = unpack_prebuilt(archive, BUILD)
    finally:
        archive.unlink(missing_ok=True)
    print(f"Installed {cli}")
    return cli


def cmake_flags(kind: str, nvcc: str | None = None, arch: str | None = None) -> list[str]:
    flags = ["-DCMAKE_BUILD_TYPE=Release"]
    if kind == "cuda-source":
        flags += ["-DGGML_CUDA=ON",
                  # `native` asks the card at configure time; a known arch skips that.
                  f"-DCMAKE_CUDA_ARCHITECTURES={arch or 'native'}",
                  f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    return flags


def build_command(jobs: int) -> list[str]:
    """Always with a job count. A bare `-j` is unbounded, and on a 96-CPU, 31 GB pod the
    kernel killed the nvcc jobs it started."""
    return ["cmake", "--build", str(BUILD), "--target", "trellis-cli", "-j", str(jobs)]


def fetch_source(ref: str) -> None:
    """Check out pixal3d.cpp at `ref` into VENDOR, which may already hold the weights.

    `git clone` refuses a non-empty directory, and a `--weights-only` run beforehand makes
    it non-empty, so this initialises in place and fetches instead.
    """
    if not (VENDOR / ".git").is_dir():
        print(f"Fetching {UPSTREAM} @ {ref}", flush=True)
        VENDOR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(VENDOR)], check=True)
        subprocess.run(["git", "-C", str(VENDOR), "remote", "add", "origin", UPSTREAM],
                       check=True)
        subprocess.run(["git", "-C", str(VENDOR), "fetch", "-q", "--depth", "1", "origin",
                        ref], check=True)
        subprocess.run(["git", "-C", str(VENDOR), "checkout", "-q", "FETCH_HEAD"],
                       check=True)
    print("Fetching vendored ggml and friends", flush=True)
    subprocess.run(["git", "-C", str(VENDOR), "submodule", "update", "--init",
                    "--recursive"], check=True)


def apply_steps_patch(runner=subprocess.run) -> bool:
    """Patch the fetched source so `--steps` works (scripts/patch_pixal3d_steps.py).

    Never fatal: if upstream moved the anchor the build still succeeds, and the wrapper's
    "auto" steps simply stay at 12. Returns whether the patch is in place.
    """
    result = runner([sys.executable, str(REPO / "scripts" / "patch_pixal3d_steps.py")],
                    capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("Warning: could not apply the steps patch; Pixal3D will run 12 steps.\n"
              f"  {(result.stderr or result.stdout).strip()}", flush=True)
        return False
    print(result.stdout.strip(), flush=True)
    return True


def source_ref(kind: str) -> str:
    """The Mac build has always tracked upstream's default branch; the CUDA build pins the
    commit the prebuilts come from, which is the one tested on NVIDIA."""
    return "HEAD" if kind == "metal-source" else PREBUILT_COMMIT


def build_from_source(kind: str) -> Path:
    """Clone and compile: Metal on a Mac, CUDA on Linux with the toolkit installed."""
    needed = ("cmake", "ninja", "git") if kind == "metal-source" else ("cmake", "git")
    for tool in needed:
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found. Install it and run this again.")
    if kind == "metal-source" and subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False).returncode != 0:
        # Printed rather than run: both need sudo or change a system-wide setting.
        raise SystemExit(
            "Metal compiler unavailable. With full Xcode installed, this is usually:\n"
            "  sudo DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer "
            "xcodebuild -license accept\n"
            "  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer "
            "xcodebuild -downloadComponent MetalToolchain\n"
            "then re-run this script with DEVELOPER_DIR set."
        )
    fetch_source(source_ref(kind))
    apply_steps_patch()
    flags = cmake_flags(kind, find_nvcc(), host.compute_capability())
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    print(f"Building ({'Metal' if kind == 'metal-source' else 'CUDA'})", flush=True)
    subprocess.run(["cmake", "-S", str(VENDOR), "-B", str(BUILD), *generator, *flags],
                   check=True)
    subprocess.run(build_command(host.build_jobs()), check=True)
    if not build_present():
        raise SystemExit("The build finished without trellis-cli.")
    return cli_path()


def install_build(key: str, kind: str) -> Path:
    return install_prebuilt(key) if kind == "prebuilt" else build_from_source(kind)


def flatten_matte(models: Path) -> None:
    """`trellis-cli` looks for models flat in `--models`; BiRefNet lands in `q8/`."""
    nested = models / "q8" / "birefnet.gguf"
    if nested.exists() and not (models / "birefnet.gguf").exists():
        shutil.move(str(nested), str(models / "birefnet.gguf"))
    if (models / "q8").is_dir() and not any((models / "q8").iterdir()):
        (models / "q8").rmdir()


def install_weights(models: Path = MODELS) -> None:
    # Plain HTTP rather than Xet, as the shell bootstrap this replaced always used.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    models.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching {WEIGHTS_REPO} ({WEIGHTS_GB:.1f} GB, resumable)...", flush=True)
    snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REVISION, local_dir=models,
                      max_workers=2)
    hf_hub_download(MATTE_REPO, "q8/birefnet.gguf", revision=MATTE_REVISION,
                    local_dir=models)
    flatten_matte(models)
    print(f"  weights in {models}")
    install_background_remover()


def install_background_remover(target: Path | None = None, download=None) -> None:
    """BiRefNet-lite, unless it is already there. Same file bootstrap_matte.py installs."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def rebuild_existing(runner=subprocess.run) -> Path:
    """Patch and recompile an existing source build in place.

    Not build_from_source: that re-checks for the Metal compiler, which a fresh Terminal
    usually cannot see (xcode-select points at the Command Line Tools), and refused on the
    maintainer's own Mac. The patches touch C++ only, so an incremental `cmake --build` of
    the already-configured tree is all a rebuild needs.
    """
    apply_steps_patch(runner)
    runner(build_command(host.build_jobs()), check=True)
    if not build_present():
        raise SystemExit("The rebuild finished without trellis-cli.")
    return cli_path()


def build_decision(present: bool, rebuild: bool, kind: str) -> str:
    """What to do about trellis-cli: "install", "keep", "rebuild" or "cannot-rebuild".

    An existing build is normally left alone. `--rebuild` recompiles a source build so it
    picks up this repo's patches (the 8-step default needs scripts/patch_pixal3d_steps.py
    compiled in); only the changed files rebuild, so it takes minutes and fetches no
    models. A prebuilt download has no source to patch, so it says so instead.
    """
    if not present:
        return "install"
    if not rebuild:
        return "keep"
    return "cannot-rebuild" if kind == "prebuilt" else "rebuild"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--build-only", action="store_true",
                        help="Install trellis-cli and stop, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming trellis-cli is present.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Recompile an existing source build so it picks up this "
                             "repo's patches (e.g. the 8-step default). No downloads.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile for this card with nvcc (Linux) even when the "
                             "prebuilt would run: ~2x faster per model, 15+ minutes once.")
    args = parser.parse_args(argv)

    key = target()
    kind = current_kind(key, args.compile)
    if kind is None:
        print(no_route_message(key))
        return 1

    build = not args.weights_only
    weights = not args.build_only
    print(announcement(build=build, weights=weights, prefer_compile=args.compile))

    if not args.yes:
        # Non-interactive without --yes must not silently proceed, and must not hang
        # waiting on a stdin nobody is attached to.
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if build:
        decision = build_decision(build_present(), args.rebuild, kind)
        if decision == "keep":
            print(f"\ntrellis-cli is already installed at {cli_path()}, leaving it alone "
                  "(--rebuild recompiles it with this repo's patches).")
        elif decision == "cannot-rebuild":
            print("\nThis machine uses the prebuilt trellis-cli download, which cannot be "
                  "patched or rebuilt; it keeps its stock settings (12 steps).")
        elif decision == "rebuild":
            print(f"\nRebuilding trellis-cli at {cli_path()} with this repo's patches.")
            rebuild_existing()
        else:
            install_build(key, kind)
    if weights:
        install_weights()
    print("\nDone. Generate with:\n"
          "    python scripts/pixal3d_generate.py input.png output.glb --res 1024")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
