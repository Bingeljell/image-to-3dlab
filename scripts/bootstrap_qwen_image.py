#!/usr/bin/env python3
"""Install the text-to-image route: a stable-diffusion.cpp binary and Qwen-Image weights.

Two halves, the same way the viewer tracks every other backend. The **build** is a
prebuilt `stable-diffusion.cpp` release binary. Nothing is compiled: upstream publishes a
Metal build for Apple Silicon, a CUDA build for Windows, a Vulkan build for Linux that
runs on NVIDIA cards, and a ROCm build for AMD cards on Linux. The **weights** are three
files totalling about 13.4 GB.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_qwen_image.py            # says what it wants, then asks
    python scripts/bootstrap_qwen_image.py --yes      # for the viewer and for scripts
    python scripts/bootstrap_qwen_image.py --build-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host
from image_to_3dlab.sdcpp import NO_AMD_GPU_HELP, NO_GPU_HELP, gpu_found

VENDOR = REPO / "vendor" / "sdcpp"
BINARY = host.executable(VENDOR, "sd-cli")
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/{which}"
# Every machine's engine is pinned to one release, so a fresh install gets what was tested
# rather than whatever upstream pushed last. On a Mac, the 22 Sep build took 134 s of a 249 s
# picture just decoding it (its Metal path for the VAE's 3D convolutions was slow); master-975
# decodes in 10 s, same weights, same settings (2026-10-19). Measured on the Mac only; the
# others take it as an upgrade until tested (its AMD build targets ROCm 10.1, the one before 7.14).
PINNED_RELEASE = "master-975-95b3090"
PINNED = dict.fromkeys(("macos-arm64", "windows-nvidia", "linux-nvidia", "linux-amd"), PINNED_RELEASE)
RELEASE_FILE = "release.txt"

# (repo, filename, approximate gigabytes). These are the files the upstream Qwen-Image 2.1
# guide names, at the quantisations that were measured to be worth it: Q8 for the image
# model, Q4_K_M for the text encoder, and the VAE unquantised because it is small anyway.
WEIGHTS = [
    ("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q8_0.gguf", 7.69),
    ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5.03),
    ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68),
]

# Optional: the LLM's vision projector. Edit mode needs it (drawing a back picture from the
# front for a front + back build); plain pictures do not. Fetched only with --vision.
VISION = ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", 0.75)

LICENCE = (
    "Qwen Research License. The pictures you generate are yours (Qwen confirmed it);\n"
    "  the weights stay on this machine. Built with Qwen.\n"
    "  https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
)


def release_url(key: str | None) -> str:
    return RELEASES.format(which=f"tags/{PINNED[key]}" if key in PINNED else "latest")


def record_release(destination: Path, tag: str) -> None:
    (destination / RELEASE_FILE).write_text(tag + "\n", encoding="utf-8")


def installed_release(destination: Path | None = None) -> str | None:
    """Which release is installed; None for installs from before we kept a record."""
    path = (destination or VENDOR) / RELEASE_FILE
    return path.read_text(encoding="utf-8").strip() or None if path.is_file() else None


def binary_stale(destination: Path | None = None) -> bool:
    """A pinned machine running any other release (or one we never recorded)."""
    key = target()
    return key in PINNED and installed_release(destination) != PINNED[key]


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def announcement(build: bool = True, weights: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is.

    A separate function so a test can assert the backend, the route, the size and the
    licence are all named without running a download.
    """
    build_for = BUILDS.get(target())
    route = build_for.route if build_for else "no prebuilt build for this machine"
    lines = ["", "About to install:", "", "  backend: Qwen-Image 2.1 (text to image)",
             f"  route:   GGUF via stable-diffusion.cpp, {route}"]
    if build and build_for:
        lines.append(f"  build:   prebuilt sd-cli release ({build_for.size}) -> vendor/sdcpp/")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total ->  Hugging Face cache")
        for repo, filename, size in WEIGHTS:
            lines.append(f"             {size:>5.2f} GB  {filename}  ({repo})")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def vision_announcement() -> str:
    repo, filename, size = VISION
    return "\n".join(["", "About to install:", "",
                      "  backend: Qwen-Image 2.1, edit mode (draws the back picture for",
                      "           front + back 3D builds)",
                      f"  weights: {size:.2f} GB  {filename}  ({repo}) -> Hugging Face cache",
                      "", "  licence: " + LICENCE, ""])


def install_vision() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    repo, filename, size = VISION
    print(f"\nFetching {filename} ({size:.2f} GB) from {repo}...", flush=True)
    print(f"  {hf_hub_download(repo_id=repo, filename=filename)}")


def binary_present() -> bool:
    """Whether sd-cli is already installed. A function rather than an inline
    `BINARY.exists()` so the idempotence check is testable without touching the disk."""
    return BINARY.exists()


@dataclass(frozen=True)
class Build:
    """One machine's prebuilt download: what to call it, how big, and which assets."""

    route: str
    size: str
    # Each inner tuple is one asset, found by all of its substrings. Matched by substring
    # rather than exact name because release names carry the builder's OS version
    # (`...-Darwin-macOS-26.6.2-arm64.zip`), which changes with upstream's CI and is none
    # of our business.
    assets: tuple[tuple[str, ...], ...]


BUILDS = {
    "macos-arm64": Build("Metal on Apple Silicon", "~35 MB", (("darwin", "arm64"),)),
    # The CUDA build needs the CUDA runtime DLLs beside it, which upstream ships as a
    # second archive. Without them sd-cli.exe fails to start with no useful message.
    "windows-nvidia": Build("CUDA 12 on Windows", "~850 MB, CUDA runtime included",
                            (("win-cuda12", "x64"), ("cudart", "win", "cu12"))),
    # Upstream publishes no Linux CUDA binary. Vulkan is the prebuilt that runs on an
    # NVIDIA card; the NVIDIA driver ships the Vulkan support it needs.
    "linux-nvidia": Build("Vulkan on Linux (NVIDIA)", "~40 MB",
                          (("linux", "x86_64", "vulkan"),)),
    # An AMD card with ROCm on it gets the native HIP build rather than the Vulkan one:
    # same weights, faster sampling. It carries ggml's HIP backend but not ROCm itself:
    # libamdhip64 and rocBLAS come from the system's ROCm, which the viewer puts on the
    # loader path for each run (image_to_3dlab.host.with_rocm_libraries).
    "linux-amd": Build("ROCm on Linux (AMD)", "~278 MB",
                       (("linux", "x86_64", "rocm"),)),
}


# Looked up through the module so a test can pretend to be another machine.
target = host.build_target


def pick_assets(assets: list[dict], build: Build) -> list[dict]:
    """Every archive `build` needs from a release's asset list, or [] if any is missing.

    All or nothing: half a Windows install (the binary without its CUDA runtime) is worse
    than a clear "not in this release".
    """
    picked = []
    for needles in build.assets:
        match = next((a for a in assets
                      if a.get("name", "").lower().endswith(".zip")
                      and all(n in a["name"].lower() for n in needles)), None)
        if match is None:
            return []
        picked.append(match)
    return picked


def install_binary(destination: Path | None = None) -> Path:
    destination = destination or VENDOR
    key = target()
    build = BUILDS.get(key)
    if build is None:
        raise SystemExit(
            "There is no prebuilt stable-diffusion.cpp for this machine. Supported: an "
            "Apple Silicon Mac, or Linux/Windows with an NVIDIA card, or Linux with an "
            f"AMD card and ROCm. Otherwise build it from source and put sd-cli in {destination}."
        )
    print(f"Finding stable-diffusion.cpp release {PINNED.get(key, '(latest)')}...")
    try:
        with urllib.request.urlopen(release_url(key), timeout=30) as response:
            release = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach GitHub: {exc}") from exc
    assets = pick_assets(release.get("assets", []), build)
    if not assets:
        raise SystemExit(
            f"Release {release.get('tag_name')} has no {build.route} build. "
            f"Build from source and put sd-cli in {destination}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        archive = destination / "release.zip"
        print(f"Downloading {asset['name']} ({asset.get('size', 0) / 1e6:.0f} MB)...",
              flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
            restore_symlinks(bundle, destination)
        archive.unlink(missing_ok=True)
    installed = finish_install(destination)
    record_release(destination, release.get("tag_name", ""))
    return installed


def restore_symlinks(bundle: zipfile.ZipFile, destination: Path) -> None:
    """Turn extracted symlink entries back into real links.

    `zipfile` writes a stored symlink as a plain text file holding its target, and the
    dynamic loader then dies with "file too short" on the first versioned library in the
    chain (`libggml.so` -> `.so.0` -> `.so.0.25.3`). The link target is relative to the
    link's own directory, so entries are fixed where they were extracted, before any
    flattening moves them.
    """
    for info in bundle.infolist():
        if (info.external_attr >> 28) != 0xA:
            continue
        target = bundle.read(info.filename).decode("utf-8").strip()
        path = destination / info.filename
        if not link_stays_inside(path, target, destination):
            continue
        if path.is_symlink() or not path.exists():
            continue
        if path.is_file() and path.read_bytes().decode("utf-8", "replace").strip() == target:
            path.unlink()
            os.symlink(target, path)


def link_stays_inside(link: Path, target: str, destination: Path) -> bool:
    """Whether a link at `link` pointing to `target` resolves inside `destination`.

    The archive decides where its links point, so one aimed at `/etc` or `../../..` is
    left as the plain file zipfile wrote rather than turned into a way out of the folder.
    """
    if os.path.isabs(target):
        return False
    resolved = Path(os.path.normpath(link.parent / target))
    root = Path(os.path.normpath(destination))
    return resolved == root or root in resolved.parents


def finish_install(destination: Path) -> Path:
    """Flatten a nested archive and make the binary runnable. Returns its path."""
    binary = host.executable(destination, "sd-cli")
    if not binary.exists():
        found = next(destination.rglob(binary.name), None)
        if found is None:
            raise SystemExit("The release archive contained no sd-cli binary.")
        # Some releases nest everything one directory down; flatten so the path the
        # catalogue probes is the path that exists.
        for item in found.parent.iterdir():
            shutil.move(str(item), str(destination / item.name))
        # rmdir what the move left behind, innermost first.
        for leftover in (found.parent, found.parent.parent):
            try:
                leftover.rmdir()
            except OSError:
                break
    # zipfile drops the executable bit, and a Linux build also needs it on sd-server.
    for name in ("sd-cli", "sd-server"):
        path = host.executable(destination, name)
        if path.exists():
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed {binary}")
    return binary


def probe_gpu(binary: Path = BINARY) -> str:
    """Start sd-cli just long enough to load its backends, and return what it printed.

    It is pointed at a model that does not exist, so it exits right after announcing
    which backends it found.
    """
    try:
        result = subprocess.run(
            [str(binary), "--diffusion-model", str(VENDOR / "no-such-model.gguf"),
             "-p", "probe"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or "") + (result.stderr or "")


# Where the Vulkan loader looks for driver files (ICDs). An NVIDIA container often mounts
# the driver library but none of these files, so sd-cli finds no GPU and runs on the CPU
# (RunPod A40, 2026-10-22). Writing one by hand made pictures take 29 s instead of minutes.
VULKAN_ICD_DIRS = (Path("/etc/vulkan/icd.d"), Path("/usr/share/vulkan/icd.d"),
                   Path("/usr/local/share/vulkan/icd.d"))
NVIDIA_GLX = "libGLX_nvidia.so.0"
NVIDIA_ICD = {"file_format_version": "1.0.0",
              "ICD": {"library_path": NVIDIA_GLX, "api_version": "1.3"}}


def user_icd_dir() -> Path:
    """The per-user ICD folder the loader also reads, so no root is needed to write one."""
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "vulkan" / "icd.d"


def nvidia_glx_present() -> bool:
    """Whether the NVIDIA driver's Vulkan library can be loaded on this machine."""
    import ctypes
    try:
        ctypes.CDLL(NVIDIA_GLX)
    except OSError:
        return False
    return True


def add_nvidia_icd(search_dirs: tuple[Path, ...] | None = None, write_dir: Path | None = None,
                   glx_present: bool | None = None) -> Path | None:
    """Write the NVIDIA ICD file if none exists and the driver library is there.

    Returns the file written, or None when there was nothing to do.
    """
    search_dirs = VULKAN_ICD_DIRS if search_dirs is None else search_dirs
    write_dir = user_icd_dir() if write_dir is None else write_dir
    glx_present = nvidia_glx_present() if glx_present is None else glx_present
    if not glx_present:
        return None
    for folder in (*search_dirs, write_dir):
        if folder.is_dir() and any(folder.glob("nvidia*.json")):
            return None
    write_dir.mkdir(parents=True, exist_ok=True)
    path = write_dir / "nvidia_icd.json"
    path.write_text(json.dumps(NVIDIA_ICD, indent=2) + "\n")
    return path


def install_weights() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    for repo, filename, size in WEIGHTS:
        print(f"\nFetching {filename} ({size:.2f} GB) from {repo}...", flush=True)
        path = hf_hub_download(repo_id=repo, filename=filename)
        print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--build-only", action="store_true",
                        help="Install the binary and stop, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming sd-cli is already present.")
    parser.add_argument("--vision", action="store_true",
                        help=f"Fetch only the {VISION[2]:.2f} GB vision projector edit mode "
                             "needs (back pictures for front + back builds).")
    args = parser.parse_args(argv)

    build = not args.weights_only and not args.vision
    weights = not args.build_only and not args.vision
    print(vision_announcement() if args.vision else announcement(build=build, weights=weights))

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
        if binary_present() and binary_stale():
            print(f"\nUpdating sd-cli at {BINARY} to {PINNED[target()]}: pictures about twice "
                  "as fast. The weights stay as they are.")
            install_binary()
        elif binary_present():
            print(f"\nsd-cli is already installed at {BINARY}, leaving it alone.")
        else:
            install_binary()
        # The Linux/Windows builds load their GPU backend at run time and fall back to
        # the CPU without a word if it cannot reach the driver. Catch that now, not
        # 13 GB later.
        help_text = NO_AMD_GPU_HELP if target() == "linux-amd" else NO_GPU_HELP
        if target() in ("linux-nvidia", "windows-nvidia", "linux-amd"):
            found = gpu_found(probe_gpu())
            if not found and target() == "linux-nvidia":
                icd = add_nvidia_icd()
                if icd is not None:
                    print(f"Vulkan had no NVIDIA driver file; wrote {icd}.")
                    found = gpu_found(probe_gpu())
            if not found:
                print("\n" + help_text + "\nThe weights were not downloaded.")
                return 1
            print("sd-cli found the GPU.")
    if weights:
        install_weights()
    if args.vision:
        install_vision()
        print("\nDone. Back pictures: python scripts/make_back_view.py front.png back.png")
        return 0
    print("\nDone. In the studio, press + Create and start from an idea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
