"""What each backend needs on disk, so nothing is ever downloaded by surprise.

`AGENTS.md` makes this a rule: no script, bootstrap, viewer button or agent tool may fetch
weights until the user has confirmed which pipeline and which route. A rule needs numbers
to be honest about, and until now the viewer had none: `/api/backends` described stages
and settings but never said that picking TRELLIS.2 means ~16 GB from Hugging Face.

This module is that missing half. It is deliberately pure data plus pure functions over a
filesystem, with no imports from `generate_api`, so the onboarding screen, the CLI
bootstraps and the tests can all read the same catalogue without a circular import.

**Sizes are approximate and measured, not authoritative.** They come from a real install
on 2026-09-21 (`du -sh`) and exist to set expectations before a download, not to verify
one. Treat a mismatch as a stale constant, never as a failure.

**Licence text here is a pointer, not advice.** Each entry names the licence and links to
the original; reading it is the user's business. The one exception is a restriction with
real teeth, such as Hunyuan's territorial limits, which is surfaced as a caveat because
silently downloading those weights in a restricted region is a harm we would be causing.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import host as _host
from image_to_3dlab import matte as _matte
from image_to_3dlab.host import AMD, APPLE, NVIDIA
from image_to_3dlab.provenance import QWEN_OUTPUT_RIGHTS

HF_HUB_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

GB = 1024 ** 3

# Which machines a backend can run on is per-backend data rather than one "is this a
# Mac?" test, so a route gains NVIDIA support by adding a string to its `runs_on`. The
# detection itself lives in `image_to_3dlab.host`, shared with the bootstraps.
PLATFORM_LABELS = {APPLE: "an Apple Silicon Mac", NVIDIA: "an NVIDIA GPU",
                   AMD: "an AMD GPU with ROCm"}
# Setup page tabs, in order. Every tab names the backends its machine can run.
TAB_LABELS = {APPLE: "Mac (Apple Silicon)", NVIDIA: "NVIDIA (Linux)",
              AMD: "AMD (ROCm, Linux)"}
VIEW_PLATFORMS = (APPLE, NVIDIA, AMD)


def venv_python(project: Path) -> Path:
    """The interpreter inside a project's `.venv`, named the way this OS names it.

    Windows puts it in `Scripts/python.exe`, everywhere else it is `bin/python`. Hardcoding
    the POSIX spelling is what turned a Windows visit into "[WinError 2] The system cannot
    find the file specified" with no clue as to which file.
    """
    if os.name == "nt":
        return project / ".venv" / "Scripts" / "python.exe"
    return project / ".venv" / "bin" / "python"


def host_platform() -> str:
    """What this machine is: APPLE, NVIDIA, AMD or "other". See `image_to_3dlab.host`.

    Kept as a name here because the viewer and its tests reach for it on this module.
    """
    return _host.host_platform()


def host_label(host: str) -> str:
    """This machine, named the way its owner would name it."""
    if host in PLATFORM_LABELS:
        return PLATFORM_LABELS[host].removeprefix("an ").removeprefix("a ")
    return f"{platform.system() or 'this machine'} ({platform.machine()})"


def runs_on_phrase(backend: Backend) -> str:
    return " or ".join(PLATFORM_LABELS.get(p, p) for p in backend.runs_on)


@dataclass(frozen=True)
class WeightSet:
    """One downloadable unit, and where it lands."""

    label: str
    source: str
    bytes_expected: int
    path: Path
    note: str | None = None

    def describe(self) -> dict[str, Any]:
        present, actual = _dir_state(self.path)
        return {
            "label": self.label,
            "source": self.source,
            "source_url": self.source if self.source.startswith("https://")
            else f"https://huggingface.co/{self.source}",
            "bytes_expected": self.bytes_expected,
            "human_expected": human_bytes(self.bytes_expected),
            "present": present,
            "bytes_present": actual,
            "human_present": human_bytes(actual),
            "path": str(self.path),
            "note": self.note,
        }


HOST_OVERRIDABLE = frozenset({"label", "tradeoff", "caveat", "setup_fetches_weights",
                              "setup_minutes"})


@dataclass(frozen=True)
class Backend:
    """One generation route, as the onboarding table presents it."""

    id: str
    label: str
    best_for: str
    tradeoff: str
    license_name: str
    license_url: str
    weights: tuple[WeightSet, ...]
    install: str
    rank: int | None = None
    setup_minutes: int | None = None
    caveat: str | None = None
    # The gated Hugging Face repo the caveat warns about, if that is what it warns about.
    # The Setup page drops the caveat once the sign-in check says the account has access.
    gated_repo: str | None = None
    # Whether running this backend's setup actually fetches the weights. TRELLIS's
    # bootstrap does not: it clones, patches and builds the Metal port, and the weights
    # arrive lazily on the first generation run. The distinction changes what the
    # confirmation says and whether byte progress means anything.
    setup_fetches_weights: bool = True
    # Files that prove the code side is installed: a compiled binary, a venv interpreter.
    # Weights and build are independent, and conflating them offers "Set up" to someone
    # who already has the build, which re-runs a bootstrap that then fails on its own
    # already-applied patches (hit for real 2026-09-21).
    build_probes: tuple[Path, ...] = ()
    extra_steps: tuple[str, ...] = field(default_factory=tuple)
    # The machines this route works on. Default rather than per-entry because every route
    # is Apple-only today; the day one of them runs on CUDA, it says so here.
    runs_on: tuple[str, ...] = (APPLE,)
    # Whether the viewer can install this route by itself. Two of them it cannot: SF3D
    # wants a shell bootstrap and dgrauet's Hunyuan shape stage is a manual vendor clone
    # because it is Tencent-licensed. Listing them with a button that throws is worse than
    # listing them with the command to run, so the page says which it is.
    automated_setup: bool = True
    # What this route produces. Everything here made a mesh until Qwen-Image arrived, and a
    # text-to-image step sits one stage upstream of the rest of the pipeline: it is for
    # people who do not have a source image yet. The page groups on this rather than
    # guessing from the label.
    kind: str = "3d"
    # The official project behind a Mac port, as (label, url). TRELLIS.2 and Hunyuan3D are
    # NVIDIA-first upstream; only our wrappers are Apple-only. Without this, a Linux user
    # read "needs Apple Silicon" as if the model itself could not run on their card.
    upstream: tuple[str, str] | None = None
    # A route that installs differently per machine (TRELLIS.2: the Metal port on a Mac,
    # Microsoft's own code on NVIDIA) overrides `install` and `build_probes` per host here.
    # Hosts not listed use the plain fields.
    install_by_host: dict[str, str] = field(default_factory=dict)
    build_probes_by_host: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    # What the card says, per machine, where one route is really different code: TRELLIS
    # is the Metal port on a Mac and Microsoft's own code on NVIDIA. Keys: HOST_OVERRIDABLE.
    overrides_by_host: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Operating systems a listed machine type still cannot use, as (host, os family):
    # TRELLIS.2 runs on NVIDIA under Linux, but its CUDA build is not set up for Windows.
    excludes: tuple[tuple[str, str], ...] = ()
    # A separate route in this catalogue that does the same job on NVIDIA (the MLX Hunyuan
    # ports -> Tencent's own Hunyuan3D-2.1). Named in the note an NVIDIA machine sees.
    nvidia_route: str | None = None
    # Why an AMD machine cannot run this route, when the honest reason is specific.
    # TRELLIS.2 and Hunyuan3D-2.1 need NVIDIA-only compiled extensions; Stable Fast 3D is
    # Apple-only because its pins broke the lab's shared environment, which is a different
    # sentence entirely. Without this field every blocked route borrows the extension
    # excuse, and SF3D ends up claiming an NVIDIA-only extension it does not have.
    amd_blocked_reason: str | None = None

    @property
    def bytes_expected(self) -> int:
        return sum(w.bytes_expected for w in self.weights)

    def install_for(self, host: str | None = None) -> str:
        return self.install_by_host.get(host or host_platform(), self.install)

    def probes_for(self, host: str | None = None) -> tuple[Path, ...]:
        return self.build_probes_by_host.get(host or host_platform(), self.build_probes)

    def built_on(self, host: str | None = None) -> bool:
        """True when nothing is declared, so a weights-only backend is never 'unbuilt'."""
        return all(p.exists() for p in self.probes_for(host))

    @property
    def build_present(self) -> bool:
        return self.built_on()

    def excluded_os(self, host: str | None = None, family: str | None = None) -> str | None:
        """The OS family that rules this machine out, or None."""
        family = family or _host.os_family()
        return family if ((host or host_platform()), family) in self.excludes else None

    def runs_here(self, host: str | None = None, family: str | None = None) -> bool:
        host = host or host_platform()
        return host in self.runs_on and self.excluded_os(host, family) is None

    def _platform_note(self, host: str | None = None, family: str | None = None) -> str:
        excluded = self.excluded_os(host, family)
        if excluded:
            return (f"{self.label} is not supported on {excluded.capitalize()} yet. "
                    f"On an NVIDIA card it runs under Linux.")
        # AMD comes before the generic notes below, because both of those talk about
        # NVIDIA and an AMD reader has no NVIDIA machine to act on.
        if (host or host_platform()) == AMD and AMD not in self.runs_on:
            reason = self.amd_blocked_reason or (
                f"{self.label} needs NVIDIA-only compiled extensions, and upstream ships no "
                f"ROCm build of them, so a download here would end in a failed build.")
            return (f"Not on AMD yet. {reason} Pixal3D runs on this card and needs nothing "
                    f"NVIDIA-only.")
        # The "use the official version" advice is only true while this lab cannot run it
        # on NVIDIA itself; once a route lists NVIDIA, the plain note is the honest one.
        twin = BY_ID.get(self.nvidia_route) if self.nvidia_route else None
        if twin is not None and (host or host_platform()) == NVIDIA:
            return (f"This lab runs the Apple Silicon port here. On an NVIDIA machine, use "
                    f"{twin.label} instead: {self.upstream[0] if self.upstream else 'it'} "
                    f"with the vendor's own code, set up from this page.")
        if self.upstream and NVIDIA not in self.runs_on:
            return (f"This lab runs the Apple Silicon port. {self.upstream[0]} itself is "
                    f"built for NVIDIA: on an NVIDIA machine, use the official version "
                    f"for now. Built into this lab later.")
        return (f"Needs {runs_on_phrase(self)}. Setting it up on this machine would "
                f"download gigabytes and then fail, so the button is off.")

    def setup_fetches_on(self, host: str | None = None) -> bool:
        """Whether this machine's setup downloads the weights (vs. the first run doing so)."""
        said = self.overrides_by_host.get(host or host_platform(), {})
        return said.get("setup_fetches_weights", self.setup_fetches_weights)

    def describe(self, host: str | None = None) -> dict[str, Any]:
        weights = [w.describe() for w in self.weights]
        present = sum(w["bytes_present"] for w in weights)
        built = self.built_on(host)
        supported = self.runs_here(host)
        said = self.overrides_by_host.get(host or host_platform(), {})
        fetches = self.setup_fetches_on(host)
        return {
            "build_present": built,
            "supported_here": supported,
            "requires": runs_on_phrase(self),
            # Said once, in words, so the screen can explain instead of a button failing.
            "platform_note": None if supported else self._platform_note(host),
            "upstream": ({"label": self.upstream[0], "url": self.upstream[1]}
                         if self.upstream else None),
            "id": self.id,
            "label": said.get("label", self.label),
            "kind": self.kind,
            "rank": self.rank,
            "recommended": self.rank == 1,
            "best_for": self.best_for,
            "tradeoff": said.get("tradeoff", self.tradeoff),
            "license": {"name": self.license_name, "url": self.license_url},
            "caveat": said.get("caveat", self.caveat),
            "gated_repo": self.gated_repo,
            "install": self.install_for(host),
            "setup_minutes": said.get("setup_minutes", self.setup_minutes),
            "setup_fetches_weights": fetches,
            "extra_steps": list(self.extra_steps),
            "weights": weights,
            "bytes_expected": self.bytes_expected,
            "human_expected": human_bytes(self.bytes_expected),
            "bytes_present": present,
            "human_present": human_bytes(present),
            # What Remove would free: files other routes share are kept (see is_shared).
            "bytes_removable": sum(d["bytes_present"] for d, w in zip(weights, self.weights)
                                   if not is_shared(self, w)),
            "automated_setup": self.automated_setup,
            "state": "unsupported" if not supported else
                     _state(weights, built, fetches),
            "action": "none" if not supported else
                      _action(weights, built, fetches,
                              self.automated_setup),
            "percent_present": _percent(present, self.bytes_expected),
        }


# Ranked the way the README ranks them, because two orderings of the same advice is one
# too many. Rank 1 is what a newcomer should pick.
CATALOG: tuple[Backend, ...] = (
    Backend(
        id="pixal3d",
        label="Pixal3D (C++/GGML)",
        rank=1,
        best_for="Best results we have. One pass, no repaint needed.",
        tradeoff=(
            "On a Mac it compiles locally and needs full Xcode for the Metal compiler. "
            "On NVIDIA it downloads a ready-made CUDA build; no compiling. On AMD it "
            "compiles with HIP; no prebuilt is published."
        ),
        overrides_by_host={
            APPLE: {"tradeoff": "Compiles on your Mac, and needs full Xcode for the Metal "
                                "compiler."},
            NVIDIA: {"tradeoff": "Downloads a ready-made CUDA build, no compiling "
                                 "(driver 575 or newer)."},
            AMD: {"tradeoff": "Compiles on your machine with HIP, which needs ROCm's "
                              "hipcc. Expect 10-20 minutes of compiling, once."},
        },
        license_name="MIT (code + flow weights); DINOv3 License (bundled encoder)",
        license_url="https://huggingface.co/raven38/pixal3d-sv-q8_0-v1",
        install="scripts/bootstrap_pixal3d.py",
        runs_on=(APPLE, NVIDIA, AMD),
        setup_minutes=20,
        build_probes=(_host.executable(REPO / "vendor" / "pixal3d-cpp" / "build", "trellis-cli"),),
        weights=(
            # One entry, not two: the bootstrap moves BiRefNet *into* pixal3d-sv/, so a
            # second set pointed at the parent directory would count everything twice.
            WeightSet("Pixal3D single-view Q8_0, with BiRefNet matting",
                      "raven38/pixal3d-sv-q8_0-v1", int(8.4 * GB),
                      REPO / "vendor" / "pixal3d-cpp" / "models" / "pixal3d-sv",
                      note="Includes the BiRefNet matting model (ilintar/trellis2-gguf), "
                           "used to cut out a subject when the image has no alpha."),
            # Installed with Pixal3D since 0.3.5: without it the cut-out falls back to
            # u2net, which ate a white robot's upper arms on a fresh Linux install.
            WeightSet("BiRefNet-lite background remover", _matte.LITE_URL, _matte.LITE_BYTES,
                      _matte.model_file(_matte.LITE_MODEL),
                      note="Cuts the subject out before generation, keeping thin and "
                           "light-coloured parts. Shared with the other routes."),
        ),
    ),
    Backend(
        id="hunyuan_xiong",
        label="Hunyuan3D-MLX (Xiong, full pipeline)",
        rank=2,
        best_for="Fast, clean results, and the quickest to run from a fresh clone.",
        tradeoff="Shape and paint are separate venvs, each set up on its own.",
        license_name="MIT (code); Tencent Hunyuan Community License (weights)",
        license_url="https://huggingface.co/tencent/Hunyuan3D-2.1",
        install="python scripts/bootstrap_hunyuan_xiong.py",
        upstream=("Hunyuan3D-2.1", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"),
        nvidia_route="hunyuan-cuda",
        # Neither half of Hunyuan reaches an AMD card: this port is Apple Silicon, and
        # Tencent's own is NVIDIA-only. Saying so beats the generic note, which would
        # point an AMD reader at a machine they do not have.
        amd_blocked_reason=(
            "Hunyuan is not available on AMD in any form: this port is Apple Silicon, and "
            "Tencent's own needs NVIDIA-only compiled extensions."
        ),
        setup_minutes=25,
        build_probes=(venv_python(REPO / "hunyuan_mlx" / "shape"),
                      venv_python(REPO / "hunyuan_mlx" / "paint")),
        caveat=(
            "The Hunyuan weights are not licensed for use in the EU, the UK or South Korea. "
            "Check the licence before downloading."
        ),
        weights=(
            WeightSet("Hunyuan3D-2 shape (default route)", "tencent/Hunyuan3D-2",
                      int(5.0 * GB), REPO / "hunyuan_mlx" / "shape" / "weights" / "Hunyuan3D-2"),
            WeightSet("Hunyuan3D-2.1 paint (PBR), MLX port",
                      "zimengxiong/hunyuan3d-mlx-paint-large", int(8.7 * GB),
                      REPO / "hunyuan_mlx" / "paint" / "weights"),
        ),
    ),
    Backend(
        id="hunyuan-cuda",
        label="Hunyuan3D-2.1 (NVIDIA)",
        rank=2,
        best_for="Hunyuan on an NVIDIA card: Tencent's own shape and PBR paint, one run.",
        tradeoff=(
            "Linux only, and paint needs about 21 GB of GPU memory: a 24 GB card or bigger. "
            "Compiles its paint rasterizer for your card on setup."
        ),
        license_name="Tencent Hunyuan Community License (code + weights)",
        license_url="https://huggingface.co/tencent/Hunyuan3D-2.1",
        install="scripts/bootstrap_hunyuan_cuda.py",
        runs_on=(NVIDIA,),
        excludes=((NVIDIA, "windows"),),
        setup_minutes=30,
        build_probes=(REPO / "vendor" / "hunyuan-cuda" / ".i2l-build-complete",),
        caveat=(
            "The Hunyuan weights are not licensed for use in the EU, the UK or South Korea. "
            "Check the licence before downloading."
        ),
        weights=(
            WeightSet("Hunyuan3D-2.1 shape + VAE", "tencent/Hunyuan3D-2.1", int(8.03 * GB),
                      REPO / "vendor" / "hunyuan-cuda" / "models" / "tencent" / "Hunyuan3D-2.1"),
            WeightSet("Hunyuan3D-2.1 PBR paint", "tencent/Hunyuan3D-2.1", int(6.89 * GB),
                      HF_HUB_DIR / "models--tencent--Hunyuan3D-2.1"),
            WeightSet("DINOv2-giant image encoder", "facebook/dinov2-giant", int(4.55 * GB),
                      HF_HUB_DIR / "models--facebook--dinov2-giant",
                      note="The paint model reads the picture with it. Upstream fetches it "
                           "unannounced on the first run; here it is fetched at setup."),
            WeightSet("RealESRGAN x4plus upscaler",
                      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
                      "RealESRGAN_x4plus.pth",
                      int(0.067 * GB),
                      REPO / "vendor" / "hunyuan-cuda" / "ckpt" / "RealESRGAN_x4plus.pth"),
            WeightSet("BiRefNet-lite background remover", _matte.LITE_URL, _matte.LITE_BYTES,
                      _matte.model_file(_matte.LITE_MODEL),
                      note="Cuts out pictures without alpha. Shared with the other routes."),
        ),
    ),
    Backend(
        id="hunyuan-mlx",
        label="Hunyuan3D-MLX (dgrauet shape + Xiong paint)",
        rank=4,
        best_for="The older Hunyuan pairing: dgrauet's shape stage, Xiong's paint stage.",
        tradeoff=(
            "Two projects bolted together, and the shape half is a manual vendor clone. "
            "Prefer the Xiong full pipeline unless you specifically want this shape model."
        ),
        license_name="MIT (Xiong paint code); Tencent Hunyuan Community License (weights)",
        license_url="https://huggingface.co/tencent/Hunyuan3D-2.1",
        install="Manual: clone dgrauet's port into vendor/hunyuan-mlx, then uv sync",
        upstream=("Hunyuan3D-2.1", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"),
        nvidia_route="hunyuan-cuda",
        amd_blocked_reason=(
            "Hunyuan is not available on AMD in any form: this port is Apple Silicon, and "
            "Tencent's own needs NVIDIA-only compiled extensions."
        ),
        automated_setup=False,
        setup_minutes=40,
        build_probes=(REPO / "vendor" / "hunyuan-mlx" / ".venv" / "bin" / "python",
                      venv_python(REPO / "hunyuan_mlx" / "paint")),
        caveat=(
            "The Hunyuan weights are not licensed for use in the EU, the UK or South Korea. "
            "Check the licence before downloading."
        ),
        extra_steps=(
            "The shape half stays vendor-cloned on purpose: it is Tencent-licensed code, "
            "not just weights. See docs/hunyuan-mlx-recipes.md.",
        ),
        weights=(
            WeightSet("Hunyuan3D-2.1 shape, MLX port", "dgrauet/hunyuan3d-2.1-mlx",
                      int(13.0 * GB),
                      HF_HUB_DIR / "models--dgrauet--hunyuan3d-2.1-mlx"),
            WeightSet("Hunyuan3D-2.1 paint (PBR), MLX port",
                      "zimengxiong/hunyuan3d-mlx-paint-large", int(8.7 * GB),
                      REPO / "hunyuan_mlx" / "paint" / "weights",
                      note="The same paint weights the Xiong route uses. Downloading it "
                           "for one route installs it for both."),
        ),
    ),
    Backend(
        id="sf3d",
        label="Stable Fast 3D",
        rank=5,
        best_for="Fastest of the lot, and the smallest download. Good for a quick look.",
        tradeoff="Lowest fidelity here, and it bakes lighting into the texture.",
        license_name="Stability AI Community License (non-commercial under $1M revenue)",
        license_url="https://huggingface.co/stabilityai/stable-fast-3d",
        install="scripts/bootstrap_sf3d.py",
        # Not NVIDIA: SF3D's pins (old huggingface-hub, rembg) broke the lab's shared
        # environment there, and TRELLIS.2, Hunyuan3D-2.1 and Pixal3D beat it anyway.
        runs_on=(APPLE,),
        # Not the NVIDIA-only-extension story the other blocked routes use. SF3D is plain
        # PyTorch; it is Apple-only here because of those pins, not because of the card.
        amd_blocked_reason=(
            "Stable Fast 3D is Apple-only in this lab: its pinned huggingface-hub and rembg "
            "break the shared environment, so it was left out of the Linux routes rather "
            "than given its own 2 GB of packages to fight over them."
        ),
        setup_minutes=20,
        build_probes=(REPO / "vendor" / "stable-fast-3d" / "sf3d" / "system.py",),
        caveat=(
            "The SF3D weights are gated: accept Stability's licence on Hugging Face and "
            "sign in under Hugging Face sign-in at the top of this page before setting it up."
        ),
        gated_repo="stabilityai/stable-fast-3d",
        weights=(
            WeightSet("Stable Fast 3D", "stabilityai/stable-fast-3d", int(3.75 * GB),
                      HF_HUB_DIR / "models--stabilityai--stable-fast-3d"),
            WeightSet("DINOv2 image encoder", "facebook/dinov2-large", int(1.13 * GB),
                      HF_HUB_DIR / "models--facebook--dinov2-large",
                      note="SF3D reads the picture with it. Without this entry it was "
                           "fetched unannounced on the first run."),
        ),
    ),
    Backend(
        id="trellis",
        label="TRELLIS.2 (clean port)",
        rank=3,
        best_for="Highest fidelity, closest to the official demo.",
        tradeoff=(
            "Slowest, and its material model bleaches flat or vector-style illustrations. "
            "Prefer photographs or softly lit 3D-style references."
        ),
        license_name="MIT (code + weights); DINOv3 License (image encoder)",
        license_url="https://huggingface.co/microsoft/TRELLIS.2-4B",
        install="viewer",
        upstream=("TRELLIS.2", "https://github.com/microsoft/TRELLIS.2"),
        # Mac: the Metal port. Linux + NVIDIA: Microsoft's own code, built for CUDA.
        # Windows is not supported for TRELLIS.2 yet (the bootstrap refuses it).
        runs_on=(APPLE, NVIDIA),
        excludes=((NVIDIA, "windows"),),
        install_by_host={NVIDIA: "scripts/bootstrap_trellis_cuda.py"},
        build_probes_by_host={
            NVIDIA: (REPO / "vendor" / "trellis-cuda" / ".i2l-build-complete",),
        },
        caveat=(
            "Its DINOv3 image encoder is gated: request access to "
            "facebook/dinov3-vitl16-pretrain-lvd1689m on Hugging Face (Meta approves by "
            "hand) and sign in under Hugging Face sign-in at the top of this page, or the first run stops "
            "after the 14 GB download."
        ),
        gated_repo="facebook/dinov3-vitl16-pretrain-lvd1689m",
        setup_minutes=60,
        setup_fetches_weights=False,
        overrides_by_host={NVIDIA: {
            "label": "TRELLIS.2 (NVIDIA)",
            "tradeoff": ("Microsoft's own code, built for your card. Its material model "
                         "bleaches flat or vector-style illustrations. Prefer photographs "
                         "or softly lit 3D-style references."),
            "caveat": ("Its DINOv3 image encoder is gated: request access to "
                       "facebook/dinov3-vitl16-pretrain-lvd1689m on Hugging Face (Meta "
                       "approves by hand) and sign in under Hugging Face sign-in at the top of this page. Setup checks "
                       "access before downloading anything."),
            # The CUDA bootstrap fetches the weights once the build is done.
            "setup_fetches_weights": True,
        }},
        build_probes=(venv_python(REPO / "vendor" / "trellis-space-mac"),),
        weights=(
            # Measured on a real download, 2026-10-01: 15.1 GiB (14.0 undercounted by a
            # gigabyte, so the card read "16.4 GB of 15.3 GB").
            WeightSet("TRELLIS.2-4B", "microsoft/TRELLIS.2-4B", int(15.15 * GB),
                      HF_HUB_DIR / "models--microsoft--TRELLIS.2-4B"),
            WeightSet("TRELLIS image-large decoder", "microsoft/TRELLIS-image-large",
                      148 * 1024 ** 2,
                      HF_HUB_DIR / "models--microsoft--TRELLIS-image-large",
                      note="One decoder file TRELLIS.2's config pulls in on the first run."),
            WeightSet("DINOv3 image encoder", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                      int(1.1 * GB),
                      HF_HUB_DIR / "models--facebook--dinov3-vitl16-pretrain-lvd1689m"),
            WeightSet("TinyCLIP input advisor", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                      92 * 1024 ** 2,
                      HF_HUB_DIR / "models--wkcn--TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                      note="Advisory only. Generation works without it."),
            WeightSet("BiRefNet-lite background remover", _matte.LITE_URL, _matte.LITE_BYTES,
                      _matte.model_file(_matte.LITE_MODEL),
                      note="Cuts out pictures without alpha on NVIDIA. Fetched by the "
                           "NVIDIA setup; the Mac port wants pre-masked images."),
        ),
    ),
    Backend(
        id="qwen-image",
        label="Qwen-Image 2.1 (text to image)",
        kind="image",
        best_for="Makes the source image when you do not have one. Prompt in, picture out.",
        tradeoff="Runs on your machine, and the pictures you make are yours.",
        license_name="Qwen Research License",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
        install="Prebuilt stable-diffusion.cpp binary in vendor/sdcpp/",
        runs_on=(APPLE, NVIDIA, AMD),
        setup_minutes=15,
        build_probes=(_host.executable(REPO / "vendor" / "sdcpp", "sd-cli"),),
        weights=(
            WeightSet("Qwen-Image 2.1 diffusion model (Q8_0)",
                      "leejet/Qwen-Image-2.1-GGUF", int(7.69 * GB),
                      HF_HUB_DIR / "models--leejet--Qwen-Image-2.1-GGUF"),
            WeightSet("Qwen3-VL-8B text encoder (Q4_K_M)",
                      "Qwen/Qwen3-VL-8B-Instruct-GGUF", int(5.03 * GB),
                      HF_HUB_DIR / "models--Qwen--Qwen3-VL-8B-Instruct-GGUF",
                      note="Qwen-Image reads your prompt with a vision-language model, "
                           "which is why the text encoder is this large."),
            WeightSet("Qwen-Image 2.1 VAE", "Comfy-Org/Qwen-Image-2.1", int(0.68 * GB),
                      HF_HUB_DIR / "models--Comfy-Org--Qwen-Image-2.1",
                      note="Turns the generated latent back into pixels."),
        ),
    ),
    Backend(
        id="matte",
        label="Background remover (BiRefNet-lite)",
        kind="tool",
        best_for=("Cuts the subject out before every generator, keeping the thin and "
                  "light-coloured parts the built-in u2net loses."),
        tradeoff="Optional. Without it, runs fall back to u2net.",
        license_name="MIT",
        license_url="https://github.com/ZhengPeng7/BiRefNet",
        install="scripts/bootstrap_matte.py",
        runs_on=(APPLE, NVIDIA, AMD),
        setup_minutes=2,
        weights=(
            WeightSet("BiRefNet-lite", _matte.LITE_URL, _matte.LITE_BYTES,
                      _matte.model_file(_matte.LITE_MODEL),
                      note="One file, used by Pixal3D, TRELLIS and SF3D alike."),
        ),
    ),
    Backend(
        id="autorig",
        label="Auto-rig (SkinTokens)",
        kind="tool",
        best_for=("Gives a humanoid model a skeleton and skin weights, so the Rig tab can "
                  "play preset animations on it."),
        tradeoff="Humanoids in a T-pose only for now. Four-legged creatures are coming.",
        overrides_by_host={
            AMD: {"tradeoff": "Humanoids in a T-pose only for now, and attention comes "
                              "from PyTorch rather than flash-attn, which has no ROCm "
                              "wheel: a little slower and a little heavier on memory."},
        },
        license_name="MIT",
        license_url="https://github.com/VAST-AI-Research/SkinTokens/blob/main/LICENSE",
        install="scripts/bootstrap_autorig.py",
        runs_on=(APPLE, NVIDIA, AMD),
        setup_minutes=10,
        build_probes=(venv_python(REPO / "vendor" / "SkinTokens"),
                      REPO / "vendor" / "SkinTokens" / "demo.py"),
        upstream=("VAST-AI SkinTokens", "https://github.com/VAST-AI-Research/SkinTokens"),
        weights=(
            WeightSet("SkinTokens rigging + skinning checkpoints", "VAST-AI/SkinTokens",
                      1_131_603_979 + 487_311_745,
                      HF_HUB_DIR / "models--VAST-AI--SkinTokens",
                      note="Pinned to the revision we tested. Setup also fetches "
                           "Qwen3-0.6B's config and tokenizer (~16 MB, no weights)."),
        ),
    ),
)

BY_ID = {backend.id: backend for backend in CATALOG}

def is_shared(backend: "Backend", weight: "WeightSet") -> bool:
    """Whether another route also uses this file, so removing `backend` must keep it.

    The background remover is listed by every route that cuts pictures out; Pixal3D's
    Remove once offered to delete it from under TRELLIS and Hunyuan. Its own card (a
    tool) can still remove it: that is the explicit choice."""
    if backend.kind == "tool":
        return False
    return any(other.id != backend.id and any(w.path == weight.path for w in other.weights)
               for other in BY_ID.values())


# The Generate tab spells one route differently from the catalogue, and renaming either
# would break a saved setting or a download key for no gain. One alias costs a line; two
# half-synchronised id namespaces cost an afternoon, which is what they already cost once.
ALIASES = {"hunyuan-mlx-xiong": "hunyuan_xiong"}


def resolve(backend_id: str) -> Backend | None:
    """One way to look a route up, whichever spelling the caller happens to hold."""
    return BY_ID.get(ALIASES.get(backend_id, backend_id))


def readiness(backend_id: str, host: str | None = None) -> dict[str, Any] | None:
    """Is this route installed, in the shape the Generate tab's readiness strip reads.

    This is the catalogue's answer, derived from the same data the Setup & Status page
    shows, so a route is never installed according to one screen and unknown to the other.
    Backends with a live probe of their own layer detail on top of this (which of three
    Hunyuan checkpoints is present, how many of nine GGUF files); everything else, and
    every image route, is served from here alone.

    ``ready`` means "a run can start", not "everything is downloaded". TRELLIS fetches its
    weights lazily on first use, so gating generation on a full cache would refuse a
    working install.
    """
    backend = resolve(backend_id)
    if backend is None:
        return None
    described = backend.describe(host)
    weights = {
        w["source"]: {"label": w["label"], "present": w["present"], "human": w["human_present"]}
        for w in described["weights"]
    }
    missing = [w["label"] for w in described["weights"] if not w["present"]]
    supported = described["supported_here"]
    built = described["build_present"]
    hint = None
    if not supported:
        hint = described["platform_note"]
    elif not built:
        hint = f"{backend.label} is not installed — run: {backend.install_for(host)}"
    return {
        "schema_version": 1,
        "backend": backend.id,
        "build": {"present": built, "hint": hint},
        "weights": weights,
        "missing_weights": missing,
        "ready": bool(supported and built),
        "warning": ("first use will download missing weights" if missing and built
                    else None),
        "supported_here": supported,
        "platform_note": described["platform_note"],
    }


def catalog_status(host: str | None = None) -> dict[str, Any]:
    """The whole catalogue merged with what is actually on disk, for this machine.

    The host is reported alongside the backends because "nothing is installed" and "nothing
    can be installed here" look identical in a list of states, and only one of them is
    worth a download button.
    """
    host = host or host_platform()
    backends = [backend.describe(host) for backend in sorted(CATALOG, key=_rank_key)]
    ready = [b for b in backends if b["state"] == "ready"]
    runnable = [b for b in backends if b["supported_here"]]
    return {
        "schema_version": 1,
        # The onboarding screen exists for exactly this condition, so the server decides
        # it rather than leaving each client to re-derive the rule.
        "needs_onboarding": not ready,
        "ready_count": len(ready),
        "host": {
            "id": host,
            "label": host_label(host),
            "any_backend_runs_here": bool(runnable),
            "supported": sorted({PLATFORM_LABELS.get(p, p)
                                 for b in CATALOG for p in b.runs_on}),
        },
        "backends": backends,
        # The Setup page's tabs. Each machine family sees only the backends that run on it,
        # worded for it; the tab for this machine is the only one with buttons.
        "platforms": [{"id": p, "label": TAB_LABELS[p], "coming": p not in VIEW_PLATFORMS}
                      for p in TAB_LABELS],
        "views": {p: [b.describe(p) for b in sorted(CATALOG, key=_rank_key) if p in b.runs_on]
                  for p in VIEW_PLATFORMS},
    }


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _dir_state(path: Path) -> tuple[bool, int]:
    """Whether a weight directory exists, and how many bytes are in it.

    Size is what drives the progress readout during a download, so an unreadable file is
    skipped rather than raised: a partially written cache is the normal case here, not an
    error worth failing the whole status call for.
    """
    if path.is_file():
        # A single-file model (BiRefNet-lite) shares its folder with other models, so the
        # file is the unit, not the folder.
        return True, path.stat().st_size
    if not path.is_dir():
        return False, 0
    total = 0
    seen: set[tuple[int, int]] = set()
    for item in path.rglob("*"):
        try:
            # Count each real file once, by inode. The Hugging Face cache links
            # `snapshots/` at `blobs/`, so counting both doubles every byte (16 GB reads
            # as 30 GB). Newer huggingface_hub puts the blobs outside this folder
            # entirely, so links must be followed, or 13.4 GB reads as 120 B.
            if not item.is_file():  # follows links; a dangling one is not a file
                continue
            info = item.stat()
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += info.st_size
        except OSError:
            continue
    return total > 0, total


def _weights_state(weights: list[dict[str, Any]]) -> str:
    """ready, partial or missing, judged against expected size rather than mere existence.

    A directory that exists but holds a tenth of the bytes is an interrupted download, and
    calling that "ready" is how someone ends up debugging a backend that was never fully
    fetched. The 85% floor leaves room for the size constants above being approximate.
    """
    if not weights:
        return "ready"
    if all(w["bytes_present"] >= w["bytes_expected"] * 0.85 for w in weights):
        return "ready"
    if any(w["bytes_present"] > 0 for w in weights):
        return "partial"
    return "missing"


def _state(weights: list[dict[str, Any]], built: bool, setup_fetches: bool) -> str:
    """The backend's state, which is the build and the weights together.

    The subtlety is TRELLIS: its bootstrap installs the code and fetches nothing, so a
    built TRELLIS with no weights is *usable* -- the weights download on the first
    generation run. Reporting that as "missing" sent someone to a Set up button that
    re-ran a completed bootstrap.
    """
    if not built:
        return "missing"
    if not setup_fetches:
        return "ready"
    return _weights_state(weights)


def _action(weights: list[dict[str, Any]], built: bool, setup_fetches: bool,
            automated: bool = True) -> str:
    """What the button should offer: build, fetch, resume, nothing, or a command to run."""
    if not built:
        return "build" if automated else "manual"
    if not setup_fetches:
        return "none"
    state = _weights_state(weights)
    return {"ready": "none", "partial": "resume"}.get(state, "download")


def _percent(present: int, expected: int) -> int:
    if expected <= 0:
        return 100
    return max(0, min(100, round(present / expected * 100)))


def _rank_key(backend: Backend) -> tuple[int, str]:
    return (backend.rank if backend.rank is not None else 99, backend.label)
