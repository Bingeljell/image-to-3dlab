"""Whether the torch inside a venv can actually run on this machine's NVIDIA card.

Rig is SkinTokens, which runs on torch, and torch carries device code for a fixed list of
card architectures (`torch.cuda.get_arch_list()`, `sm_75 sm_80 sm_86 ...`). On a card older
than everything in that list the first kernel dies with

    CUDA error: no kernel image is available for execution on the device

which is true, useless, and arrives after the model has loaded. One comparison against the
card's compute capability answers the same question in milliseconds, so the studio can say
one plain sentence instead (Quadro P5000 = 6.1, 2026-11-13).

The decision is here rather than in the SkinTokens checkout: `vendor/` is git-ignored, and a
sentence users read is ours to write. The probe that asks the venv's own torch is the thin
part; everything else is a pure function a test can drive.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

#: The oldest architecture our NVIDIA installs carry device code for: an RTX 20-series. Also
#: what the wording promises, so a card below it is refused before any download.
OLDEST_SM = 75
OLDEST_CARD = "an NVIDIA RTX 20-series or newer"

#: One JSON line out of the venv's torch: the architectures it was built for, and this card.
PROBE_SOURCE = (
    "import json, torch\n"
    "facts = {}\n"
    "try:\n"
    "    facts['archs'] = list(torch.cuda.get_arch_list())\n"
    "    facts['capability'] = list(torch.cuda.get_device_capability(0)) "
    "if torch.cuda.is_available() else None\n"
    "    facts['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
    "except Exception as exc:\n"
    "    facts['error'] = str(exc)\n"
    "print(json.dumps(facts))\n"
)


def sm_number(entry: str) -> int | None:
    """`sm_75` -> 75, `sm_90a` -> 90, `sm_75+PTX` -> 75; anything else -> None."""
    name = entry.split("+", 1)[0]
    if not name.startswith("sm_"):
        return None
    digits = name[3:].rstrip("a")
    return int(digits) if digits.isdigit() else None


def torch_supports(archs: Sequence[str], capability: tuple[int, int]) -> bool:
    """Whether torch built for `archs` can run a card of `capability`, as `(major, minor)`.

    A card matches when torch carries its own code, or when torch carries PTX no newer
    than the card: the driver compiles PTX at load time for the card it runs on, which
    works forwards (PTX for sm_75 runs on sm_120) and never backwards (never on sm_61).
    """
    card = capability[0] * 10 + capability[1]
    for entry in archs:
        built = sm_number(entry)
        if built is None:
            continue
        if built == card:
            return True
        if "PTX" in entry.upper() and built <= card:
            return True
    return False


def too_old(card: str | None = None) -> str:
    """The sentence for a card this torch has no code for."""
    named = f" ({card})" if card else ""
    return (f"Your graphics card{named} is older than Rig supports: it needs {OLDEST_CARD}.")


def too_new(card: str | None = None) -> str:
    """The sentence for a card newer than this torch was built for."""
    named = f" ({card})" if card else ""
    return (f"Rig's graphics software is older than your graphics card{named}. Set up the "
            "auto-rig again from Setup & Status to install a newer one.")


def problem(archs: Sequence[str], capability: tuple[int, int] | None,
            card: str | None = None) -> str | None:
    """The plain sentence when this torch cannot run on this card, or None.

    None also when the facts are missing: a machine that cannot tell us is left to fail or
    work on its own rather than being refused on a guess.
    """
    if not archs or capability is None:
        return None
    if torch_supports(archs, capability):
        return None
    built = [number for number in (sm_number(entry) for entry in archs) if number is not None]
    if not built:
        return None    # no sm_ architectures at all: a ROCm or Metal build, not ours to judge
    card_number = capability[0] * 10 + capability[1]
    if card_number < min(built):
        return too_old(card)
    return too_new(card)


def setup_problem(capability: str | int | None, card: str | None = None) -> str | None:
    """The same sentence before an install, from what the driver says about the card.

    Setup knows the card's capability long before a venv exists, and the architectures our
    NVIDIA builds carry never go below `OLDEST_SM`, so a card below it is refused before the
    two gigabytes of packages are fetched. A card at or above it is left to the real check
    at Rig time, which asks the torch that was actually installed.
    """
    try:
        number = int(str(capability).strip())
    except (TypeError, ValueError):
        return None
    return None if number >= OLDEST_SM else too_old(card)


def card_name(run: Callable = subprocess.run) -> str | None:
    """This machine's graphics card, as the driver names it. None when it cannot be told."""
    from image_to_3dlab import host

    return host.gpu_name(run=run)


def probe(python: Path, run: Callable = subprocess.run,
          timeout: float = 180) -> dict | None:
    """Ask this venv's torch what it was built for and what card it is sitting on.

    None when the answer is unavailable for any reason: a missing venv, an old torch, a
    torch without CUDA. The caller treats None as "cannot tell", never as "unsupported".
    """
    if not Path(python).is_file():
        return None
    try:
        result = run([str(python), "-c", PROBE_SOURCE], capture_output=True, text=True,
                     timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in reversed((result.stdout or "").strip().splitlines()):
        try:
            facts = json.loads(line)
        except ValueError:
            continue
        return facts if isinstance(facts, dict) else None
    return None


def card_problem(python: Path, card: str | None = None,
                 run: Callable = subprocess.run) -> str | None:
    """Does the torch in this venv run on this card? The sentence when it cannot, else None."""
    facts = probe(python, run)
    if not facts:
        return None
    capability = facts.get("capability")
    capability = tuple(capability) if isinstance(capability, list) else None
    return problem(facts.get("archs") or [], capability, card or facts.get("device"))
