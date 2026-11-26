"""Read name maps for the animation commands."""

import argparse
import json
from pathlib import Path


def read_name_map(value: str) -> dict[str, str]:
    """Read a JSON object that maps default control names to rig-specific names."""
    try:
        names = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Could not read name map {value!r}: {exc}"
        ) from exc
    if not isinstance(names, dict) or any(
        not role or not isinstance(name, str) or not name
        for role, name in names.items()
    ):
        raise argparse.ArgumentTypeError(
            "Name map must be a JSON object of nonempty name pairs."
        )
    return names
