"""Create's rules for agents: which steps run, and the prompt that is sent.

A Python twin of `viewer/studio/plan.js`, so an agent's request runs exactly the steps the
studio's Create sheet would, with the same wording added. `tests/test_agent_plan.py` runs
both files on every combination and fails the moment they disagree, so a change to the
studio's rules can't quietly leave agents behind.
"""

from __future__ import annotations

import re
from typing import Any

HAVES = ("idea", "picture", "model")
WANTS = ("picture", "prop", "set", "character")
# What each starting point can lead to; a string is the reason it can't.
ALLOWED: dict[str, dict[str, bool | str]] = {
    "idea": {"picture": True, "prop": True, "set": True, "character": True},
    "picture": {"picture": "You already have the picture.", "prop": True, "set": True, "character": True},
    "model": {"picture": "Start from an idea to make a picture.", "prop": True,
              "set": "Upload the 3×3 sheet as a picture instead.", "character": True},
}
FULL = {
    "picture": ["picture"],
    "prop": ["picture", "model", "finished"],
    "set": ["picture", "model", "split"],
    "character": ["picture", "model", "finished", "rigged", "animated"],
}


def plan_for(have: str, want: str, *, finish_upload: bool = False, front_back: bool = False) -> list[str]:
    """The steps to run, in order, before any "stop after" (plan.js `planFor`)."""
    skip = {"picture": ["picture"], "model": ["picture", "model"]}.get(have, [])
    steps = [s for s in FULL.get(want, []) if s not in skip]
    if have == "model" and not finish_upload:
        steps = [s for s in steps if s != "finished"]
    if front_back and want != "set" and "model" in steps:
        steps.insert(steps.index("model"), "back")
    return steps


def stop_after(steps: list[str], last: str | None) -> list[str]:
    """Cut a plan after a step, or keep it whole (plan.js `stopAfter`)."""
    return steps[: steps.index(last) + 1] if last in steps else steps


def compose_prompt(text: str, goal: str, recipes: dict[str, Any] | None, use_recipe: bool = True) -> str:
    """The description plus the goal's proven wording (plan.js `composePrompt`)."""
    base = re.sub(r"[\s,.]+$", "", str(text or "").strip())
    wording = ((recipes or {}).get("goals", {}).get(goal) or {}).get("wording") if use_recipe else ""
    return f"{base}, {wording}" if wording else base


def refusal(have: str, want: str) -> str | None:
    """Why this combination can't run, or None when it can."""
    if have not in HAVES:
        return f"have must be one of {', '.join(HAVES)}"
    if want not in WANTS:
        return f"want must be one of {', '.join(WANTS)}"
    allowed = ALLOWED[have][want]
    return allowed if isinstance(allowed, str) else None
