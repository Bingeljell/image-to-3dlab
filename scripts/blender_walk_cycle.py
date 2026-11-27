#!/usr/bin/env python3
"""Author a looping gait on a quadruped armature in the live Blender scene.

Run (live): python scripts/blender_walk_cycle.py --help

Two gaits, and they differ in *phase*, not merely in speed:

* **trot** (the default) — a two-beat **diagonal** gait. Front-left and back-right
  swing together, then front-right and back-left. The body vaults over each diagonal
  pair in turn, which produces the trot's characteristic bounce.
* **walk** — a four-beat **lateral** gait. Each foot lands a quarter-cycle after the
  last, so at least two feet are always down. Slower and flatter.

Each leg superimposes three motions:

* **swing** — the upper bone (upper arm or thigh) rocks forward and back, a cosine;
* **middle fold** — the elbow or knee folds while the foot is airborne, a half-wave
  rectified sine, shaped so it eases rather than snapping at the contact frames. A leg
  that bends while bearing weight looks broken, hence the clamp to the airborne half;
* **lower fold** — the wrist or ankle, folding slightly *later* than the joint above
  it. The joint nearest the ground is the last to leave it and the first to reach for
  it, and splitting the fold across both joints is what stops the lower limb hinging
  up as one rigid piece.

Note the front leg's visible mid-leg bend is a **wrist**, not a knee, and it carries
most of that leg's fold — leaving it stiff is what makes a front leg read as a stick.

**Fold and swing control different things and must not be confused.** Measured on this
rig: swing alone produces ~0.20 of paw travel and fold alone ~0.21, so *travel is
dominated by the swing arc*. Cutting fold to reduce travel therefore does not reduce
travel — it only removes the bending, leaving stiff legs that rise and fall, which reads
as swimming. Use **swing** to control how far a paw travels and **fold** to control how
much the leg bends.

`--front-crouch` exists because the front legs are counter-rotated to stay vertical under
a dropped chest, so they hang at full extension while the back legs keep a permanent
fold. Without it the front paws sink below the back ones.

The body adds a vertical bob at twice stride frequency plus a small counter-sway on
spine, tail and head. Too much lateral sway reads as a waddle.

Frame 1 and frame `frames + 1` are authored identically and the scene end is set to
`frames`, so playback loops seamlessly.

Everything is a flag; re-running replaces the animation cleanly, so tune by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animation_cli import read_name_map
from blender_joint_markers import send
from blender_worker_source import build_worker_code


def build_code(arguments: dict[str, object]) -> str:
    """Bundle gait controls and optional bone and marker name maps."""
    return build_worker_code(
        Path(__file__).resolve().with_name("blender_walk_cycle_worker.py"), arguments
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--rig-name", help="armature object to animate")
    parser.add_argument(
        "--bone-map",
        dest="bone_names",
        type=read_name_map,
        help="JSON file mapping default bone names to your rig's bone names",
    )
    parser.add_argument(
        "--marker-map",
        dest="marker_names",
        type=read_name_map,
        help="JSON file mapping paw roles to marker object names",
    )
    parser.add_argument("--mesh-name", help="mesh object to use for paw planting")
    parser.add_argument(
        "--frames", type=int, default=20, help="length of one full stride"
    )
    parser.add_argument("--swing-front", type=float, default=24.0)
    parser.add_argument("--swing-back", type=float, default=20.0)
    parser.add_argument("--bend-front", type=float, default=34.0)
    parser.add_argument("--bend-back", type=float, default=38.0)
    parser.add_argument(
        "--wrist",
        type=float,
        default=40.0,
        help="Front-leg wrist fold. The visible mid-leg bend on a front leg is a "
        "wrist, and it carries most of the fold — a stiff wrist reads as a stick",
    )
    parser.add_argument(
        "--ankle", type=float, default=22.0, help="Back-leg ankle (hock) fold"
    )
    parser.add_argument(
        "--gait",
        choices=("trot", "walk"),
        default="trot",
        help="trot is a two-beat diagonal gait; walk is a four-beat lateral gait",
    )
    parser.add_argument(
        "--base-bend",
        type=float,
        default=8.0,
        help="Standing bend held all cycle; a real limb is never locked straight",
    )
    parser.add_argument("--bob", type=float, default=0.016)
    parser.add_argument(
        "--sway",
        type=float,
        default=2.2,
        help="Lateral body roll. Too much reads as a waddle rather than a walk",
    )
    parser.add_argument(
        "--head-yaw",
        type=float,
        default=-32.0,
        help="Static counter-yaw on the neck, to straighten a head the subject was "
        "generated with turned. Negative turns toward the character's right",
    )
    parser.add_argument(
        "--head-pitch",
        type=float,
        default=0.0,
        help="Drop the head forward and down. A moving quadruped reaches ahead and low; "
        "an upright head reads as a standing animal with moving legs",
    )
    parser.add_argument(
        "--head-level",
        type=float,
        default=0.0,
        help="Counter-rotation on the head bone so the muzzle levels out instead of "
        "aiming at the floor once the neck is pitched down",
    )
    parser.add_argument(
        "--chest-drop",
        type=float,
        default=0.0,
        help="Pitch the chest down. This is what lowers the shoulders — and the neck "
        "and head ride down with it. Rotating the neck alone only pivots the muzzle",
    )
    parser.add_argument(
        "--shoulder-tuck",
        type=float,
        default=0.0,
        help="Static offset pulling the front legs back under the body instead of "
        "splaying them out ahead",
    )
    parser.add_argument(
        "--hind-lead-l",
        type=float,
        default=0.0,
        help="Static forward bias on the LEFT hind leg's swing. swing-back alone is "
        "symmetric — more forward reach always buys an equally bigger backward "
        "kick. This shifts that side's swing range forward without changing its "
        "span, so you can keep the forward peak and shrink the rearward extreme. "
        "Split left/right because hand-placed markers are rarely symmetric",
    )
    parser.add_argument(
        "--hind-lead-r",
        type=float,
        default=0.0,
        help="Same as --hind-lead-l, for the RIGHT hind leg.",
    )
    parser.add_argument(
        "--front-crouch",
        type=float,
        default=0.0,
        help="Static fold on the front legs so their paws sit level with the back "
        "paws. Without it the front legs hang at full extension and sink below",
    )
    parser.add_argument(
        "--toe",
        type=float,
        default=20.0,
        help="Toe flex during push-off. The foot is two segments so it can roll: heel "
        "plants, foot flattens, toes push off",
    )
    parser.add_argument(
        "--front-fold-sign",
        type=float,
        default=-1.0,
        help="Which way the front elbow/wrist close. Depends on rest geometry: with "
        "the elbow set behind the shoulder, negative folds and positive "
        "straightens. Flip if the front legs extend instead of bending",
    )
    parser.add_argument(
        "--crouch",
        type=float,
        default=0.0,
        help="Semi-crouch: static fold at elbow/wrist and knee/ankle so the chest and "
        "head come down while the paws stay planted. The front gets more because "
        "its shoulder/elbow/wrist markers sit nearly collinear, so it starts "
        "straight while the back leg has ~44 degrees of built-in fold",
    )
    parser.add_argument(
        "--no-auto-plant",
        dest="auto_plant",
        action="store_false",
        help="Disable re-planting the paws on the rest floor after folding the legs",
    )
    parser.add_argument(
        "--body-drop",
        type=float,
        default=0.0,
        help="Lower the whole rig, for a slightly hunched travelling posture",
    )
    args = parser.parse_args()

    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in ("host", "port") and value is not None
    }
    code = build_code(arguments)
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
