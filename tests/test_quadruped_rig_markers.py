"""The marker table and the skeleton table must agree, and left must mean left.

Two defects found by eye on the Tempest Ram, both cheap to catch here:

* Every `*_L` marker sat at x<0.5 and every `*_R` at x>0.5 -- mirrored, so the rig's
  left leg was the character's right. Only visible once something is animated.
* `blender_build_rig.py`'s SKELETON referenced four `*_toe` markers that
  `blender_joint_markers.py` never defined, crashing the build the first time the
  pipeline ran end to end on a non-fox character.

Both tables are plain module-level data with no bpy dependency, so this costs
milliseconds against a rig build that costs a Blender session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from blender_build_rig import EAR_BONES, SKELETON  # noqa: E402
from blender_joint_markers import QUADRUPED_MARKERS  # noqa: E402


def test_every_bone_endpoint_has_a_marker():
    """The exact failure that crashed the first non-fox rig build."""
    needed = {m for _, head, tail, _, _ in SKELETON for m in (head, tail)}
    needed |= {m for pair in EAR_BONES for m in pair}
    missing = sorted(needed - set(QUADRUPED_MARKERS))
    assert not missing, f"SKELETON references markers that do not exist: {missing}"


def test_no_marker_is_placed_but_never_used():
    """A marker nothing binds to is a joint the user hand-placed for nothing."""
    used = {m for _, head, tail, _, _ in SKELETON for m in (head, tail)}
    used |= {m for pair in EAR_BONES for m in pair}
    assert not sorted(set(QUADRUPED_MARKERS) - used)


def test_left_markers_sit_on_the_characters_left():
    """+X is the character's own left (the subject's front is at minimum Y).

    On screen that puts L markers on the viewer's right whenever the subject faces the
    camera -- correct, and the opposite of what looks right, which is how they came to
    be swapped in the first place.
    """
    for name, (x, _, _) in QUADRUPED_MARKERS.items():
        if name.endswith("_L") or name.startswith(("frontL", "backL")):
            assert x > 0.5, f"{name} at x={x} is on the character's right"
        elif name.endswith("_R") or name.startswith(("frontR", "backR")):
            assert x < 0.5, f"{name} at x={x} is on the character's left"


def test_left_and_right_markers_mirror_each_other():
    """Asymmetry here is a typo, not a stylistic choice -- the defaults are a template
    the user nudges, and a lopsided starting rig hides its own placement errors."""
    for name, (x, y, z) in QUADRUPED_MARKERS.items():
        twin = None
        if name.endswith("_L"):
            twin = name[:-2] + "_R"
        elif name.startswith("frontL") or name.startswith("backL"):
            twin = name.replace("L_", "R_", 1)
        if twin is None:
            continue
        assert twin in QUADRUPED_MARKERS, f"{name} has no counterpart {twin}"
        tx, ty, tz = QUADRUPED_MARKERS[twin]
        assert x + tx == 1.0, f"{name}/{twin} are not mirrored in x ({x} vs {tx})"
        assert (y, z) == (ty, tz), f"{name}/{twin} differ off the mirror axis"


def test_centreline_markers_sit_on_the_centreline():
    """Spine, head and tail must not drift off-axis, or the whole rig leans."""
    for name, (x, _, _) in QUADRUPED_MARKERS.items():
        if name.endswith(("_L", "_R")) or name.startswith(("frontL", "frontR", "backL", "backR")):
            continue
        assert x == 0.5, f"{name} is off the centreline at x={x}"


def test_every_marker_is_inside_the_bounding_box():
    """These are fractions of the mesh bounding box; outside it is off the model."""
    for name, position in QUADRUPED_MARKERS.items():
        assert len(position) == 3, name
        for axis, value in zip("xyz", position):
            assert 0.0 <= value <= 1.0, f"{name}.{axis} = {value} is outside the mesh"


def test_bone_names_are_unique():
    names = [bone for bone, _, _, _, _ in SKELETON]
    assert len(names) == len(set(names)), "a duplicate bone silently overwrites its twin"


def test_every_parent_is_defined_before_its_child():
    """Blender builds bones in order; a forward reference is an unparented limb."""
    seen: set[str] = set()
    for bone, _, _, parent, _ in SKELETON:
        if parent is not None:
            assert parent in seen, f"{bone} is parented to {parent}, defined later"
        seen.add(bone)


def test_the_root_is_the_only_unparented_bone():
    roots = [bone for bone, _, _, parent, _ in SKELETON if parent is None]
    assert roots == ["spine_01"], f"expected one root, got {roots}"


def test_the_foot_is_two_segments_so_it_can_roll():
    """A single wrist->paw bone pivots the foot as a peg: lifting it raises the toe
    instead of dropping it. Every leg needs a separate toe segment."""
    bones = {bone: (head, tail, parent) for bone, head, tail, parent, _ in SKELETON}
    for side in ("L", "R"):
        for limb in (f"front{side}", f"back{side}"):
            assert f"{limb}_toe" in bones, f"{limb} has no toe segment"
            _, _, parent = bones[f"{limb}_toe"]
            assert parent == f"{limb}_paw"


def test_the_skeleton_is_left_right_symmetric():
    bones = {bone: (head, tail, parent, conn) for bone, head, tail, parent, conn in SKELETON}
    for bone in list(bones):
        if "L_" not in bone and not bone.endswith("_L"):
            continue
        twin = bone.replace("L_", "R_", 1) if "L_" in bone else bone[:-2] + "_R"
        assert twin in bones, f"{bone} has no counterpart {twin}"
