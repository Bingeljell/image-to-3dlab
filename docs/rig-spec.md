# Quadruped rig spec

One skeleton for every quadruped character in the project. Fixed bone names, fixed
hierarchy, per-character joint positions. Any character rigged to this spec can share
animation, and a gait authored once retargets by re-fitting markers rather than by
re-authoring motion.

Verified on: **moss fox** (foliage, thin, 16,467 boundary edges) and **Snag** (stone
brute, chunky, 505 boundary edges). Those two bracket the range the pipeline produces.

Consumed by `docs/bake-spec.md`, which is authoritative for camera, framing and sheet
layout. This document is authoritative for the rig; neither should restate the other.

---

## 1. Coordinate conventions

Positions are given in **Blender space, after the glTF importer's Y-up → Z-up
conversion**, as fractions of the mesh's bounding box:

| axis | meaning |
|---|---|
| **X** | left (−) / right (+) |
| **Y** | **front is minimum Y**, back is maximum |
| **Z** | up |

Generated assets arrive normalised to roughly a unit bounding box, so absolute numbers
are meaningless across characters — always fit markers as fractions of *that*
character's box.

---

## 2. Bone hierarchy — 25 bones

```
spine_01                 (pelvis → spine_mid)          root
├── spine_02             (spine_mid → chest)
│   ├── neck             (chest → neck_base)
│   │   └── head         (neck_base → head)
│   │       ├── jaw      (head → jaw)              detached
│   │       ├── ear_L    extrapolated, see §4
│   │       └── ear_R
│   ├── frontL_upperarm  (shoulder → elbow)        detached
│   │   └── frontL_forearm  (elbow → wrist)
│   │       └── frontL_paw  (wrist → paw/heel)
│   │           └── frontL_toe  (paw → toe)
│   └── frontR_upperarm  … same chain
├── backL_thigh          (hip → knee)              detached
│   └── backL_shin       (knee → ankle)
│       └── backL_paw    (ankle → paw/heel)
│           └── backL_toe   (paw → toe)
├── backR_thigh          … same chain
└── tail_01              (tail_base → tail_mid)    detached
    └── tail_02          (tail_mid → tail_tip)
```

"Detached" means parented but **not connected** — the bone's head does not have to sit
on its parent's tail. Limb roots and the tail need that freedom because a shoulder sits
inboard of the spine, not on it.

**Bone names are the contract.** `blender_walk_cycle.py` and every other animation script
address bones by these exact names. Do not rename per character.

---

## 2a. `root` — specified, not yet built

**Status: agreed, not implemented.** Nothing in `output/` has it, and adding it changes
the bone count from 25 to 26.

Every bone above sits *inside* the body, so they can bend the character but cannot
**move** it. Rotating the spine tilts the chest; nothing lifts the creature off the
ground. That is a real limitation, not a theoretical one:

- Snag's slam attack can only arch and lean. A genuine rear needs the body to rise, and
  there is nothing to rise with.
- A walk cycle animates in place; it cannot travel.
- `docs/bake-spec.md` §4 budgets ±0.25 BU horizontal / ±0.15 BU vertical for lunges,
  hops and the downed sprawl. An earlier draft assigned that translation to "the topmost
  bone of the spine chain", which stretches the legs — the feet stay planted while the
  torso slides away from them.

### The bone

```
root                     at ground level, between the feet, pointing forward
├── spine_01
└── (everything else, unchanged)
```

| property | value |
| --- | --- |
| head | `(0, 0, 0)` in character space — floor level, centred between the paws |
| tail | `(0, -0.25, 0)` — points forward, so its local axes are legible |
| parent | none. It becomes the armature root; `spine_01` parents to it, detached |
| weights | **none.** It carries no vertices; it exists only to move everything below |

`spine_01` stops being the root and becomes root's child. No other parenting changes, no
marker changes — root is derived from the mesh's floor plane and centre, not fitted.

### Consequences

- **Animation scripts must not touch `root`** unless they intend the whole character to
  move. A gait that rotates `root` will yaw the creature rather than turn its body.
- **Translate `root`, rotate everything else.** That keeps the framing contract in
  `bake-spec.md` §4 checkable: read root's translation, compare against the budget.
- Existing clips stay valid — they drive bones that keep their names and parents.

---

## 3. Marker set — 31 markers

The only per-character work. Names are fixed; positions are fitted.

| group | markers |
|---|---|
| spine | `pelvis`, `spine_mid`, `chest`, `neck_base`, `head`, `jaw` |
| tail | `tail_base`, `tail_mid`, `tail_tip` |
| ears | `ear_L`, `ear_R` (base only) |
| front legs ×2 | `front{L,R}_shoulder`, `_elbow`, `_wrist`, `_paw`, `_toe` |
| back legs ×2 | `back{L,R}_hip`, `_knee`, `_ankle`, `_paw`, `_toe` |

**Every marker must exist**, even where the anatomy doesn't. Snag has no tail: its three
tail markers are parked as a short stub inside the rump, so the bones exist, sit within
the mesh, and pick up negligible weight. Omitting them fails the build.

### The foot is two segments, not one

`paw` is the **heel**; `toe` is ahead of it. A single wrist→paw bone pivots the whole
foot like a peg, so lifting it raises the toe instead of dropping it. With a heel joint
and a toe segment the foot rolls correctly: heel plants, foot flattens, toes push off.

Fit both from the foot's actual ground-level extent — heel at ~72% back along the foot,
toe at ~12%.

---

## 4. Ears

Ears have a **base marker only**. The bone is extrapolated outward from the head along
the head→ear direction, so it lies along the ear and pivots at its base, which is how an
ear moves. `--ear-length` controls the extrapolation: too long and the tip pushes outside
the mesh. Fox ~0.12, Snag ~0.06 (nubs, not blades).

---

## 5. Weighting: always the voxel proxy

**Bone-heat weighting does not work on generated meshes. Use
`scripts/blender_voxel_weights.py`, not the direct bind.**

Heat weighting diffuses influence across the mesh *surface*, which is what stops a leg
bone grabbing a nearby ear. It needs a clean manifold to solve on. Measured:

| mesh | boundary edges | heat weighting result |
|---|---|---|
| moss fox | 16,467 | fails — 21 empty groups |
| Snag | 505 | **fails — 25 empty groups, all of them** |

Snag is 33× cleaner than the fox and heat still cannot solve it. This is not a
fox-specific workaround; it is the default path for anything TRELLIS produces.

The proxy route: voxel-remesh a copy into a watertight blob → heat-weight the proxy,
which solves cleanly → transfer weights back by nearest-surface interpolation → bind the
real mesh. The proxy is throwaway and its ugliness does not matter; only the transfer
needs to be accurate. Result on Snag: 25 groups, **0 empty, 0 unweighted vertices**.

---

## 6. Verification: pose a leg, measure the head

`blender_build_rig.py` reports `ENVELOPE-LIKE BLEED` when a leg's influenced vertices
reach head height. **That heuristic assumes the head is the highest point on the body,
and it produces false positives on any hunched character.**

Snag tripped it on all four limbs. The head marker sits at z 0.157 while its shoulder
hump legitimately reaches 0.27 — the check was measuring anatomy, not bleed.

The decisive test, which should be preferred whenever the heuristic fires:

> Rotate one limb root by ~55°. Measure how far the centroid of the vertices weighted to
> `head` + `jaw` moves. On a correct rig it does not move.

Snag: **0.0003 units on a 0.788-tall mesh — 0.04%.** Correctly weighted.

---

## 7. Fitting a new character

1. Measure the mesh: bounding box, and the leg columns (slice at ~12% height and cluster
   the occupied cells in the X/Y plane — a clean quadruped gives four clusters).
2. Write the 31 markers as fractions of that box. Limb roots pull slightly inboard
   (~85% of the leg's X) so they sit inside the body.
3. `blender_joint_markers.py load --file <character>_joints.json`
4. `blender_build_rig.py --ear-length <fitted>` — expect it to report heat failure.
5. `blender_voxel_weights.py`
6. Pose-test per §6.
7. `blender_walk_cycle.py` — gaits are authored against the fixed bone names, so they
   transfer without re-authoring.

**Always save markers to JSON.** They live nowhere but the Blender scene until written
out, and several scripts here clear the scene before working. Losing them is
unrecoverable, autosave included.

---

## 8. Export naming

Exported GLBs use generic names so a character is identified by its own name, not by
whichever character the rig was first authored on:

| thing | name |
|---|---|
| armature / scene root | `Rig` |
| animation clip | the gait — `Walk`, `Trot`, … |
| mesh object | the character — `Snag` |

The scripts default to `FoxRig` / `FoxRigAction`, which is wrong on anything that is not
the fox. Rename before export.

Export with `export_animations`, `export_skins`, `export_yup`. Verify by reading the GLB
header back — one mesh node, one skin, the expected joint count, one animation — and by
rendering the animation **from the exported file** rather than from the live scene, which
is the only thing that proves the export round-tripped.

---

## 9. Known gaps

- **`root` is specified but not built** (§2a). Until it lands, no pose that leaves the
  ground is authorable, and `bake-spec.md` §4's translation budget has nothing to apply
  to.
- **Solidified meshes are untested under deformation.** Assets fixed per
  `docs/open-questions.md` §1d carry doubled shells; whether plates hold together when
  bones move is unverified, and is the next thing to check.
- **No IK.** Feet are posed by forward kinematics, so foot planting is authored rather
  than solved. Fine for looping gaits, weak for uneven ground.
- **Bipeds are out of scope.** The marsupial Flicker needs a different hierarchy.
- **No retarget validation** — that a fox gait *looks* right on Snag's proportions is
  assumed, not measured.
