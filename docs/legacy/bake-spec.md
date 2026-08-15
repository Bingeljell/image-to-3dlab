# 3D → Sprite Bake Spec

The authoring contract for generating Workling and foe sprites from Blender. Everything
here is a **parameter**, not a baked-in authoring choice: the camera is a turntable and
the framing is derived, so changing the angle later is a re-render, not a re-author.

Companion docs: [Characters](characters.md) for the roster and the pose contract,
[Sprite prompts](sprite-prompts.md) for the interim image-model prompts this pipeline
eventually replaces.

## 0. Where the assets come from

**This spec was written against Blender-authored characters. Ours are not.** Every
creature here is generated from a single concept image by TRELLIS (or SF3D), which
changes what arrives and what has to happen before §1 is even true. The chain:

| Step | Tool | Why it is not optional |
| --- | --- | --- |
| 1. Generate | `pipeline.py --run-manifest` | Always via a manifest, so a bake is reproducible |
| 2. Judge | `blender_render_asset.py --culled` | Plain grey, backface-culled. A textured render hides every defect that matters |
| 3. **Solidify** | Blender Solidify, `offset = 0.0` | **See the warning below — skipping this ships holes** |
| 4. Scale to BU | §1 | Generated meshes arrive unit-normalised; they are all the same size |
| 5. Fit markers | `blender_joint_markers.py` | 31 markers as bounding-box fractions |
| 6. Build rig | `blender_build_rig.py` | Expect it to report heat-weighting failure — that is normal |
| 7. Weight | `blender_voxel_weights.py` | Bone heat **never** solves on a generated mesh |
| 8. Verify | pose a limb, measure the head | See `docs/rig-spec.md` §6 |

Full rig detail lives in **`docs/rig-spec.md`**, which is authoritative. This document
should reference it, never restate it.

> ### Solidify is mandatory, and Film → Transparent is why
>
> TRELLIS renders fine detail — fur, armour plates, carved relief — as **zero-thickness
> sheets**. A sheet is invisible from behind, so at a three-quarter angle you see
> straight through it. With **Film → Transparent ON (§3)** that does not read as a dark
> patch: it becomes **alpha**, and the sprite ships with holes punched through it.
>
> Measured hole perimeter, relative to the mesh diagonal, before → after Solidify:
> pangolin 97.82 → **0.00**, moss fox 126.58 → **0.00**, Monolith 44.78 → **0.00**.
> Costs roughly 4× the faces, which is irrelevant for a sprite bake.
>
> A sprite bake is *less* forgiving of this than a game engine, not more. Background:
> `docs/open-questions.md` §1d.

---

## 1. World scale

One number everything else derives from:

> **1.00 Blender Unit = the height of a standard Workling, standing in idle.**

**Height means the bounding-box height of the posed idle**, not nose-to-ground. The two
differ sharply on a hunched quadruped: Snag's nose sits *below* its shoulder hump, so
"nose-to-ground" would scale it wrong against an upright Workling.

Size differences between creatures are expressed by **scaling the model**, never by
moving the camera or changing the ortho scale.

### The rescale step

Generated assets arrive **normalised to a unit bounding box**, so a Mote and a Monolith
come out of the pipeline the same size. Authored height is therefore applied, not
inherited:

1. Import the solidified GLB and pose it to **idle**.
2. Measure the posed bounding-box height `h`.
3. Scale the object uniformly by `target_BU / h`.
4. Apply the scale, and drop the lowest vertex to `z = 0`.

Do this **before** fitting markers, so marker fractions and bone lengths are in final
units. Record the applied factor in the bake manifest — without it, "why is this one
5% short" is unanswerable.

| Creature | Height (BU) |
| --- | --- |
| Workling (all five families) | 1.00 |
| Mote | 0.35 |
| Flicker | 0.70 |
| Snag | 0.95 |
| Monolith (boss) | 1.80 |

These are art decisions, not measurements. Snag at 0.95 reads as barely larger than a
pet, which is worth a second look if it is meant to be a brute.

## 2. Camera

Never rotate the camera directly — Euler order will diverge across rigs. Use a turntable:

```
Empty (plain axes), named CAM_TARGET, at (0, 0, 0.5)   ← mid-body of a 1.0 BU creature
└── Camera (child of CAM_TARGET)
      local location  = (0, -6, 0)
      local rotation  = (90°, 0°, 0°)      ← set once, never touched again
      type            = Orthographic
      Orthographic Scale = 1.5
```

The **only** two knobs, both on `CAM_TARGET`:

| Knob | Blender field | Value |
| --- | --- | --- |
| Elevation — the 3/4 down-angle | `CAM_TARGET` rotation **X** | **−18°** |
| Azimuth — the 3/4 side-turn | `CAM_TARGET` rotation **Z** | **+35°** |

Orthographic, not perspective: a perspective lens makes the Strike lunge read subtly
larger than the idle, and you will fight that forever. Ortho also means the ortho scale
is a single global constant, which is what makes the framing contract in §4 automatic.

**Ortho Scale is derived from cell size, not global.** The invariant is
**px/BU = 341.3**, so `orthoScale = cellPx / 341.3` — 1.5 at the 512 cell, 3.0 at the
1024 boss cell (§4). Within one cell size it is constant for every pose, every family,
every foe: never re-frame a shot.

### Facing

Existing sheets face **left**. All three current anchors carry asymmetric accessories
(the moss-fox's bell, the pangolin's back key, the newt's tail ember), so a horizontal
mirror puts them on the wrong side. Bake both facings rather than flipping in code:

- `face-left` → `CAM_TARGET` Z = **+35°**
- `face-right` → `CAM_TARGET` Z = **−35°**

Same pose data, one extra render pass, no extra authoring.

## 3. Lighting

Three **Sun** lamps (not points — sun lamps light identically wherever in frame the pose
sits), all parented to `CAM_TARGET` so they follow the angle if it ever changes.

| Lamp | Azimuth (rel. camera) | Elevation | Strength | Role |
| --- | --- | --- | --- | --- |
| **Key** | −35° | 50° | 1.0 | The upper-left key light the existing sprites read as |
| **Fill** | +110° | 15° | 0.20 | Lifts the shadow side, keeps the silhouette readable |
| **Rim** | +160° | 60° | 0.60 | The "gentle rim light" that separates from a dark dungeon |

Render settings:

- **Film → Transparent: ON.** No ground plane, no baked shadow.
- **View transform: Standard.** Not Filmic/AgX — those desaturate, and the palettes are
  hand-picked in `assets/worklings-<family>-spritesheet.png`. Match the existing sheets.
  Note the trade: Standard clips highlights hard, so **emissive materials blow out to
  white** — the Monolith's glowing eye slot needs its emissive strength tuned against
  Standard, not against a viewport using AgX.
- **Colour: sRGB, straight alpha, 8-bit PNG.**

### Contact shadow

The sprite carries **no shadow**. The game draws a contact ellipse under the sprite at
runtime, so it can move and scale per encounter. This is what lets a 3/4 angle read as a
place while the sprite itself stays a clean transparent cutout.

If you want a shape-matched shadow later, render it as a **separate pass to its own PNG**
— never composite it into the cell.

## 4. Framing contract

Because the camera is fixed and ortho, all of this is automatic. It only breaks if you
**translate the object** between poses — so pose the bones, never the object.

Lunges, hops and the downed sprawl translate the **root bone**, with a budget of
**±0.25 BU horizontal / ±0.15 BU vertical**; the safe margin below is sized for exactly
that envelope.

> **This requires a root bone, which the rig does not yet have.** An earlier draft said
> to translate "the topmost bone of the spine chain". That does not work: moving a spine
> bone drags the torso while the feet stay planted, so the legs stretch. It is also why
> Snag's attack can only arch and lean — there is nothing to lift the body with. See
> `docs/rig-spec.md` §2a; the rig must gain the bone before any pose in §7 that leaves
> the ground is authorable.

At the 512px cell (§5), with Ortho Scale 1.5:

- **341 px per Blender Unit.** A standard 1.0 BU Workling is **341px tall in a 512px
  cell** — 67% of cell height.
- **Ground contact sits at y = 488** (24px above the cell bottom).
- **Root/hips centre at x = 256.**
- **Safe margin: 16px.** Nothing may enter the outer 16px of any cell.

Frame once for the **widest** pose in the set — Strike's forward lunge and Victory's
raised paw — and let the quiet poses sit in empty space. Do not tighten the frame on
idle; a sheet that re-frames per pose visibly breathes in game.

The Monolith at 1.8 BU renders 614px and overflows a 512 cell. **Bosses bake into a
1024px double cell** — same camera, same angle, but the **ortho scale must double to
3.0**.

That last part is not optional, and an earlier draft got it wrong by saying "same ortho
scale". Ortho scale sets how many Blender Units span the frame, so px/BU is
`cell ÷ orthoScale`. Hold the scale and double the pixels and you double px/BU, which
overflows the bigger cell too:

| cell | ortho scale | px/BU | Monolith @ 1.8 BU | |
| --- | --- | --- | --- | --- |
| 512 | 1.5 | 341 | 614px | overflows — hence the double cell |
| 1024 | **1.5** | 683 | **1229px** | **still overflows** |
| 1024 | **3.0** | **341** | **614px** | fits, with margin |

**The invariant is px/BU = 341, not the ortho scale.** Any cell size works provided
`orthoScale = cellPx / 341.3`. That is what makes a boss's size relative to a Workling
physically true rather than art-directed.

## 5. Output — the 1080p update

**This supersedes the previous 256px cell.** The dungeon stage bakes once at a ceiling
that serves a 1080p fullscreen stage (and downsamples cleanly to the 720p window), so the
old 256px cell is roughly half the resolution needed.

| | Old | **New** |
| --- | --- | --- |
| Cell | 256 × 256 | **512 × 512** |
| Sheet (4×5 grid) | 1024 × 1280 | **2048 × 2560** |
| Render size (per pose) | 1024² | **2048²** |
| Downsample | 4× Lanczos | **4× Lanczos** |

Render each pose at **2048², downscale to 512²** with Lanczos. The 4× supersample is what
produces the clean pixel-cluster read without dither noise.

The app no longer hardcodes the cell size — `WorklingSprite` derives it from the sheet's
own width (`sheet.width / 4`), so **1024-wide and 2048-wide sheets both work**. Families
can be re-baked one at a time; nothing has to land as a big-bang swap.

### Grid layout

Fixed 4-column × 5-row grid, matching `WorklingSpriteFrame` in
`Sources/Worklings/WildkinPetView.swift`:

| Row | Col 0 | Col 1 | Col 2 | Col 3 |
| --- | --- | --- | --- | --- |
| **0** | idle | idleBlink | walkContact | walkPassing |
| **1** | walkContactOpposite | walkPassingOpposite | happy | caredFor |
| **2** | hungry | sleepy | sad | wary |
| **3** | strike | hurt | lowHP | victory |
| **4** | downed | brace | signature | *(unused)* |

### Naming

Name renders with a **frame index from day one**, even though every pose is a single
still today:

```
wildkin_strike_face-left_0000.png
```

The whole point of moving to 3D is that Strike becomes a 6-frame loop later. The index
costs nothing now; retrofitting it across five families of assets costs a lot.

### Weight

Commit the **assembled sheets only**. Keep `.blend` files and per-pose intermediate
renders out of git (or under LFS) — the repo's history is already heavy, and a 2048×2560
RGBA sheet is roughly 4× the old one.

## 6. The rig — as built

**This section records the rig that exists, and supersedes the earlier draft in this
doc's history.** The approach changed in a way worth naming: motion is not transferred
between characters as posed data, it is **generated by script against fixed bone names**.
Per-character work is *fitting*, not posing.

### The contract

**25 bones, fixed names, fixed hierarchy.** The names are the contract — `blender_walk_cycle.py`
addresses bones by name, so a gait authored once **retargets to a new character by
re-fitting markers rather than by re-authoring motion**.

That inverts the earlier plan. Under a pose-asset library, every pose is authored data
that has to be applied and fixed up per rig. Here the motion lives in the script, keyed to
names, and a new character costs a marker pass. The consequence is the same as before but
sharper: **a rig that renames a bone silently opts out of every gait**, so names are not
a convention, they are the interface.

### Hierarchy and markers — see `docs/rig-spec.md`

**`docs/rig-spec.md` is authoritative for the rig.** It carries the 25 bone names with
their parenting (lifted from `SKELETON` in `blender_build_rig.py`, not retyped), the 31
marker names, the measured rotation-sign convention, the weighting procedure, and the
fitting checklist for a new character.

Restating any of that here would guarantee drift — an earlier draft of this section
already carried two "to fill in" placeholders and pointed at the wrong script for the
bone table. What belongs in *this* document is only what the bake needs:

| Fact the bake depends on | Value |
| --- | --- |
| Bone count, fixed for every creature | 25 (26 once the root bone lands) |
| Markers, the only per-character work | 31, as bounding-box fractions |
| Weighting | voxel proxy, always — bone heat never solves |
| Translation for lunges and hops | the root bone, §4 |

### Why the foot is two segments

`paw` is the **heel**; `toe` sits ahead of it.

A single `wrist → paw` bone pivots like a peg: lifting it **raises the toe instead of
dropping it**, and the foot reads as a stiff paddle in every gait and every planted pose.
Heel plus toe lets the foot **roll** — heel strike, roll through, toe-off — which is most
of what makes a walk cycle read as weight rather than as sliding.

This is the one place where an extra bone pair pays for itself several times over, and
it's worth protecting if the bone count is ever squeezed.

### What still holds from the earlier plan

- **Unit-normalised assets** (§1) — reinforced, not superseded: it is precisely what makes
  fractional markers portable.
- **The object never translates** (§4). Translation happens on the topmost bone of the
  spine chain, within the framing budget in §4.

### Superseded

The earlier draft specified 40 bones legged / 16 legless, a five-bone tail, no jaw bone,
face-on-shape-keys-only, and a shared pose-asset library. None of that describes the rig
that exists. The shape-key expression recipes are **parked pending a decision** — the
as-built rig has a `jaw` bone, so some expression is now bone-driven, and how much of the
face remains on shape keys is an open question rather than a settled one.

## 7. The poses

19 total: the 12 base poses (shipped) plus the 7 combat poses. Reads are the contract —
each must be distinguishable from every other at cell size.

### Base (12)

| Pose | Read |
| --- | --- |
| **idle** | Neutral standing, weight settled, alert but calm |
| **idleBlink** | Identical to idle, eyes closed — the only difference |
| **walkContact** / **walkPassing** | Two-frame walk cycle, lead leg forward |
| **walkContactOpposite** / **walkPassingOpposite** | The same cycle on the opposite legs |
| **happy** | Bright and up — ears up, tail up, open expression |
| **caredFor** | Softer than happy — contented, eyes half-closed, settled |
| **hungry** | Drooping, looking up/off expectantly, one paw raised |
| **sleepy** | Heavy-lidded, head low, mid-yawn or about to be |
| **sad** | Ears flat, head and tail down, closed posture |
| **wary** | Low crouch, head forward, alert and suspicious — tense, not scared |

### Combat (7)

| Pose | Read |
| --- | --- |
| **strike** | Mid-lunge, body stretched forward, front paw/claw landing, fierce |
| **hurt** | Recoiling back, head turned aside, off-balance, wincing |
| **lowHP** | Hunched, staggering, breathing hard — combat exhaustion, *not* sleep |
| **victory** | Chest up, one paw raised high, joyful and heroic — more than happy |
| **downed** | Collapsed on its side, limbs splayed, eyes shut — defeated, *not* asleep |
| **brace** | Defensive crouch behind a raised guard (paw/tail/scaled back), focused |
| **signature** | Charging a special: heroic stance, building energy, family-coloured aura |

The signature aura is family-coloured: Wildkin swirling green nature-glow, Elemental
flaring orange fire, Relicborn radiant cyan rune-light, Glitchkin unstable violet-white
signal static, Bloomglass pale refracted starlight.

Two pairs are the ones that go wrong most often and are worth checking side by side:
**sleepy vs downed** and **lowHP vs sad**.

## 8. Manifest

Commit `assets/bake-manifest.json` **before rendering anything**. It is what makes the
camera decision reversible — without it, "re-bake at 22°" is archaeology across five
`.blend` files.

```json
{
  "version": 1,
  "world": { "unit": "1.0 BU = standard Workling height" },
  "camera": { "type": "ortho", "elevationDeg": -18, "azimuthDeg": 35,
              "orthoScale": 1.5, "orthoScaleBossCell": 3.0, "targetHeightBU": 0.5 },
  "lighting": { "rigVersion": 1, "keyAzimuthDeg": -35, "keyElevationDeg": 50,
                "fillStrength": 0.2, "rimStrength": 0.6, "viewTransform": "Standard" },
  "render": { "renderPx": 2048, "cellPx": 512, "columns": 4, "rows": 5,
              "downsample": "lanczos", "alpha": "straight" },
  "framing": { "pxPerBU": 341.3, "groundContactY": 488, "safeMarginPx": 16,
               "note": "pxPerBU is the invariant; orthoScale = cellPx / pxPerBU" },
  "source": { "backend": "trellis2", "solidified": true, "solidifyThickness": 0.004,
              "scaleFactorApplied": null },
  "heightsBU": { "workling": 1.0, "mote": 0.35, "flicker": 0.7, "snag": 0.95,
                 "monolith": 1.8 },
  "facings": ["face-left", "face-right"]
}
```

## 9. Foe pose sheets

Foes do **not** use the Workling 4×5 pose sheet. They have three poses — `idle`,
`attack`, `hurt` — delivered as **one file per pose**, and each file is either a single
still or an animation sheet.

| | Single still | Animation sheet |
| --- | --- | --- |
| Layout | one cell | **4 columns, row-major**, filled left-to-right, top-to-bottom |
| Size | `cell × cell` | `(4 × cell) × (rows × cell)` |
| Example | 512×512 | 8 frames → 2048×1024 |

Cell size is read off the sheet's own width (`width / 4`), so any cell size works. **New
foe art uses the 512px cell** (§5). Everything in §2–§4 applies unchanged — same camera,
same lighting, same ortho scale, same ground-contact line — so a foe's size relative to a
Workling comes out physically true. Snag is **0.95 BU** (§1).

### Filenames

```
assets/foes/<base>-idle.png
assets/foes/<base>-attack.png
assets/foes/<base>-hurt.png
```

`<base>` is lowercase and matches the entry in `FoeSpriteAsset.resourceBase(for:)`.
Registered today: `mote` (display name "Dungeon Scamp"), `snag` ("Snag").

### Frame count is declared, not inferred

The frame count lives in `FoeSpriteAsset.frameCount(foe:pose:)`, not in the file. A 4×4
sheet is square and therefore indistinguishable from a single still, so inferring it from
the image would be a guess that silently breaks at 16 frames. Adding an animated pose is
one line there.

### Playback

- **12 fps** (`FoeSprite.frameDuration`). Slow enough to read as hand-animated next to
  the pixel art, fast enough that an 8-frame swing lands inside a combat beat.
- **A pose plays once and holds its last frame.** It restarts from frame 0 every time the
  foe enters that pose, rather than joining a free-running loop mid-swing.
- **`idle` is the exception — it loops.**

> The consequence worth designing around: **the last frame of an animated pose is what
> the foe sits on for the rest of the beat.** Author it as a settle that reads held, not
> as the end of a follow-through.

### Recommended attack breakdown — 8 frames

8 frames at 12fps is 0.67s, which fits comfortably inside a combat beat.

| Frames | Beat | Notes |
| --- | --- | --- |
| 0–1 | **Anticipation** | Wind back *away* from the pet. The clearest read in the whole swing; don't rush it |
| 2 | **Contact** | The extreme. One frame only — impact reads as a snap, not a hold |
| 3–4 | **Follow-through** | Overshoot past the contact pose, then start back |
| 5–7 | **Recovery** | Settle toward neutral; **frame 7 should read as a held pose** |

For Snag specifically — a knot of tangled thorny threads that *grabs* rather than strikes
— the anticipation is the tendrils gathering/coiling inward and the contact is them
lashing outward. It has a Snare (an Agility debuff), so the silhouette wants to read as
*seizing*, not punching.

> **The Snag asset currently in `output/` does not match this brief.** It is a smooth
> stone quadruped brute, generated as a control for the mesh-integrity benchmark rather
> than to the design here. Its rig and walk/attack clips are real and reusable, but the
> creature is wrong: a thorn-knot that seizes has no legs to walk on and a completely
> different silhouette. Regenerate to the brief before baking. The 0.95 BU height in §1
> was set for the brute and should be revisited for the thorn-knot.

### Dropping a new foe in

1. Bake the three PNGs to `assets/foes/`.
2. Add them to `Package.swift` `resources:` — `.copy("../../assets/foes/snag-idle.png")`
   and the other two. **A `.copy` of a missing file fails the build**, which is why the
   entries are not pre-added for art that doesn't exist yet.
3. Register the base name in `FoeSpriteAsset.resourceBase(for:)` if it isn't already.
4. Declare any multi-frame pose in `FoeSpriteAsset.frameCount(foe:pose:)`.

Until step 1 lands, `hasArt` is false and the foe renders the existing placeholder — a
registered-but-unbaked foe degrades rather than breaks.

## Open

- **The angle itself.** −18° / +35° is the working default, inside the recommended
  15–25° band. It is deliberately not locked; the turntable rig and this manifest are
  what keep it a cheap parameter rather than an expensive commitment. Bake one character
  at 0° / −18° / −35° elevation and compare three stills before locking.
- **Animation.** Everything is single-frame today. The frame index in the naming
  convention is the only concession made in advance.
