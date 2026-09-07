# Clockwork Pangolin recipes

Requires the pangolin Rigify scene (`rig`, `geometry_0`) with the exact versioned
actions referenced in each script. Read the [safety guide](README.md) first.

| Script | Runtime / effects | Purpose / required input |
| --- | --- | --- |
| `pangolin_impact.py` | Live edit/save | Copies tail swipe `v03` to `v04_Whip`, and rear slam `v04` to `v05_Heavy`; preserves originals and creates a pre-impact backup. |
| `pangolin_slam_clearance.py` | Live edit/save | Refines `v05_Heavy` using original slam `v04`: preserves IK paw matrices, reduces compression and adjusts head clearance. Saves explicitly to `impact-v1.blend`. Head correction is capped; inspect reported residual clearance errors. |
| `pangolin_timing_v2.py` | Live edit/save | Copies `v04_Whip` to `v05_WhipHold` with a two-frame full-extension hold. Copies `v05_Heavy` to `v06_Heavy60`, preserving the drop/contact interval while shortening anticipation/recovery. Saves `impact-v2.blend`. |
| `pangolin_recoil_v3.py` | Live edit/save | Copies `v05_WhipHold` to `v06_SoftRecoil`: 80% at 79, 60% at 80, 30% at 81, recoiled at 82. Saves `impact-v3.blend`. |
| `pangolin_review.py` | Background Blender; PNGs | Renders the original or selected revision; flags below. |
| `pangolin_sheets.py` | Python + Pillow; PNG sheets | Original swipe frames 24–68 and slam frames 44–55, from `/private/tmp/pangolin-review/`. |
| `pangolin_impact_sheets.py` | Python + Pillow; PNG sheets | Initial impact variants from `/private/tmp/pangolin-impact/`. |
| `pangolin_timing_sheets.py` | Python + Pillow; PNG sheets | Held whip and 60-frame slam from `/private/tmp/pangolin-impact-v2/`. |

## Renderer flags

Pass flags after Blender's `--`, e.g. `-P scripts/blender_animation/pangolin_review.py -- --v3`.
Select one revision flag, optionally with `--slam-only`:

| Flag | Actions rendered | PNG directory |
| --- | --- | --- |
| none | Original swipe v03 / slam v04 | `/private/tmp/pangolin-review/` |
| `--impact` | Whip v04 / heavy slam v05 | `/private/tmp/pangolin-impact/` |
| `--v2` | WhipHold v05 / Heavy60 v06 | `/private/tmp/pangolin-impact-v2/` |
| `--v3` | SoftRecoil v06 only | `/private/tmp/pangolin-impact-v3/` |
| `--slam-only` | Skip swipe in the selected pair | Same as selected revision |

## Latest session result

File: `clockwork-pangolin-rigify-impact-v3.blend` (external asset directory).

- `Pangolin_Attack_TailSwipe_CW180_Sprite_v06_SoftRecoil`: 100 frames;
  full extension at 77–78, then 80%, 60%, 30%, and fully recoiled at 79–82.
- `Pangolin_Special_RearSlam_Sprite_v06_Heavy60`: 60 frames; contact at 26,
  weight held through 28, small recoil at 32, recovery to ready by 60.

The sequence above records dependencies, not an automatic replay command. Some
scripts assert that the output action does not already exist; others create Blender
numeric suffixes or overwrite a saved review file on rerun. Inspect first.
Percentage values describe interpolation between authored tail-control poses,
not linear percentages of world-space tail-tip distance.
