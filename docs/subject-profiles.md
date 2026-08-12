# Subject profiles

Proposal. Different kinds of subject need genuinely different settings and stages, and
we keep rediscovering that by hand. A **subject profile** turns that tribal knowledge
into executable defaults.

## The evidence this is real, not tidiness

Every one of these was measured, not assumed:

| Finding | Human (Nikita) | Fuzzy quadruped (moss fox) |
|---|---|---|
| Best `pipeline_type` | **512** — 1024 exposed shell interior | **1024_cascade** — 512 shredded the leaves |
| Boundary edges (holes) | 2.5% of faces | **17.5%** — 7x more |
| Generation time | ~2-5 min | ~14-18 min |
| Rigging strategy | analytic from a T-pose; no marking needed | needs guided joint marking |
| Best 2-view pair | front + back (face carries identity) | opposing 3/4s (flanks are the big surface) |
| Foliage labelling | not applicable | required for wind |
| Rigid props | the beer mug, bound to a hand bone | none |

The resolution result is the sharpest: **a setting that won decisively on one subject
lost decisively on the other.** Carrying a "best setting" across subjects is an error,
and it has already cost us one wasted run.

The hole-count gap is the second: it tracks **thin geometry** (fur, leaf tips finer
than the voxel grid) rather than anything about the pipeline. A fuzzy subject will
always perforate more than a clothed human, no matter what we fix.

> **Corrected 2026-08-10 — the cause is surface detail, not thinness.** A five-subject
> controlled comparison separated the two variables. Holding the surface smooth and
> changing the form from chunky to thin moved hole size 0.48 → 1.07; holding the form
> chunky and changing the surface from smooth to carved moved it 0.48 → **44.78**. A
> ceramic creature with paper-thin ears and a tapering tail came out essentially clean.
> The fox perforates because of its fur, not because it is fine. See
> `docs/open-questions.md` §1c, and §1d for the fix (the holes are zero-thickness
> sheets, and a Solidify pass closes all of them).

## Recommended shape: profiles, not parallel pipelines

The instinct is "two pipelines". That would be a maintenance trap — generation, texture
bake, provenance, licence gating, labelling, and export are **identical** across
subjects. What actually differs is (a) parameter defaults and (b) which optional stages
run.

So: **one pipeline, a subject profile that supplies defaults and switches stages on.**
This fits the manifest, which is already the traceable way to run:

```json
"subject": {
  "class": "humanoid",
  "features": ["cloth", "hair", "props"]
}
```

- **`class`** — sets generation defaults and picks the rigging strategy.
  Candidates: `humanoid`, `quadruped`, `object`.
- **`features`** — switches on optional stages, each of which already exists or is
  planned: `foliage` (mask labelling + stiffness bake), `cloth`, `hair`, `props`
  (rigid bind to a bone), `emissive`.

Explicit parameters in the manifest keep overriding the profile, so a profile is a
starting point rather than a straitjacket. Provenance should record the profile used,
so a run stays reproducible.

## Why the split is class + features rather than one axis

The user's framing was right: it is two dimensions, not one. A plain clothed humanoid
and a caped, long-haired humanoid share generation settings but need completely
different downstream stages. Likewise a simple prop and a moss fox share `object`-ish
generation but only one needs foliage labelling.

Collapsing that into a single list of subject types would multiply badly
(`humanoid_simple`, `humanoid_cloth`, `humanoid_cloth_hair`, ...). Class for *how to
generate and rig*, features for *what extra stages to run*.

## What each class would carry today

Based only on what we have measured, so this is a starting point, not a spec:

**`humanoid`** — `pipeline_type: 512`; T-pose source guidance; analytic rigging from
measured landmarks; props bound rigidly to hand bones. Expect few holes.

**`quadruped`** — `1024_cascade`; standing pose, limbs separated, tail clear of the
body; guided joint marking for rigging; expect many holes on fuzzy subjects and do not
treat that as a defect to chase.

**`object`** — no rigging; the interesting axis is materials rather than skeletons.

## Open questions

- Should a profile *gate* like `validate_run_policy` does — refusing a T-pose humanoid
  profile when the input clearly is not one — or only supply defaults?
- Can class be **detected** from the source image rather than declared? Tempting, but a
  wrong guess would silently apply wrong settings; declaring is safer.
- Where do multi-view defaults live — the profile knows a quadruped wants opposing 3/4s
  while a humanoid wants front + back.
