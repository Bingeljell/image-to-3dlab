# Resume here — 2026-08-13 morning

Supersedes `RESUME-HERE-2026-08-12.md` for what to do *next*; that document is still the
reference for the two systemic bugs fixed on 12 Aug.

## The one finding that matters

**We delete TRELLIS's metallicRoughness map on every organic asset, by default.**

`--material-mode matte` is the default (`cli.py:86`) and `trellis_backend.py:118-124` pops
`metallicRoughnessTexture`, then pins `metallicFactor: 0.0`. The Hugging Face reference ships
that map at 3072² with `metallicFactor: 1.0`. Ours is mathematically flat under any light.

Verified the orphaned texture really is an MR map before wiring it back: red channel
identically zero (glTF's unused channel), G/B distributions matching the reference's own map
(ours G 190.9±24.7 / B 107.0±32.0, reference G 196.3±20.2 / B 101.2±25.2).

**The map is still inside every GLB we have shipped** — `matte` only rewrote the JSON chunk,
so the image never left the file. `scripts/restore_pbr_material.py` re-attaches it in seconds.

**Unfinished:** whether restored-PBR actually looks better. Staged in Blender but not judged —
the fox chosen for the comparison turned out to be a bad asset, which derailed it. Redo on
Flicker (`winding_consistent: true`, the clean subject).

## What is now measured, against a control

`scripts/glb_forensics.py` — run it on anything, including the paid controls when they land.

| | HF Flicker | ours | HF moss fox | ours |
|---|---|---|---|---|
| faces | 281,889 | 293,488 | 283,149 | 282,610 |
| boundary edges | 75,673 | 88,890 | 224,057 | 295,892 |
| non-manifold edges | — | — | **1** | **1,233** |
| winding consistent | yes | yes | yes | **no** |
| `doubleSided` | false | **true** | false | **true** |
| metallicRoughness map | 3072² | **none** | 3072² | **none** |

Face counts match the reference — the cap fix landed, stop tuning them. Boundary-edge count
is retired as a gate: the reference carries 224k of them and looks great.

## Two things I got wrong this morning — do not re-derive them

1. **"Non-manifold edges prevent winding repair from converging."** False.
   `output/repair/fox_repaired.glb` has **0** non-manifold edges and is *still*
   `winding_consistent: false`. The mechanism is unknown. What is real: **every fox in
   `output/` is winding-inconsistent and the HF control is not.**
2. **The good-looking sweep renders cannot settle solidity.** `blender_lineup_sweep.py`
   never sets `use_backface_culling`, and our GLBs declare `doubleSided: true`, so
   `foxR_sweep.png` was rendered double-sided. That is not evidence the fox is bad — it is
   evidence that render cannot answer the question. Judge culled (`scripts/blender_stage.py`).

## Next, in order

1. **Judge restored-PBR on Flicker, culled.** `scripts/restore_pbr_material.py`, then
   `scripts/blender_stage.py A.glb B.glb`. If it wins, flip the `cli.py` default to `pbr`
   and re-run the fleet — minutes, no regeneration.
2. **Find what makes our meshes winding-inconsistent** when the reference is not. Work the
   chain backwards — decode → fill_holes → weld → simplify → winding repair → export —
   running `glb_forensics.py` at each stage.
3. **Diff the paid controls on arrival** (see §8 of `hunyuan-eval-2026-08-13.md` for the
   questions they should answer).
4. **Hunyuan** — weights are on disk (14 GB, `tencent/Hunyuan3D-2.1`). Deliberately not
   installed yet: proving Hunyuan beats us while our own baseline still ships damaged
   geometry would repeat the error that made every pre-12-Aug measurement worthless.

## New this morning

- `scripts/glb_forensics.py` — what a GLB actually contains
- `scripts/restore_pbr_material.py` — re-attach the orphaned MR map
- `scripts/mark_asset.py` — **record verdicts**, `output/verdicts.jsonl`. Built because a
  render that looked right could not be traced to the GLB that made it. Use it every time
  something looks good or bad, with `--culled` or `--double-sided`.
- `docs/hunyuan-eval-2026-08-13.md` — the Hunyuan position, claims graded MEASURED/READ/CLAIMED
