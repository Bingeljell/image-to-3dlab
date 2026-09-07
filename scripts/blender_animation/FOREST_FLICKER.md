# Forest Flicker recipes

Requires the catlike Flicker Rigify scene: `rig`, `geometry_0` and the named
`ForestFlicker_*` actions. Read the [safety guide](README.md) first.

| Script | Runtime / effects | Purpose / status |
| --- | --- | --- |
| `forest_flicker_inspect.py` | Live diagnostic; screenshots | Samples existing actions and restores the initial pose/action. |
| `forest_ear_diagnose.py` | Live diagnostic | Audits ear weights and evaluated motion. |
| `forest_flicker_polish.py` | Live edit/save | Repairs ear weighting, strengthens right swipe and double-paw slam; preserves original actions. |
| `forest_flicker_death.py` | Live edit/save | **Intermediate v5 builder**: damage-wince lead-in, 44-frame left-side collapse, tail/ear follow-through and FK limbs. Adds an ear bone/weights. Its leg pose required later correction. |
| `forest_death_finalize.py` | Live edit/save | Adds missing neutral root keys to other actions to prevent death-roll carryover; checks end stillness and walk reset. **Saves the current file.** |
| `forest_hind_paw_fix.py` | Live edit/save | **Superseded unsuccessful experiment.** Redirects the foot but leaves the thigh problem. Retained for history, not recommended for use. |
| `forest_hind_chain_fix.py` | Live edit/save | Accepted **v8** correction: forward knee, backward hock, controlled droop, preserved bone roll. Copies the death action and archives its input. Assumes an existing `ForestFlicker_Death_44f`; inspect the baseline before reuse. |
| `forest_flicker_preview.py` | Background Blender; PNGs | Default: polished swipe/slam. `-- --death`: 44-frame death. |
| `forest_paw_reverse_view.py` | Background Blender; PNGs | Opposite-side view of frames 31 and 44 of the **active action**. Uses the sibling preview setup. Output filenames retain the historical `v7` label regardless of input version. |

## Lessons to retain

- The accepted final death is in `forest-flicker-rigify-polished-v8.blend`, not
  reproduced by the death builder alone. These are session recipes, not a clean
  replay chain. Keep backups; do not apply intermediate fixes to the final asset.
- Redirecting the paw is not sufficient when the thigh/knee/hock arrangement is
  reversed. Inspect the whole chain from the user's angle and intermediate poses
  (especially frame 31), not only the final pose or a convenient render camera.
- On a left-side fall, right limbs settle against the body/other limbs. Gravity
  alignment alone can drive them through the body and floor.
- Ear tip weighting and ear droop are separate concerns: the polish corrects
  binding, while the death builder introduces an independently deforming ear.
