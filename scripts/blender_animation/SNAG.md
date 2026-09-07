# Snag recipes

Requires the tentacle/root-monster scene with `SnagRig`. Read the toolkit
[safety guide](README.md) before execution. `Live edit/save` is not a diagnostic.

| Script | Runtime / effects | Purpose and prerequisites |
| --- | --- | --- |
| `snag_attack_inspect.py` | Live diagnostic; screenshots | Samples active `Attack_Whip`; inspects keys and poses. |
| `snag_attack_review.py` | Live edit/save | Builds `Attack_Whip_24f_Review` from `Attack_Whip`; stronger coil, release and follow-through. Default runs the build. `check` checks ready-pose return and death carryover **and saves the current file**; requires `Death_48f_Review`. |
| `snag_death_review.py` | Live; mode-dependent | Default `inspect` reads scene state. `build` creates `Death_48f_Review` from `Death` and saves. `snapshot FRAME` captures a pose. `validate` checks stillness/twitch **and saves the current file**. |
| `snag_idle_axis_fix.py` | Live edit/save | Preserves `Idle` as `Idle_BeforeAxisFix` and keys missing neutral transforms; requires the reviewed attack/death actions. |
| `snag_attack_preview.py` | Background Blender; PNGs | Temporary studio render of the attack review. |
| `snag_death_preview.py` | Background Blender; PNGs | Temporary studio render of the death review, including a temporary ground plane. |

Session history: death review → attack review → idle axis fix. This is context,
not permission to blindly replay the sequence against a finished asset. Inspect
action names and output paths in each file first.
