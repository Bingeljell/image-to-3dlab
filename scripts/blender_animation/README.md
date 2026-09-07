# Blender animation toolkit

Model-specific recipes from the Snag, Forest Flicker and Clockwork Pangolin
animation sessions. Consult these before writing new animation automation.
Reuse their sampling, retiming, pose-keying and visual-review techniques; they
are not a generic, one-command rebuild pipeline.

## Safety and prerequisites

- Live recipes use the existing `../blender_inspect.py` client via `_rpc.py`,
  connecting to Blender's execute-code server at `127.0.0.1:9876`.
- They assume exact rig, mesh, action and bone names. Most were authored for
  Blender 5.2 / Rigify. Inspect the loaded file, active action and source first.
- Many scripts execute immediately, including when imported. **Do not import
  recipes, run them with `--help`, or run all scripts as a test.** Some default
  to modifying animation and saving files; unknown arguments are not rejected.
- Edit recipes may rename/remove actions, alter mesh weights and overwrite
  versioned outputs. They are not uniformly idempotent. Back up unsaved work;
  check destinations and use a new review copy before deliberately running one.
- Paths under `/Users/nikhilshahane/projects/` and `/private/tmp/` are retained
  as session context, not portable defaults. Inspect/adapt them first. Saved
  `.blend` assets live outside this repository and are not supplied by these scripts.
- Preview scripts run **inside Blender** on a disposable/background copy. They
  change render/camera settings in memory and write PNGs, but do not save the blend.
- Contact-sheet scripts use Pillow in the repository `.venv`. They require the
  corresponding PNG sequences and may overwrite existing sheets.

## Running and verifying

From the repository root, after reviewing the relevant recipe and prerequisites:

```sh
# Read-only inspection of the live scene using the existing general tool:
python3 scripts/blender_inspect.py

# Example live diagnostic (requires the Forest Flicker scene):
python3 scripts/blender_animation/forest_ear_diagnose.py

# Example isolated render (replace the blend path with a review copy):
/Applications/Blender.app/Contents/MacOS/Blender -b /path/to/review.blend \
  -P scripts/blender_animation/pangolin_review.py -- --v3
```

Do not validate solely through transforms or a single camera. Review anticipation,
contact, intermediate frames and recovery visually, including the opposite side
when limbs overlap. Check foot/ground clearance, anatomical knee/hock direction,
quaternion continuity, final holds and switching from other actions. Preserve
original actions and identify the new active action clearly at handoff.

For repository maintenance, compile source strings (including embedded RPC code)
without importing recipes or connecting to Blender. Syntax checks alone do not
establish that animation or deformation is correct.

## Existing general-purpose tools

Keep using the existing `scripts/blender_*.py` tools for scene inspection,
Rigify setup, generic animation cycles, rendering and rig/weight work. This folder
supplements them with creature-specific recipes; it does not replace them.
