# Rig Edit handoff — 2026-08-31

This is the restart point for the browser rigging work. The durable product and architecture
decisions remain in [browser-workshop.md](browser-workshop.md).

## Working product boundary

The browser owns inspection, semantic fit-joint correction, pose/animation preview, and job
launching. Blender remains the authoritative worker for metarig editing, Rigify generation,
weight transfer, constraint evaluation, animation baking, and production GLB export.

Rig Edit and Pose are intentionally different modes:

- **Rig Edit** changes the rest/fit skeleton and requires a Blender rebind before deformation
  can be judged.
- **Pose** changes transforms on an already-bound skeleton and is the basis for browser and
  LLM animation work.

The GLB normally contains the generated deform rig and baked clips, not Rigify's complete
Blender control rig. Browser animation controls therefore need their own semantic control
layer; production-quality IK/FK evaluation can be sent back to Blender for baking.

## What is implemented

- Reusable Compare, Generate, Rig Edit, and Pose/Animate rooms.
- GLB/GLTF skeleton inspection with independently toggled current-rig bones and joints.
- Searchable deform hierarchy and selectable viewport bone/joint overlays.
- High-contrast overlays: orange editable fit joints, slate-grey fit bones, cooler current-rig
  markers, and cyan selection.
- Front, Back, Left, Right, Top, Bottom, and Reset camera controls.
- Strict `.rig.json` parsing plus GLB and Blender-scene fingerprint verification.
- Semantic fit joints that can be moved by camera-plane drag, Blender-axis XYZ gizmos, or
  numeric Blender armature-local coordinates.
- Mirrored edits, undo, redo, selected-joint reset, and complete edit reset.
- A queued **Rebind in Blender** workflow returning a replacement GLB, prepared `.blend`,
  refreshed sidecar, and diagnostic report without overwriting the source files.
- A versioned Rigify Basic Quadruped adapter for Blender 5.2.
- Pose/Animate clip loading, playback, scrubbing, looping, skeleton inspection, and pose reset.

The latest alignment fix converts Blender's Z-up armature coordinates into the browser's
Y-up glTF coordinates for display, applies the inverse conversion to edits, rotates the XYZ
gizmo consistently, and restores bind pose when Rig Edit loads. This is commit `545c3c0`.

## Pangolin bundle

The current verified test bundle is in `output/pangolin-rig-edit/`:

- `clockwork-pangolin.glb`
- `clockwork-pangolin.rig.json`
- `clockwork-pangolin-prepared.blend`

Load all three in one multi-file selection or one drag into the single Rig Edit drop zone.
Sequential drops replace the current load and should be avoided.

The exported GLB contains one skin, 283 joints, and 11 clips. The sidecar contains 27
semantic fit joints. Its GLB and `.blend` fingerprints were verified. The original open file
`clockwork-pangolin-rigify.blend` was not overwritten; it was dirty at export time, and the
prepared copy captured its then-current state.

## Normal Rig Edit workflow

1. Start the viewer from the repository:

   ```bash
   source .venv/bin/activate
   python viewer/serve.py
   ```

2. Open `http://127.0.0.1:8777/viewer/index.html` and enter **Rig Edit**.
3. Load the GLB, matching `.rig.json`, and prepared `.blend` together.
4. Confirm the right panel says the sidecar and Blender scene are verified.
5. Select an orange semantic joint. Grey fit bones select an endpoint joint; generated
   helper bones such as `MCH-*` and `VIS_*` may correctly report **No mapped fit joint**.
6. Drag the cyan marker in the camera plane, drag a colored XYZ handle for a constrained
   move, or enter coordinates numerically.
7. Use Undo/Redo or **Reset Rig Edits** freely. Moving a marker changes only the fit sidecar;
   it does not alter the loaded GLB's weights.
8. Press **Rebind in Blender** after at least one correction. Inspect the returned GLB rather
   than judging deformation from the pre-rebind preview.

## Decisions made

- Bones are derived from connected fit joints; a bone body is not independently translated.
- Matching GLB/sidecar/Blender bundles must align automatically. No manual axis-flip control
  is needed now.
- Do not add a display-only model-versus-rig flip: it would show an alignment Blender rebind
  cannot reproduce. If separately oriented inputs are supported later, alignment must become
  explicit sidecar metadata and be applied by the Blender worker.
- `.rig.json` remains a useful download/debug escape hatch, but the normal user flow is the
  browser-launched Blender rebind.
- Pillow belongs in the project virtual environment; it does not need a global install.

## First checks next session

1. Hard-refresh the viewer, reload the three-file pangolin bundle, and press **Reset Rig
   Edits** to discard the accidental correction made while the overlay was misaligned.
2. Inspect fit alignment from all six fixed camera directions.
3. Move one obvious joint a small amount along each XYZ handle. Confirm the marker, connected
   bones, and numeric fields agree; then test Undo and Reset.
4. Submit one small correction through **Rebind in Blender** and compare the returned bind and
   deformation to the source.
5. Open the pangolin in Pose/Animate and manually play every clip, especially
   `Pangolin_Walk_InPlace_v01`.

## Known issues and pending work

### Rig/export

- The new coordinate/bind-pose alignment fix has automated coverage but still needs the
  manual pangolin inspection above.
- Blender's glTF exporter warned that vertices with more than four joint influences were
  reduced to the strongest four. Deformation impact needs inspection.
- The Walk Action produced invalid-fcurve-path warnings during export. The clip is present in
  the GLB, but browser playback must confirm it contains the intended motion.
- A hidden generated `rig` can make Blender export a mesh-only GLB while still reporting
  success. The reusable bundle exporter should explicitly unhide/include the armature and
  reject output without a skin and expected clips. The current pangolin GLB passed those
  validations after the rig was made visible.
- The Blender bridge has shown port drift between 9876 and 9877 after background Blender
  processes were launched. Process/port ownership should be stabilized, but this does not
  block the file-based browser workflow.
- Only `rigify.basic-quadruped.blender-5.2.v1` is currently supported. Generalization requires
  more versioned profiles or a guided mapping/export surface, not assumptions about bone
  names.

### Rig Edit

- Add fixed deformation-review poses and explicit before/after comparison after rebind.
- Add deformation metrics and focused weight-repair tools only after the basic rebind loop is
  visually reliable.
- Large `.blend` uploads are currently buffered; streaming multipart uploads are a later
  robustness improvement.

### Animation

- Decide the first semantic browser control set for the quadruped: likely root, torso, head,
  four foot targets, and four pole targets.
- Add FK rotation and lightweight browser IK controls over the exported deform rig.
- Define a versioned, rig-agnostic animation recipe shared by manual controls and the LLM.
- Build immediate Three.js clip preview from that recipe.
- Build the Blender production-bake job that maps semantic controls to Rigify, evaluates its
  IK/FK constraints, bakes deform bones, and returns a new GLB clip.
- Use a small acceptance motion set first: idle, walk/trot, look-at, and one attack.

## Verification and relevant commits

The alignment change passed the focused browser-module tests and the full suite:

```text
641 passed, 4 existing warnings
```

Recent Rig Edit commits, newest first:

- `545c3c0` — align Blender fit skeletons and force bind pose
- `672e263` — distinguish inspection-only current-rig state
- `bf311de` — preserve skeleton overlay colors
- `37c1b3d` — edit semantic joints without relying on generated bone names
- `e11acc9` — render editable fit skeleton clearly
- `7128eda` — add fit-joint translation gizmo
- `dc9bdf5` — add standard camera views
- `6496ab2` — clarify current-rig and fit-skeleton layers
