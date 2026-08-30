# Browser Creature Workshop

## Product sentence

Upload a picture of a creature, generate a 3D asset, make it deformable with a known rig,
paint how it looks, prompt or author an animation, and export a GLB.

This document is the durable product and architecture boundary for evolving the existing
local viewer into that workshop. It should be updated when an implementation decision
changes; progress logs and experiments belong elsewhere.

## Product boundary

The browser is the working surface. It owns the interactions that benefit from immediate
visual feedback:

- generate and compare image-to-3D results;
- inspect meshes, materials, UVs, skeletons, and animation clips;
- adjust PBR materials and paint texture or semantic mask layers;
- make light, non-destructive form edits;
- pose an existing skeleton, preview and edit clips, and collect animation intent;
- launch long-running jobs and present their results;
- export a portable asset.

Blender remains a local worker for operations whose correctness depends on its geometry,
deformation, or constraint systems:

- voxel remeshing and production topology preparation;
- weight transfer and weight cleanup;
- Rigify metarig construction and rig generation;
- evaluation of Rigify controls, constraints, and IK;
- baking production animation onto deform bones;
- final GLB packaging when browser output is insufficient.

The workshop is not intended to reproduce Blender, Rigify, high-poly sculpting, a modifier
stack, custom rig construction, or grooming.

## Rooms

The primary workflow is one asset moving through five rooms:

1. **Generate** — choose an image and backend, run the job, and inspect the GLB.
2. **Look** — edit materials, paint texture and semantic masks, and apply light sculpt
   layers.
3. **Rig** — choose a rig profile, place semantic joint markers, run the Blender bind job,
   and inspect deformation.
4. **Animate** — play clips, pose the skeleton, author keyframes or prompt a motion, and
   preview the result.
5. **Export** — bake requested layers and clips and download the finished GLB and workshop
   metadata.

Compare remains a cross-cutting inspection tool rather than a stage of the workflow.

## Current implementation status

As of 2026-08-30, the dependency-light browser application has reusable Compare,
Generate, Rig Review, and Animate rooms. Rig Review currently supports:

- GLB/GLTF deform-skeleton inspection with selectable tapered bone bodies and joints;
- a searchable hierarchy, mesh opacity, visibility, and x-ray controls;
- strict, fingerprint-verified `.rig.json` fit-skeleton loading;
- direct fit-joint selection and camera-plane dragging;
- precise armature-local XYZ edits, profile-defined mirroring, undo, redo, and reset;
- corrected sidecar download without mutating the source GLB or its bind pose.

This completes the browser half of joint correction. It does not yet make the corrected
placement authoritative: that requires the Blender rebind worker described below.

## Architecture direction

Keep the current dependency-light application: native ES modules, vendored Three.js, and
the local Python HTTP server. A framework migration or bundler is not needed to establish
good component boundaries.

The viewer is split into four layers:

```text
App shell
  -> mode controllers (Compare, Generate, Look, Rig, Animate, Export)
      -> reusable UI components
          -> shared 3D and asset primitives
              -> Three.js and local job APIs
```

The central reusable object is a `ViewportPane`. It owns one scene, camera, renderer,
controls, loaded asset, render invalidation, resizing, and disposal. A mode composes one or
more panes and adds its own tools; it does not recreate the renderer or loader.

An `AssetSession` is the shared model passed between rooms. It will eventually describe:

- source and current asset URLs;
- mesh and material inventory;
- skeleton, rig profile, and semantic bone map;
- embedded and workshop-authored clips;
- texture, mask, morph, and displacement layers;
- job history and export settings.

The initial refactor must preserve all existing behavior before introducing this state
model wholesale. Extract proven code first, then make the data contract explicit as new
rooms require it.

## Proposed module map

```text
viewer/
  index.html                 app shell and stable DOM landmarks
  app.js                     mode routing and application startup
  styles/
    base.css                 tokens, shell, and shared controls
    compare.css              compare layout and panes
    generate.css             generation and job progress
    workshop.css             shared room layout, added with the first new room
  core/
    render-loop.js           shared demand-driven frame scheduling
    model-loader.js          GLB, GLTF, and OBJ loading and normalization
    viewport-pane.js         scene/camera/controls/asset lifecycle
    asset-session.js         cross-room asset state, added when required
  components/
    drop-zone.js
    progress-panel.js
    property-panel.js
    tabs.js
  modes/
    compare.js
    generate.js
    look.js
    rig.js
    animate.js
    export.js
  animation/
    player.js
    skeleton.js
    timeline.js
    animation-recipe.js
```

Only create a module when it owns a coherent responsibility or is reused. The map is a
direction, not a requirement to create empty abstractions.

## Animation contract

### Decisions recorded 2026-08-29

- Browser animation is a real authoring and export path, not merely a read-only preview.
- A Blender/Rigify rebake remains available as the high-fidelity path; the two paths share
  timing and semantic control data rather than sharing an implementation.
- The first supported runtime rig profile is one canonical quadruped profile derived from
  the existing Rigify quadruped workflow.
- Human manipulation and LLM animation operate the same versioned semantic controls.
- Incorrect bind-joint placement is corrected through browser markers followed by a
  Blender rebind. It is not silently treated as a pose edit.

### Joint placement correction

A bound skeleton has two distinct states that the UI must not conflate:

- **rest state** — joint placement, bone length/orientation, inverse bind matrices, and the
  basis against which vertex weights were authored;
- **pose state** — temporary transforms evaluated relative to that rest state.

Dragging a pose bone cannot safely correct a misplaced bind joint. Moving a rest joint
after binding also changes inverse bind matrices, IK chain lengths, Rigify relationships,
and the deformation expected by nearby weights. Updating matrices alone would preserve
the old, potentially wrong weight field.

The supported correction loop is therefore:

```text
Blender bind
  -> browser Rig Review
  -> edit semantic correction markers
  -> Blender rebind and weight transfer
  -> replacement rigged GLB
```

Rig Review shows the skeleton through the mesh and exposes only meaningful markers such as
shoulders, hips, knees, paws, neck, head, and tail root. It may mirror adjustments, compare
original/corrected locations, and preview limb reach, but `Rebind` is the operation that
makes a correction authoritative. Weight painting or local weight repair is a separate
future tool.

### Rig Review contract

Rig Review deliberately presents two related skeletons:

- the **deform skeleton** embedded in the GLB, which is authoritative for runtime skinning
  and animation playback;
- the **fit skeleton** in a versioned sidecar, which is authoritative for editable Rigify
  metarig placement and Blender rebinding.

The browser may map selection between them, but it never writes a joint-placement
correction directly into a `DEF-*` bone and calls the bind fixed. A deform bone can be
derived, segmented, or constrained differently from the metarig bone that produced it.

The Blender bind result therefore becomes:

```text
creature.glb
creature.rig.json
```

The minimum rig-sidecar shape is:

```json
{
  "schemaVersion": 1,
  "rigProfile": "rigify.quadruped.v1",
  "assetFingerprint": "sha256:...",
  "coordinateSpace": "armature-local",
  "mirror": { "axis": "X", "origin": 0 },
  "joints": {
    "front_left.shoulder": {
      "label": "Front left shoulder",
      "position": [0.18, 0.42, 0.11],
      "sourceBone": "DEF-upper_arm.L",
      "parent": "spine.chest",
      "mirrorOf": "front_right.shoulder"
    }
  },
  "corrections": {}
}
```

Coordinates are always armature-local, never viewer-normalized or camera/world space. The
viewport currently centers and scales assets for display; edit gizmos must invert that
display transform before serializing a correction.

Each correction records the source rest position as well as the target and delta:

```json
{
  "front_left.shoulder": {
    "sourcePosition": [0.18, 0.42, 0.11],
    "targetPosition": [0.20, 0.45, 0.09],
    "delta": [0.02, 0.03, -0.02],
    "mirrored": false
  }
}
```

The source position plus asset fingerprint prevents a correction from being silently
applied to a different bind or rig revision. A stale mismatch is an error requiring user
review, not an automatic best-effort import.

Selection has two levels:

- selecting a rendered bone body identifies the deform bone and highlights its mapped fit
  joints;
- selecting a joint marker chooses the connected endpoint that Rig Review can move.

Moving a joint preserves chain connectivity: a knee is both the upper leg's endpoint and
the lower leg's origin. Leaf endpoints that cannot be recovered from glTF must be explicit
in the sidecar.

Rig Review controls are introduced in this order:

1. tapered deform-bone rendering, joint markers, names, hierarchy, x-ray, and mesh opacity;
2. fit-skeleton sidecar validation and overlay;
3. original/corrected ghosting and translation gizmos;
4. profile-defined mirroring, reset, undo, and correction export;
5. Blender rebind and fixed deformation-review poses.

Animation playback is disabled while editing fit joints. The browser may preview alignment
and reach, but only the returned rebind is authoritative for deformation quality.

### Headless Blender rebind contract

The browser owns the interaction, while a backend starts a fresh Blender process for every
rebind. It does not drive an already-open Blender UI or depend on whatever scene happens to
be active there.

The first reliable implementation accepts:

```text
source.blend                 preserved clean bind scene with metarig and source mesh
source.glb                   exact browser-reviewed asset
corrected.rig.json           full sidecar downloaded/generated by Rig Review
```

The `.blend` is required in v1 because glTF exports deform bones and baked clips, not the
complete Rigify metarig, control rig, constraints, drivers, or generation parameters. A
later deterministic rig profile may reconstruct that authoring state from semantic joints,
but a GLB alone is not treated as sufficient evidence.

The worker performs this transaction:

```text
validate schema + source GLB SHA-256
  -> open a copy of source.blend
  -> resolve profile joint ids to metarig edit-bone endpoints or JOINT_* markers
  -> apply correction targetPosition values in armature-local coordinates
  -> regenerate Rigify
  -> rebuild/transfer production weights
  -> export corrected GLB
  -> save corrected .blend
  -> write refreshed .rig.json and bind-report.json
```

It writes into a new job directory and never overwrites the source scene. Any unknown joint,
missing profile mapping, fingerprint mismatch, Rigify failure, weight-transfer failure, or
missing output is a terminal error. Partial outputs remain diagnostic artifacts and are not
presented as a successful bind.

The local API surface is intentionally parallel to Generate:

```text
POST /api/rig/rebind
GET  /api/rig/rebind/<job>/events
GET  /api/rig/rebind/<job>/status
POST /api/rig/rebind/<job>/cancel
GET  /api/rig/rebind/<job>/result.glb
GET  /api/rig/rebind/<job>/scene.blend
GET  /api/rig/rebind/<job>/rig.json
GET  /api/rig/rebind/<job>/report.json
```

The browser retains the selected source files and its in-memory correction session. The
user presses **Rebind**, watches staged progress, and Rig Review automatically replaces the
preview with the returned GLB while keeping before/after access. JSON download remains an
escape hatch rather than the primary workflow.

The returned scene is optional for ordinary review. A local **Open in Blender** helper may
be added later, but the portable contract is a downloadable `.blend`; remote browsers must
not be allowed to launch arbitrary desktop files.

Implementation proceeds in independently testable slices:

1. pure-Python sidecar validation, fingerprint verification, and correction planning;
2. Blender-side application of that plan to a preserved scene, with report and copy output;
3. profile-specific Rigify regeneration, voxel weight transfer, GLB export, and fixed poses;
4. queued backend endpoints, cancellation, progress, and artifact serving;
5. browser submission, progress, automatic result loading, and before/after review.

### Rigify and glTF

Rigify's authoring controls, drivers, and Blender constraints do not become an equivalent
runtime control rig in glTF. A browser normally receives deform bones and baked animation
clips. The workshop therefore supports two animation paths.

**Interactive browser path**

- use the exported deform skeleton;
- inspect, play, scrub, blend, and loop glTF clips;
- select bones and author FK rotations with viewport gizmos;
- provide small runtime helpers such as look-at, foot placement, or lightweight IK;
- store edits as a portable, semantic animation recipe;
- build a Three.js `AnimationClip` for immediate preview.

**Production bake path**

- send the animation recipe and rig profile to a headless Blender job;
- map semantic controls to the corresponding Rigify controls;
- evaluate Blender constraints and IK;
- bake the result onto deform bones;
- return a GLB containing the production clip.

The preview and baked output must share timing and intent, but they need not use the same
solver internally.

### Runtime control rig

The GLB carries the skinned mesh, deform skeleton, and baked clips. A versioned rig-profile
sidecar reconstructs the smaller browser control rig that glTF cannot carry:

```text
creature.glb
  mesh + materials + deform bones + baked clips

rigify.quadruped.v1.json
  semantic bone map
  rest transforms and symmetry
  joint limits and preferred bend directions
  IK chains and pole-vector definitions
  browser-control -> Rigify-control mapping
```

The initial quadruped control surface consists of:

- master/root, pelvis, chest, and head-look controls;
- four paw targets and four knee/elbow pole targets;
- FK rotations for limbs, spine, neck, and tail;
- per-limb IK/FK blend;
- foot rotation and, later, ground locking.

FK writes constrained local bone rotations. IK solves a configured limb chain from its
target and pole vector. An IK/FK blend evaluates both solutions and blends their local
position/quaternion/scale transforms before skinning. The first limb solver should be a
predictable analytical two-bone solver; generic whole-body IK is not required for v1.

The timeline records semantic control tracks where possible, then evaluates and bakes them
to deform-bone glTF tracks for portable browser export. The production path maps those same
tracks to Rigify controls in Blender, evaluates constraints and drivers, and bakes the
deform bones there.

### Semantic recipes

An LLM must return validated data, not JavaScript and not unconstrained raw bone names. A
rig profile maps stable semantic controls onto a particular exported skeleton and Rigify
setup.

Example:

```json
{
  "schemaVersion": 1,
  "clip": "alert_head_turn",
  "rigProfile": "rigify.quadruped.v1",
  "fps": 30,
  "duration": 1.4,
  "loop": false,
  "beats": [
    { "time": 0.0, "pose": "neutral" },
    {
      "time": 0.35,
      "controls": {
        "head.yaw": -18,
        "neck.pitch": 8,
        "ears.alert": 0.85
      },
      "easing": "easeOut"
    }
  ]
}
```

The recipe schema should support progressive levels of control:

1. named motions and parameters (`walk`, `headbutt`, `idle`, speed, energy);
2. semantic pose controls at timed beats;
3. explicit semantic bone transforms for manual editing;
4. optional events such as footsteps or impact markers.

Raw bone tracks are an export detail produced after validation and mapping.

The control system is required even when animation is LLM-powered. The model selects and
sequences validated controls; it does not replace the solver, invent bone names, or emit
runtime JavaScript. Manual gizmos, timeline editing, procedural motions, and the LLM all
read and write the same recipe representation.

## Workshop file contract

The portable result is a GLB plus a versioned sidecar during authoring:

```text
creature.glb
creature.workshop.json
rigify.quadruped.v1.json
textures/
  albedo.png
  roughness.png
  fur-mask.png
  charge-mask.png
animations/
  idle.animation.json
  headbutt.animation.json
```

The sidecar records rig profile and bone mapping, source/result provenance, paint and sculpt
layers, animation recipes, clip inventory, and export settings. Export may fold supported
data into a single GLB, but the authoring format remains explicit and diffable.

## Delivery sequence

1. ~~Refactor Compare and Generate into reusable application modules.~~
2. ~~Build Animate inspection: clip playback, scrubbing, skeleton overlay, bone selection,
   and pose reset.~~
3. ~~Build Rig Review inspection and editable, versioned fit-joint corrections.~~
4. Add the preserved-scene Blender rebind worker and browser-launched job.
5. Add fixed deformation-review poses and before/after acceptance.
6. Add browser keyframes and versioned animation recipes.
7. Add prompted animation against the validated recipe schema.
8. Add the Blender production animation-bake job.
9. Add Look material controls, mask painting, and non-destructive sculpt layers.
10. Consolidate export and workshop project persistence.

## Remaining animation questions

These decisions can be resolved while iterating on the first canonical quadruped profile:

- Is the canonical profile the unmodified Rigify Basic Quadruped setup or the existing
  project-specific derivative?
- Should the first LLM surface generate whole clips, edit selected time ranges, or choose
  and parameterize trusted procedural motions?
- Which motions prove the product: idle, walk/trot, headbutt/attack, look-at, or a custom
  prompt?
- At which iteration do foot locking and terrain contact become required?
- Should animation recipes be reusable across differently proportioned creatures of the
  same rig profile?

## Refactor invariants

The refactor is complete only when the existing application still supports:

- GLB, GLTF, OBJ, and source-image loading;
- up to three compare panes and query-string asset loading;
- camera synchronization, fixed views, spin, overlays, and pane replacement;
- backface culling, wireframe, flat-grey, and normals inspection;
- all generation backend forms and setup checks;
- image alpha inspection, generation submission and cancellation;
- SSE progress with status polling fallback;
- generated-result preview and downloads;
- restricted embedded viewer mode;
- the same `python viewer/serve.py` local workflow.
