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
- corrected sidecar download without mutating the source GLB or its bind pose;
- a queued **Rebind in Blender** action with staged progress and cancellation;
- automatic replacement-GLB inspection plus `.blend`, sidecar, and report downloads.

The rebind worker now validates both fingerprints, applies persistent-ID endpoint mappings,
regenerates Rigify, builds a voxel weight proxy, transfers weights to the textured mesh,
exports a replacement GLB, saves a new scene, and refreshes the sidecar. The source scene
and source GLB are never overwritten.

The first built-in adapter targets Rigify Basic Quadruped in Blender 5.2. Prepare an initial
authoring bundle from an existing clean bind scene with:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background source.blend \
  --python scripts/blender_export_rig_binding.py -- \
  source.glb prepared.blend creature.rig.json \
  rigify.basic-quadruped.blender-5.2.v1
```

Load `source.glb`, `prepared.blend`, and `creature.rig.json` together in Rig Review. After
editing at least one fit joint, **Rebind in Blender** submits the complete verified bundle;
there is no manual JSON handoff during normal iteration.

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

Bone names are not a portable API. Rigify templates, Blender versions, custom rigs, and
studio naming conventions may all differ. Generality therefore comes from separating:

- a stable semantic profile (`front.left.elbow`, `spine.chest`, `head`);
- a versioned adapter that knows how to seed a particular known rig template;
- an asset-specific binding manifest exported from the actual Blender scene.

The exporter stamps the metarig object and relevant metarig bones with persistent IDs, then
embeds the resolved endpoint mapping in the sidecar:

```json
{
  "binding": {
    "adapter": "rigify.basic-quadruped.blender-5.2.v1",
    "sceneFingerprint": "sha256:...",
    "metarigObjectId": "c995...",
    "joints": {
      "front.left.elbow": {
        "targets": [
          {
            "boneId": "8fb1...",
            "boneName": "front_thigh.L",
            "endpoint": "tail"
          }
        ]
      }
    }
  }
}
```

`boneName` is a diagnostic fallback; `boneId` is authoritative. The `.blend` fingerprint
prevents that mapping from being used against a different source scene. Built-in adapters
make common Rigify human and animal templates automatic. An arbitrary custom rig requires
a one-time mapping wizard in which the user assigns semantic joints to bone endpoints; that
mapping then travels with every asset and animation. Anatomy cannot be reliably inferred
from arbitrary names alone.

Selection has two levels:

- selecting a rendered bone body identifies the deform bone and highlights its mapped fit
  joints;
- selecting a joint marker chooses the connected endpoint that Rig Review can move.

Moving a joint preserves chain connectivity: a knee is both the upper leg's endpoint and
the lower leg's origin. Leaf endpoints that cannot be recovered from glTF must be explicit
in the sidecar.

#### Rig Review interaction contract

The workshop presents skeleton work as two mutually exclusive user modes. Their state is
never interchangeable:

- **Rig Edit** changes what the rest skeleton *is*. It edits semantic fit-joint positions,
  produces pending corrections, and requires a Blender rebind before deformation is
  authoritative.
- **Pose** changes what the last successfully bound skeleton *is doing*. It owns FK/IK
  controls, transient transforms, keyframes, clips, and later LLM animation recipes. It
  never changes the fit-joint correction sidecar.

These correspond conceptually to Blender Edit Mode and Pose Mode without claiming to expose
Blender's complete toolset. The existing Rig Review room becomes Rig Edit; the existing
Animate room becomes Pose. Internal route and DOM names may remain stable while the UI uses
the clearer user terminology.

Switching modes does not reinterpret or silently apply state. If Rig Edit has pending
corrections, Pose continues to use the last successful bind and says that the pending edits
are awaiting rebind. After a successful rebind, the replacement GLB becomes authoritative
and transient pose transforms reset because the rest skeleton may have changed.

Selection must be visually unambiguous and synchronized across the viewport, hierarchy,
and inspector. A selected joint enlarges slightly and receives the high-contrast selection
colour; a selected bone becomes brighter and thicker. Hover uses a weaker version of the
same treatment. The matching hierarchy row is selected, and the inspector shows the
semantic label, coordinates, and editability. An editable selection also displays its
translation control. Colour alone is not sufficient selection feedback.

Rig Review edits the **fit joints**, not the final deform skeleton. The primary correction
interaction is:

```text
select fit joint
  -> translate with an XYZ gizmo or enter armature-local coordinates
  -> update connected fit-bone lines
  -> record a semantic joint correction
  -> Rebind in Blender
```

Camera-plane dragging may remain as a quick gesture, but the XYZ gizmo is the precise,
discoverable control. The browser converts the displayed position back into armature-local
coordinates before updating `corrections`.

A bone is derived from its head and tail joints and is not serialized as an independently
translated object. Selecting a fit bone reveals and highlights its endpoint handles; the
user then moves the relevant joint. A later limb-translation tool may move several joints
as one operation, but it must still produce ordinary joint corrections. Moving or aligning
the entire fit armature is a distinct, explicit operation so a local correction cannot
accidentally offset the whole character.

Visibility is layered even though the structures are connected. The primary layers are:

- **Current rig** — the GLB deform skeleton, with separate Bones and Joints visibility;
- **Fit controls** — the editable sidecar skeleton, with separate Bones and Joints
  visibility;
- **Mesh** — independent visibility and opacity.

The correction-oriented default is Fit joints visible, Fit bones visible but subdued,
Current rig hidden, and the mesh partially transparent. This avoids presenting two
overlapping skeletons as one dense set of controls.

The viewport includes Front, Back, Left, Right, Top, and Bottom view controls, plus Frame
model, Frame selection, and Reset perspective. Axis-aligned views use an orthographic
camera; orbiting may return to perspective. Because arbitrary GLBs do not reliably encode
which way a creature faces, the rig sidecar or profile records the character's up and
forward axes. View labels are resolved through that orientation metadata rather than
assuming every asset uses the same generator coordinates.

The viewport palette encodes structure first and selection second:

- bones use a visible light slate grey rather than black;
- fit joints use warm orange;
- the final deform rig uses a cooler blue-grey when shown;
- selection uses bright cyan or yellow, with increased size or thickness;
- hover uses a paler, lower-intensity selection treatment;
- the background remains charcoal for material inspection.

The correction JSON records changed semantic joint positions only. Bone segments are
reconstructed from their joint endpoints, which keeps the format independent of rendered
bone geometry and avoids treating arbitrary Rigify or custom-rig bone names as the public
editing API.

Reset actions are scoped and named for the state they affect:

- **Reset View** changes only the camera;
- **Reset Pose** restores the current bound skeleton without discarding rig corrections or
  saved clips;
- **Reset Rig Edits** discards pending semantic joint corrections without changing the
  camera or pose data.

There is no generic Reset action whose effects depend on hidden state.

Rig Review controls are introduced in this order:

1. unmistakable synchronized selection feedback and a legible semantic palette;
2. independent Current rig, Fit controls, Bones, Joints, and Mesh visibility;
3. precise fit-joint translation with an XYZ gizmo and numeric coordinates;
4. standard camera orientations, framing, and perspective reset;
5. original/corrected ghosting and fixed deformation-review poses.

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

Both the GLB and `.blend` fingerprints are validated before Blender starts. The source scene
fingerprint and persistent bone IDs come from the asset-specific binding manifest, so the
worker does not assume that every Blender installation uses the same bone names.

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

Implemented slices:

1. pure-Python sidecar validation, fingerprint verification, and correction planning;
2. versioned adapters plus asset-specific persistent-ID binding manifests;
3. Blender-side correction, Rigify regeneration, voxel weight transfer, GLB export, scene
   copy, refreshed sidecar, and diagnostic report;
4. queued backend endpoints, cancellation, SSE/polling progress, and artifact serving;
5. browser submission, progress, automatic result loading, and artifact downloads.

Next iterations:

1. fixed deformation-review poses and explicit before/after switching;
2. a guided Blender mapping/export surface for custom rigs and more built-in adapters;
3. streaming multipart uploads so very large `.blend` files are not buffered in memory;
4. job history and restart recovery in the browser;
5. deformation metrics and focused weight-repair tools after a rebind.

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
4. ~~Add the preserved-scene Blender rebind worker and browser-launched job.~~
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
