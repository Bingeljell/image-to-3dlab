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

## Workshop file contract

The portable result is a GLB plus a versioned sidecar during authoring:

```text
creature.glb
creature.workshop.json
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

1. Refactor the existing Compare and Generate application without behavior changes.
2. Build the first Animate slice: clip discovery, playback, scrubbing, skeleton overlay,
   bone selection, and pose reset.
3. Add browser keyframes and versioned animation recipes.
4. Add prompted animation against the validated recipe schema.
5. Add the Blender production-bake job.
6. Add Rig as a browser-launched Blender bind job.
7. Add Look material controls, mask painting, and non-destructive sculpt layers.
8. Consolidate export and workshop project persistence.

## Animation questions to resolve

These are product decisions for the animation discussion, not blockers for the viewer
refactor:

- Is browser-authored FK motion expected to be final, or primarily a fast previs that is
  always eligible for Blender rebaking?
- Which first rig profile is canonical: the existing Rigify Basic Quadruped setup, a
  project-specific quadruped derivative, or a smaller exported runtime skeleton?
- Should the first LLM surface generate whole clips, edit selected time ranges, or choose
  and parameterize trusted procedural motions?
- Which motions prove the product: idle, walk/trot, headbutt/attack, look-at, or a custom
  prompt?
- Does v1 need foot locking and terrain contact, or is FK posing plus baked clips enough?
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
