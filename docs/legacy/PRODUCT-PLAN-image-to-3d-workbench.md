# Product plan: Image-to-3D workbench

**Decision date:** 2026-08-13  
**Starting point:** the existing viewer on port 8777  
**Primary backend:** the corrected TRELLIS.2 Mac stack rooted at `shivampkumar/trellis-mac`

## Goal

Turn the diagnostic viewer into one local-first workflow:

`source image -> material intent -> generation -> validation -> finishing -> GLB export`

The product should give a non-technical user a safe, repeatable result without exposing Metal,
BVH, UV, or glTF details. An advanced user should still be able to inspect the real mesh,
understand warnings, and make fast material corrections without rerunning generation.

The target is not pixel identity. A result must authentically represent the source, remain
visually solid with backface culling enabled, and read as the intended substance.

## Product principles

1. **Separate geometry from finishing.** Sampling and remeshing may take 10–60 minutes;
   material edits should take seconds. The UI must make that boundary unmistakable.
2. **Cache every expensive boundary.** Preserve the prepared input, raw decode, corrected mesh,
   UV atlas, textures, masks, and final GLB.
3. **Intent beats blind inference.** “Living vine,” “dead bark,” and “metal” can share pixels but
   need different PBR materials. Ask for intent and validate generated data against it.
4. **Diagnose before repairing.** Every automatic change needs a reason and an A/B preview.
   Never silently hide defects with double-sided rendering.
5. **Keep edits non-destructive.** A finish recipe creates a derived GLB. The corrected master
   and original baked textures remain unchanged.
6. **Make results reproducible.** Export a recipe JSON containing hashes, seed, revisions,
   generation settings, masks, diagnostics, and finishing values.
7. **Use honest defaults.** Opaque, single-sided rendering and backface culling are the default;
   neutral-grey, normals, and wireframe remain first-class inspection modes.

## User journey

### 1. Upload source images

The first image is required. Extra views are optional and must show the same subject in the
same pose. Preview the foreground matte and warn about opaque backgrounds, painted shadows,
cropped extremities, and very thin elements.

### 2. Choose material intent

Use a few understandable profiles instead of presenting raw PBR terminology:

| Profile | Examples | Starting behaviour |
|---|---|---|
| Living organic | vines, foliage, moss, fur | non-metal body, varied roughness, optional upward-facing moss |
| Dry organic | bark, bone, leather | non-metal, warm/dry palette, high roughness, optional relief |
| Matte mineral | stone, clay, concrete | non-metal, high roughness, restrained chroma |
| Glossy dielectric | ceramic, wet skin, resin | non-metal, low/medium roughness, no inferred relief by default |
| Metallic | steel, armour, gold | retain and validate metalness; environment-lit preview |
| Mixed material | creature with eye, jewellery, blade | body profile plus user-marked feature regions |

Onboarding questions:

- Is most of the subject metallic?
- Is it living organic, dry organic, mineral, glossy, or mixed?
- Are there important eyes, gems, blades, moss, or emissive regions?
- Is preserving thin detail more important than generation speed?

Profiles establish defaults; they do not lock the controls.

### 3. Generate

Show progress as distinct stages:

1. foreground and conditioning preparation;
2. sparse-structure sampling;
3. shape sampling;
4. texture-field sampling;
5. raw decode and cache;
6. corrected BVH remesh and cleanup;
7. simplification, UVs, and texture bake;
8. opaque, single-sided GLB export;
9. validation and profile diagnostics.

Classify costs before a user starts an action:

| Cost | Operations | Expected scale |
|---|---|---|
| Slow | diffusion and high-resolution decode | 10–60 minutes |
| Medium | remesh/simplify/UV/bake from cached decode | 2–20 minutes |
| Fast | material recipes, grades, masks, GLB JSON | under 10 seconds |

Generation and rebake require an exclusive GPU-job lock. Stream logs to the browser and allow
cancellation between stages.

### 4. Validate automatically

Validation produces a report, not one misleading score.

Geometry diagnostics:

- face and vertex counts;
- winding consistency and signed volume;
- boundary and non-manifold edges, labelled as diagnostics rather than scores;
- duplicate faces and degenerate triangles;
- opaque/single-sided flags;
- fixed front, rear, profile, and three-quarter culled snapshots;
- optional comparison with double-sided rendering to reveal hidden interiors.

Material diagnostics:

- albedo luminance and chroma distributions;
- metallic and roughness percentiles;
- alpha coverage and padding;
- profile conflicts, such as metalness near 1.0 on a living-organic subject;
- highlight compression relative to the source;
- a warning when a global grade cannot recreate spatial structure such as moss.

Warnings should recommend an action. Example:

> Living-organic profile conflicts with median metalness 1.0. Preview a non-metallic body
> repair. The generated albedo is also unusually dark; adjust preview exposure before altering
> the texture.

### 5. Finish interactively

Keep preview controls separate from exported changes.

**Preview only:** lighting, environment, exposure, background, camera, spin, source overlay,
textured/grey/normals/wireframe modes, culling, and synchronized A/B cameras.

**Exported material recipe:** body profile, temperature/chroma, albedo gain, metalness,
roughness, relief strength, moss colour/coverage/upward threshold, AO strength, and per-region
colour/roughness/metalness/emission.

The user can click a feature to create a region. The backend raycasts the point, expands a 3D
selection, rasterises its UV triangles, and displays the candidate mask in magenta before it is
accepted. This turns the current hand-entered `feature_mask.py` workflow into a guided tool.

Material changes display original and candidate at identical cameras and lighting. Slider input
is debounced into a fast request; releasing a slider creates an undoable recipe revision.

### 6. Export

Export the final GLB plus recipe JSON, diagnostic JSON, attribution/licence information,
thumbnail/contact sheet, and source hashes. Optionally include the corrected master and masks
for continued editing.

## Deterministic versus assisted behaviour

Deterministic:

- generation from the same input hashes, seed, revisions, and settings;
- corrected remesh/export defaults;
- diagnostics and warning thresholds;
- profile defaults;
- masks after a user supplies a point/radius or painted selection;
- application of a saved material recipe;
- artifact hashes.

Requires user/product intent:

- living vine versus dead bark;
- paint versus carved geometry;
- moss versus a green base material;
- eye versus an amber knot in wood;
- acceptable residual openings at the intended viewing distance.

The product converts those judgements into reproducible inputs. After the user chooses “living
organic,” marks the eye, and accepts a moss mask, another machine can replay it exactly.

## Proposed recipe

```json
{
  "schema_version": 1,
  "project_id": "snag-roots",
  "source_sha256": "...",
  "generation": {
    "backend": "trellis2-macos",
    "seed": 42,
    "pipeline_type": "1024_cascade",
    "steps": 12,
    "texture_size": 1024,
    "target_faces": 500000,
    "remesh": {"enabled": true, "band": 1, "project": 0}
  },
  "profile": "living_organic",
  "material": {
    "body": {
      "metallic": 0.0,
      "roughness": [0.62, 0.94],
      "chroma_strength": 0.55,
      "albedo_gain": 1.0,
      "normal_strength": 4.0
    },
    "moss": {
      "enabled": true,
      "up_threshold": 0.25,
      "coverage": 0.45,
      "colour": "#61752f",
      "strength": 0.55
    },
    "regions": [
      {
        "id": "eye",
        "mask": "masks/eye.png",
        "base_colour": "#d47a08",
        "metallic": 0.0,
        "roughness": 0.18,
        "emission": 0.05
      }
    ]
  },
  "diagnostics": "diagnostics.json",
  "software": {"workbench_revision": "...", "backend_revision": "..."}
}
```

Recipe validation rejects unknown keys, invalid ranges, atlas-size mismatches, and source/master
hash mismatches.

## Cache model

```text
projects/<project-id>/
  source/
  prepared/
  cache/
    generation-key/decode.pt
    geometry-key/master.glb
  masks/
  recipes/
  artifacts/
  diagnostics/
  renders/
  logs/
```

Cache keys are content hashes. The generation key includes input hashes, seed, pipeline, steps,
model revision, and conditioning. The geometry key adds remesh and face-budget settings. The
material key includes the master hash, recipe, and mask hashes. A lighting change invalidates
nothing; moss colour invalidates only the fast material artifact; target-face changes start from
the cached decode.

## Backend/API boundary

Keep the Three.js renderer and controls from `viewer/index.html`, but move asset and job state
behind an application API:

- `POST /api/projects` — create project;
- `POST /api/projects/{id}/sources` — upload source/view;
- `POST /api/projects/{id}/generate` — enqueue GPU job;
- `POST /api/projects/{id}/rebake` — rebuild from cached decode;
- `POST /api/projects/{id}/finish` — apply material recipe;
- `POST /api/projects/{id}/regions` — raycast/rasterise a feature mask;
- `GET /api/jobs/{id}` and `/events` — status and streamed logs;
- `GET /api/projects/{id}/diagnostics` — structured report;
- `GET/PUT /api/projects/{id}/recipe` — retrieve or revise recipe;
- `POST /api/projects/{id}/export` — freeze the final bundle.

The first release remains bound to `127.0.0.1`. Upload names never become filesystem paths;
every artifact path must resolve inside its project root.

## Interface

Use four stages:

1. **Source** — upload, alpha preview, extra views, profile, quality.
2. **Generate** — progress, live log, timings, cancellation, cache status.
3. **Inspect** — source comparison, culled/grey/normals/wireframe, diagnostics.
4. **Finish** — profiles, masks, sliders, A/B revisions, export.

Persistent status labels every action as `preview only`, `fast material update`, `rebake from
cache`, or `full generation` before it runs.

## Tests and acceptance fixtures

Unit-test the recipe schema, profile defaults, cache invalidation, diagnostics, mask
rasterisation, deterministic material application, geometry-buffer preservation, and safe path
handling. Integration-test upload/job/event/artifact flow, cancellation, cached rebakes, recipe
revisions, and final bundles.

Acceptance fixtures:

- **Moss Fox:** rear head stays solid under culling;
- **Forest Variant:** thin blades survive and the rear does not collapse;
- **Snag:** 27.9M raw-face stress fixture completes; body is not falsely metallic; spatial moss
  and an amber/glossy eye survive recipe replay; residual openings remain visible in diagnostics;
- Hugging Face GLBs remain capability controls, not pixel targets.

Visual regressions must cover front, rear, both profiles, and three-quarter views in textured and
culled-grey modes. Topology metrics cannot replace those images.

## First build order

1. Extract the viewer renderer into reusable modules without changing behaviour.
2. Add project directories, recipe schema, and content-hash cache keys.
3. Add upload/project APIs and a job runner around the CLI and `trellis_rebake.py`.
4. Stream logs and expose the slow/medium/fast boundary.
5. Persist geometry/material diagnostics as JSON.
6. Add profiles and deterministic fast material application.
7. Add preview exposure/lighting and exported material sliders with A/B revisions.
8. Add click-to-mask regions, beginning with eyes.
9. Add spatial organic masks such as upward-facing moss.
10. Run Fox, Forest, and Snag acceptance fixtures before extracting a public repository.

## Out of scope for the first build

- cloud accounts, remote queues, and multi-user permissions;
- automatic semantic segmentation of every region;
- guaranteeing formal watertightness;
- replacing TRELLIS inference before parity benchmarks;
- publishing this entire laboratory repository as the Mac port.

The eventual public repository should be a clean, pinned, fully credited Mac pipeline and
workbench with upstream attribution, dependency revisions, patch history, fixtures, tests,
metrics, and an explicit account of what remains heuristic.
