# scripts/

Every command-line tool in the pipeline, grouped by what you are trying to do. One
script, one job: nothing here is imported by `image_to_3dlab/`, and anything worth
reusing was pulled out as a module-level function so it can be tested without Blender
or a GPU.

Each summary below is the script's own first line. `tests/test_scripts_registry.py`
fails if a script is missing from this file or if a summary has drifted from the
docstring, so a new script is not finished until it is listed here.

Most tools print their real documentation with `--help`; the docstring at the top of
each file explains *why* it exists, which is usually the part you need.

## Generate an asset

Run a backend end to end, or re-run part of one without paying for the whole thing again.

For the staged **fitted metarig → proxy weights → reference pose → tunable gait →
standing transition** workflow, start with [the quadruped pipeline guide](../docs/quadruped-pipeline.md).
It includes commands, parameter definitions, failure checks, and an LLM-assisted
tuning checklist. Manual fitting and visual weight/deformation review remain required.

| Script | What it does |
|---|---|
| `blender_quadruped_pipeline.py` | Bind, capture and animate fitted Rigify quadrupeds with tunable profiles and audit reports. |
| `quadruped_gait.py` | Gait curves for the Rigify basic quadruped metarig, as pure functions. |
| `blender_quadruped_walk.py` | Author a natural walk, trot or scamper on any Rigify basic-quadruped rig. |
| `bootstrap_macos.sh` | Create the project virtualenv and install the SF3D backend and its native dependencies (needs Python 3.10/3.11 and Homebrew's libomp). |
| `bootstrap_trellis_macos.sh` | Clone and install the `shivampkumar/trellis-mac` port, with the Metal acceleration backends when Xcode's Metal compiler is present and a slower CPU bake fallback when it is not. |
| `bootstrap_trellis_space_macos.py` | Bootstrap TRELLIS.2 on macOS from Microsoft's pinned Space source. |
| `bootstrap_sf3d.py` | Install Stable Fast 3D: its code and compiled extensions, then its gated weights. |
| `bootstrap_pixal3d.py` | Install Pixal3D (raven38/pixal3d.cpp): a `trellis-cli` build plus its Q8_0 weights. |
| `bootstrap_qwen_image.py` | Install the text-to-image route: a stable-diffusion.cpp binary and Qwen-Image weights. |
| `pixal3d_generate.py` | End-to-end Pixal3D generation: image -> textured GLB, on a Mac or an NVIDIA card. |
| `trellis_space_generate.py` | Full image -> GLB generation through the CLEAN `trellis-space-mac` port on Apple Silicon. |
| `hunyuan_mlx_generate.py` | End-to-end Hunyuan3D-MLX generation: image -> textured GLB. |
| `hunyuan_mlx_xiong_generate.py` | End-to-end, single-repo Hunyuan3D-MLX generation: image -> textured GLB. |
| `hunyuan_shape_octree_test.py` | Shape-only generation at a given octree_resolution, with visible progress. |
| `trellis_stage3.py` | Resample only TRELLIS.2's Stage-3 material field on a cached shape latent. |
| `trellis_rebake.py` | Re-bake a GLB from a cached decode, without re-sampling or loading the model. |
| `export_decode_highpoly.py` | Export a cached decode as a high-poly PLY, to bake detail from. |
| `runpod_trellis2_cuda_probe.py` | Run a frozen-shape TRELLIS.2 Stage-3 material probe on CUDA. |
| `runpod_trellis2_cuda_requirements.txt` | Pinned CUDA wheels for the RunPod control run; not used by any local path. |

## Before you spend a run

A generation run costs 15-20 minutes. These cost seconds and are worth it first.

| Script | What it does |
|---|---|
| `classify_trellis_input.py` | Advisory TinyCLIP check for TRELLIS-hostile flat/vector-style inputs. |
| `soften_markings.py` | Reduce the contrast of flat painted markings in a conditioning image. |
| `check_trellis_space_attention.py` | Cheap MPS integration gate for TRELLIS sparse self/cross attention. |
| `check_trellis_space_dino.py` | Capture the exact preprocessed image and DINO conditioning tensor. |
| `audit_model_weights.py` | Find model weights stored twice, and optionally reclaim the duplicates. |

## Repair the mesh

Everything here operates on a GLB or a cached decode and is headless.

| Script | What it does |
|---|---|
| `repair_decode.py` | Repair winding on a cached decode, before `to_glb` ever sees it. |
| `fix_winding.py` | Make a generated mesh's faces point outward. |
| `mesh_weld.py` | Weld vertices that share coordinates exactly, and drop the faces that collapse. |
| `mesh_health.py` | Measure the structural health of a generated mesh, and optionally repair it. |
| `fill_holes.py` | Close small holes in a generated mesh, leaving large openings alone. |
| `remove_loose_parts.py` | Delete disconnected junk from a generated mesh, keeping the texture intact. |
| `visibility_cull.py` | Pure helpers for visibility-based face culling: keep only what is seen from outside. |
| `remesh_to_target.py` | Decimate a mesh to a target face count via fast_simplification, with timing. |
| `crop_mesh.py` | Cut a region out of a mesh at full density, so it can be judged by eye. |

## Texture, colour and material

The asset's surface, as opposed to its shape.

| Script | What it does |
|---|---|
| `paint_eyes.py` | Repaint a generated head's eyes into a dedicated high-resolution patch. |
| `colour_match_albedo.py` | Grade a generated GLB's albedo toward the colour of its source concept art. |
| `lift_lightness.py` | Brighten a GLB's albedo without shifting its colour. |
| `surface_detail.py` | Give a generated asset a surface: normal relief and roughness variation. |
| `attach_normal_map.py` | Attach a baked normal map to a GLB's material. |
| `restore_pbr_material.py` | Re-attach the metallicRoughness map that `--material-mode matte` orphaned. |
| `fix_glb_opaque_material.py` | Make an existing TRELLIS GLB opaque and single-sided without rebaking it. |
| `compress_glb_textures.py` | Re-encode a GLB's textures, without touching its geometry. |
| `retopo_repaint.py` | Finish a generated asset: retopologise, repaint, compress. |
| `living_organic_material.py` | Apply a reproducible living-organic material recipe to an existing GLB. |
| `project_labels.py` | Project a 2D image onto a generated mesh as per-vertex colours. |
| `project_markings.py` | Paint the source image's markings back onto a generated mesh's texture. |
| `feature_mask.py` | Build a UV-space mask for one feature of a mesh, from a point in 3D. |
| `bake_stiffness.py` | Bake foliage stiffness into a GLB as vertex colours, for engine-side wind. |
| `classify_thickness.py` | Separate solid body from thin foliage by measuring local thickness. |

## Measure and judge

Numbers and renders to decide whether a change helped. Read `docs/` before trusting an old one.

| Script | What it does |
|---|---|
| `compare_to_source.py` | Put a generated asset next to the image it was generated from. |
| `mark_asset.py` | Record a human verdict on a generated asset, so winners stay findable. |
| `ribbon_metric.py` | Measure how torn a generated mesh is, and gate post-processing on it. |
| `tear_provenance.py` | Ask of every large tear: could the input view see it?. |
| `glb_forensics.py` | Dump what a GLB actually contains, so a reference asset can be diffed against ours. |
| `mesh_checkpoint_forensics.py` | Measure exact-position topology in an o_voxel `.pt` geometry checkpoint. |
| `replay_cleanup_checkpoints.py` | Replay o_voxel cleanup on an exact mesh checkpoint, saving every substage. |
| `diagnose_remesh_kernel.py` | Isolate why Metal narrow-band DC remeshing emits a wireframe instead of a surface. |
| `measure_bvh_on_surface.py` | Check MtlBVH on an invariant that scales with the real production mesh. |
| `measure_bvh_precision.py` | Measure how accurate MtlBVH's unsigned_distance actually is, against exact ground truth. |
| `xatlas_timing_probe.py` | Time xatlas.parametrize in isolation, to characterize its face-count scaling. |

## Blender: look at it

These talk to a running Blender over the `execute_code` socket on port 9876 unless they say headless.

| Script | What it does |
|---|---|
| `blender_stage.py` | Put assets side by side in the RUNNING Blender, lit, textured and ready to look at. |
| `blender_stage_bake.py` | Render a full frame sequence of a named Action at the locked dungeon-stage camera angle. |
| `blender_render_asset.py` | Import a GLB into the local Blender MCP scene and render cardinal previews. |
| `blender_turntable.py` | Render a seamlessly looping 360-degree turntable of a GLB via the Blender MCP scene. |
| `blender_light_orbit.py` | Render a GLB under an orbiting key light, holding the camera still. |
| `blender_lineup_sweep.py` | Render a camera sweep past a line-up of assets, headless. |
| `blender_orbit_camera.py` | Point an orbit camera at a target and (optionally) render. |
| `blender_inspect.py` | Inspect a Blender scene or an unopened .blend file's contents. |
| `blender_pick_pixel.py` | Raycast one render pixel into the asset currently staged in live Blender. |
| `blender_import_character.py` | Append a rigged character (armature + skinned mesh) into the currently live Blender scene, cleaned up and positioned. |
| `blender_wind_demo.py` | Animate labelled foliage with shader-style wind and render it to MP4. |
| `blender_agent.py` | Let a local LLM stage a scene in the running Blender, by calling a small set of tools. |

## Blender: change the geometry

Heavier edits that need Blender's own operators rather than trimesh.

| Script | What it does |
|---|---|
| `blender_fix_normals.py` | Recalculate outward-facing normals on a generated mesh and re-export it. |
| `blender_solidify.py` | Give a generated mesh's zero-thickness sheets real thickness, headless. |
| `blender_solidify_bake.py` | Voxel-remesh a mesh into a closed surface and bake its albedo back on. |
| `blender_voxel_remesh.py` | Rebuild one closed skin from the decode, using Blender's voxel remesh. |
| `blender_shrinkwrap.py` | Pull a clean remeshed surface back onto the original decode. |
| `blender_visibility_cull.py` | Delete every face never seen from outside the mesh, headless. |
| `blender_retopo_bake.py` | Quad-retopologise a generated mesh and transfer its texture onto the clean topology. |
| `blender_reunwrap_bake.py` | Re-unwrap a generated mesh into coherent UV islands and re-bake its texture, headless. |
| `blender_split_regions.py` | Split a generated mesh into per-region material slots, each with its own texture. |
| `blender_bake_ao.py` | Bake an ambient-occlusion map from an asset's own geometry, headless. |
| `blender_bake_normals.py` | Bake TRELLIS' discarded high-poly detail into a normal map for the low-poly mesh. |

## Rig and animate

The `*_pose.py` files are pure curve maths with no `bpy`, which is why they have tests.

| Script | What it does |
|---|---|
| `blender_joint_markers.py` | Spawn named joint markers on a mesh in Blender, and read their placed positions back. |
| `blender_build_rig.py` | Build a quadruped armature from placed joint markers and bind the mesh to it. |
| `blender_voxel_weights.py` | Weight a generated mesh via a watertight voxel proxy, then transfer the result back. |
| `blender_voxel_weights_code.py` | The Blender-side program for `blender_voxel_weights.py`. |
| `blender_rebind.py` | Apply browser fit-joint corrections to an opened prepared Blender scene. |
| `blender_bind_rig.py` | Generate a Rigify rig and bind a mesh to it through a watertight voxel proxy, headless. |
| `blender_rebind_weights.py` | Voxel-proxy weight transfer used by the headless rig rebind worker. |
| `blender_export_rig_binding.py` | Prepare an open Blender metarig scene and export its browser rig sidecar. |
| `rignet_infer.py` | Run vendored RigNet inference on one of our own generated meshes. |
| `attack_pose.py` | Pose curves for a quadruped slam attack, as pure functions. |
| `rigify_walk_pose.py` | Pose curves for a quadruped trot on a Rigify-generated rig, as pure functions. |
| `blender_walk_cycle.py` | Author a looping quadruped gait cycle on the rigged fox in the live Blender scene. |
| `blender_rigify_walk_cycle.py` | Author a trot cycle on the Rigify-generated 'rig' armature in the live Blender scene. |
| `blender_idle_cycle.py` | Author a looping idle for the rigged fox. |
| `blender_attack_cycle.py` | Author a slam-attack clip on a rigged quadruped in the live Blender scene. |

## Vendor patches

`vendor/` is git-ignored, so every fix to someone else's checkout lives here as a re-appliable patch script. Each one asserts its anchor and is idempotent; re-running after a bootstrap is the intended workflow.

| Script | What it does |
|---|---|
| `patch_trellis_space_core.py` | Apply the minimal inference-preserving Mac patch to a Microsoft Space tree. |
| `patch_trellis_metal_backends.py` | Replay proven image-to-3dlab fixes over pinned Pedro Metal backends. |
| `patch_trellis_face_cap.py` | Lift the hard 200,000-face cap in the Mac port's `generate.py`. |
| `patch_trellis_highpoly.py` | Export TRELLIS' full-resolution mesh before it is simplified away. |
| `patch_trellis_quality.py` | Expose the quality controls TRELLIS already has and the Mac port never passes. |
| `patch_trellis_multiview.py` | Teach the vendored TRELLIS.2 to condition on several views of one subject. |
| `patch_trellis_enable_cleanup.py` | Re-enable the decode-time mesh cleanup that `mps_compat.py` turns into no-ops. |
| `patch_trellis_dump_decode.py` | Teach `generate.py` to cache the decoded mesh, so baking can be re-run without sampling. |
| `patch_pixal3d_rembg.py` | Stop Pixal3D loading BRIA RMBG-2.0, before it ever downloads it. |
| `patch_pixal3d_model_subset.py` | Let Pixal3D load only the checkpoints a run actually needs. |
| `patch_pixal3d_low_vram.py` | Make Pixal3D's low-VRAM mode reachable, via `PIXAL3D_LOW_VRAM=1`. |
| `patch_sf3d_cpu_baker.py` | Let SF3D's texture baker run on the CPU while the model runs on an NVIDIA GPU. |
| `patch_trellis_no_bria.py` | Disable TRELLIS' configured background model for license-controlled runs. |
| `patch_trellis_mlx_attention.py` | Add an `mlx` sparse-attention backend to a vendored TRELLIS.2 checkout. |
| `render_glb_comparison.py` | Render several GLBs from one fixed camera and lay them out as a comparison image. |
| `patch_ovoxel_pack_options.py` | Let `o_voxel.postprocess.to_glb` forward xatlas packing options. |
| `patch_ovoxel_opaque_material.py` | Match the official TRELLIS GLB's opaque, single-sided material flags. |
| `patch_ovoxel_weld_before_simplify.py` | Weld coincident vertices before every `simplify()` in o_voxel's `to_glb`. |
| `patch_ovoxel_skip_spurious_fill.py` | Use the measured-safe cleanup order for Metal remesh output. |
| `patch_ovoxel_remesh_checkpoints.py` | Add opt-in exact geometry checkpoints to o_voxel's remesh branch. |
| `patch_metal_hashmap_miss.py` | Fix the unchecked hashmap miss in the Metal dual-contouring kernel. |
| `patch_mtlbvh_production_traversal.py` | Apply the production-scale MtlBVH traversal and dispatch-lifetime fixes. |
| `patch_rignet_macos_compat.py` | Make the vendored RigNet checkout run inference on macOS / Apple Silicon. |
| `rebuild_metallib.sh` | Recompile and install cumesh's Metal shader library after patching a `.metal` source, without rebuilding the Obj-C++ extension. |
| `rebuild_mtlbvh_metallib.sh` | Recompile and install only MtlBVH's Metal shader library after editing `bvh.metal`. |
| `rebuild_mtlbvh_native.sh` | Rebuild, install and ad-hoc sign MtlBVH's native extension. Stop every Python worker first: overwriting a loaded Mach-O image makes macOS kill them. |
