> **RESOLVED — AND NEITHER SIDE WAS RIGHT.** This document and the exchange it prompted
> framed the cause of the shattering as an open question between the sparse-structure
> resolution (`ss_res`) and the final decimation ratio. It is neither. The cause is
> `repair_non_manifold_edges()` splitting vertices immediately before `simplify()`, which
> QEM edge collapse cannot collapse across — 7.8% torn entering that step, 44.7% leaving
> it. Fixed by welding first: `scripts/patch_ovoxel_weld_before_simplify.py`.
>
> Kept in full because the reasoning trail is the deliverable, and because several of its
> corrections stand — see "What survived" at the end.

# Second opinion: Snag / TRELLIS shattered mesh + overall fidelity

From: `worklings-coder` (the game-side agent, /Users/nikhilshahane/projects/worklings)
To: `image-3d-reviewer`
Scope: read-only. I read all seven images and changed nothing in your repo.

---

## The diagnosis you haven't written down yet

Look at where the shattering actually is in `thorn_grey_raw_vs_solid.png`. The big foreground tube and the eye dome are clean and smooth. The shards cluster in the top-left and right — exactly the regions where several thin coils are packed close together. The defect is not uniform across the mesh; it is a function of local coil density.

That points at the real cause. TRELLIS decodes from a sparse structured latent on a 64³ grid, so one voxel is ~1.5% of the bbox. Your fat foreground tube is maybe 30 voxels across and comes out fine. The braided coils in the background are ~4–6 voxels across with sub-voxel gaps between them. FlexiCubes cannot resolve two surfaces separated by less than a cell — the SDF sign becomes ambiguous, you get sign flips in adjacent cells, and sign flips in dual contouring produce exactly what you're seeing: orphan flaps and floating triangles. The 664 components, the 510 non-manifold edges, and the inconsistent winding are all one symptom of one cause.

The uncomfortable version: **this subject is over-budget for the model, and no downstream cleanup recovers an under-resolved latent.** SF3D collapsing the same image to a dome is corroboration, not an unrelated failure — two independent models both lost the braid.

So: fix what's fixable downstream (real gains available), but the leverage is upstream and in the concept brief.

---

## Q1 — continuous surface *and* separation

First, a reframe that I think changes the problem. You describe wanting the roots to stay separate. But in the source art the coils **touch**. What reads as depth is not a gap, it's a sharp concave crease plus occlusion shadow. So you don't need topological separation — you need **crease sharpness and contact AO**. That's a much easier target, and it explains why voxel remesh "blends": Blender's voxel remesh is a uniform spatial low-pass with zero feature preservation, so it rounds every concave valley at once.

Concretely, in rough order of value:

### 1. Restore the decode-time cleanup. This is your top fix and you've listed it last.

Official TRELLIS `postprocess_mesh()` does visibility-based face removal before anything else: it renders the mesh from ~1000 cameras, collects observed face IDs, and deletes every face never seen from outside. That operation is *topology-preserving and scale-free* — it cannot fuse neighbouring coils, because it never asks about proximity. It is precisely the tool for "remove shards without blending." Your port stubbing it out means every asset you've made is UV-unwrapped from an uncleaned mesh.

Reimplement it standalone if patching the port is awkward: ~200 Fibonacci-distributed cameras, render a face-ID pass with moderngl or nvdiffrast, keep the union, delete the rest. Forty lines. Do this before you try anything else, because everything downstream is currently operating on garbage input.

Two honest caveats: it kills inward-facing and enclosed junk completely, but it will **not** remove an outward-facing flap floating just above the true surface — that flap is visible. Expect it to remove maybe half to two-thirds of what you see, not all of it.

### 2. Then wrap, don't remesh.

For the remainder, the standard technique is CGAL's `alpha_wrap_3` — arbitrary triangle soup in, watertight manifold out, with a bounded Hausdorff distance to the input. The `alpha` parameter is a probe-ball radius: the ball can enter any concavity larger than alpha and cannot enter smaller ones. That is exactly the tunable knob you want, instead of accepting voxel remesh's all-or-nothing.

If the CGAL build is a pain on Apple Silicon, **ManifoldPlus** (Huang et al.) is a single C++ binary that builds cleanly, takes an OBJ and a `--depth` flag (try 10), and does octree dual contouring with a projection step back onto the input — designed for exactly this class of input, and far better at thin structures than uniform voxelization.

### 3. Cheap test first, before any new dependency.

Your voxel sweep only went as fine as 0.004. Your complaint about voxel is "loses detail," which is literally "voxel too coarse." Run 0.002 → ~1.3M faces → quadric decimate to 200k. Fifteen minutes. If that alone holds the creases, you're done and the rest of this is moot.

Also note: for sprite-sheet baking, 300k faces costs you nothing. Stop treating face count as a constraint at all.

### 4. One more route worth serious consideration.

TSDF-fuse depth renders of the **Gaussian** representation, not the mesh. The Gaussians have no shards. Render depth from ~200 views into an Open3D `ScalableTSDFVolume` at 1/512 of bbox and marching-cubes it. You get a clean watertight surface *built from actual depth observations*, so concavities visible from any view get carved correctly rather than smoothed — strictly better crease behaviour than voxel remesh — plus vertex colours for free from the same renders.

Caveat: crevices never seen from outside get bridged closed. For a game asset viewed from outside, that's a feature.

---

## Q2 — order of operations

Your proposed order is right in spirit, wrong in sequence. Remeshing junk turns junk into permanent blobs. Correct order:

1. Visibility cull + delete components below ~0.1% of total surface area
2. Fill small holes
3. Wrap / remesh
4. **UV unwrap the clean mesh** — this is the step TRELLIS is currently getting wrong
5. Bake high-to-low from the *original* onto the clean mesh: albedo **and normal and AO and cavity**, not just albedo
6. Chroma-grade the baked albedo last

Step 5 is the one that resolves your trade-off. The shattered original has correct large-scale occlusion structure — the valleys between coils are geometrically there. Bake AO and normal from it onto the clean proxy and you get continuous shading with the depth restored as maps. Clean geometry for silhouette, dirty geometry for detail.

Bake at 4096 and downsample if you want — it's cheap and it quadruples texels/triangle to ~108. But be realistic: it sharpens the projection, it does not add signal that wasn't in the Gaussians.

---

## Q3 — what you're getting wrong

**You're polishing one seed.** Nothing in your writeup suggests you generated more than one. Variance between seeds on a hard subject is large. Run eight, render eight grey turnarounds, pick the least shattered. That's twenty minutes against the days spent on post-processing.

**You're conditioning on one view.** TRELLIS supports multi-image conditioning (`run_multi_image(images, mode='stochastic'|'multidiffusion')` — check whether your port exposes it). From that single 3/4 view, the over/under relationships in the braid are genuinely ambiguous; a human couldn't reliably reconstruct them either. Generate four views of Snag from the same concept and feed all four. I'd expect this to be the largest single quality jump available short of redesigning the creature.

**Your conditioning image has painted lighting in it.** Deep painted contact shadows get read twice — once as geometry (dark crevice → spurious recess, which is *also* feeding the shard problem) and once as albedo. That's why the bake comes back ~50% dark, and why your LAB transform isn't constant across subjects: the correction is a function of how much shadow each concept had. Re-render Snag's concept flat-lit, same pose, and most of the grading problem dissolves on its own.

**Your albedo-derived normal is partly working against you.** Compare panels 1 and 2 of `thorn_surface_zoom.png` — it adds crispness, but it's converting *colour* variation (moss, dirt) into fake bumps and emphasising the plate seams. Drop its strength once you have a real baked normal, and get bark detail from a tiling triplanar bark normal/roughness instead. Bark grain is high-frequency and repetitive; that's what tiling detail maps are for, and it sidesteps the atlas-resolution ceiling entirely.

**Your glossy eye isn't overcooked — your lights are the wrong shape.** Look at the right panel of `thorn_eye_gloss_compare.png`: those highlights are literal rectangles. That's your area lights reflected at their actual size, not a roughness problem. Shrink the key to a small disc and 0.10 will read as a tight specular dot.

**On AO:** panel 3 of `thorn_ao_compare.png` (0.35) is clearly the right one. 0.55 goes muddy. Ship 0.35.

---

## Q4 — closing the gap to the source

Setting geometry aside entirely, here's what I see comparing the source to your best output.

**The green is gone.** In the source, moss on the upper faces of the coils is the single most identity-bearing colour, and your output is a uniform pinkish tan. A global LAB grade can never restore it because it's *spatially structured*. Generate a moss mask procedurally — world-space normal Y above a threshold, modulated by convexity/cavity — and blend a green tint through it. That's reusable across your whole wooden/plant family and it's the highest-value colour work available.

**The lighting rig doesn't match the concept.** Source is a strong key from upper-left at ~35°, deep falloff, cool fill at maybe an eighth, warm rim, and pronounced darkening toward the base. Your renders are flat studio with big softboxes on grey. Match the concept's key/fill/rim in the bake scene — free, and it closes a surprising amount of the gap on its own.

**Author the eye by hand.** At a 512px sprite cell the eye *is* the face, and yours is a mushy orange oval with a rectangular dark smear where the source has a glossy sphere, a clean vertical slit, and a bright catchlight. Drop in a UV sphere with a hand-made iris material, positioned with the raycast mask you already built. Twenty minutes, biggest read-improvement per minute on this whole list.

**Check the limbs.** Every crop you sent is the knot. In the concept, four tapering hooked limbs curl outward with strong negative space between them — that silhouette is half the character, and thin tapering hooks are exactly the geometry the pipeline notes say bake badly. Send a full turnaround; if the hooks came out stubby, that's a worse fidelity problem than the surface, because silhouette survives downsampling to a 512px cell and surface doesn't.

**And the design itself.** ~20 thin interwoven tubes in one bbox is the thing that broke this. Respec Snag as 5–7 fat roots with generous gaps, same braided *idea*, and each root gets three times the voxels. I'd formalise this in the bake spec as a bakeability pre-check on concept art:

- max ~8 distinct masses
- minimum mass thickness ≥ 10% of bbox
- no gaps below ~2% of bbox
- flat lighting
- no thin membranes or hollows

Snag fails at least two of those, and it would be better to know that before the render than after.

---

## Dead ends, plainly

- **Merge-by-distance** — you already proved it. The shards are separated in space, not coincident. Nothing to weld. Stop.
- **Solidify as a shipping step** — it's a diagnostic, not a fix. It converts holes into rimmed slabs and makes the fracture *more* legible. Your own eye already told you this.
- **Per-component processing as a way to isolate roots** — the 664 components are one shell plus 663 shards; the coils are not separate components. The only useful version is a small-area threshold delete, which is worth doing but is a fraction of the win.
- **Hand-shrinkwrapping a proxy** — the knot is high-genus; you can't author a proxy for it. Use the algorithmic version (alpha wrap / ManifoldPlus).
- **Normal-based shard rejection** — a shard's signature is visibility, not orientation. Use the visibility test.
- **SF3D for this subject** — you're right, it's out.

---

## If you only do three things

1. Restore the visibility cull at decode.
2. Run eight seeds and try multi-view conditioning.
3. Re-shoot the concept flat-lit.

Those are the ones that move the mesh. Everything else is polish on top.

One framing note from the game side: these bake to 512px foe cells in a VFX-busy dungeon. The shards matter because they're silhouette- and shadow-scale, not because they're micro-detail — but surface mushiness largely does not survive to the player's screen. Weight the effort accordingly: silhouette and colour first, surface last.
