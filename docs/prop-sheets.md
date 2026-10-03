# Prop sheets: many props from one image, in one run

![A generated 3x3 sheet of medieval props, and the nine separate game-ready props made from it](images/prop-sheet-one-image-nine-props.jpg)

A **prop sheet** is one image holding a grid of separate props: barrels, crates, a chest.
Pixal3D turns the whole sheet into 3D in a single run, and two scripts take it apart into
separate, upright, named props with game-ready levels of detail.

Nine props took 14 to 16 minutes on an M5 MacBook with 16 GB, under two minutes a prop.

It is for **props**: things that stand on their own and read from any side. Characters
want the detail of a run to themselves.

## 1. Make the sheet

Use **Generate Image** in the viewer, or bring your own picture. This prompt made the
sheet above (Qwen-Image 2.1, default settings, 768x768, about 3 minutes on the same Mac):

```text
A 3x3 grid of nine separate medieval fantasy game props, evenly spaced with generous
empty space between them, each one isolated and not touching any other: a wooden
barrel, a wooden crate, an iron-bound treasure chest, a clay pot, a wooden bucket, a
burlap grain sack, a small wooden stool, an iron anvil, a tree stump. All props at a
similar size, all shown from the same slightly elevated three-quarter view, stylized
hand-painted game asset style, soft even studio lighting, plain white background, no
ground shadows, no text, no labels, no grid lines, no borders
```

What matters in it:

- **Space between props.** Props that touch or overlap in the picture are split as one.
- **Similar sizes.** Pixal3D drops parts under 3% of the largest one as crumbs, so a
  much smaller prop can be dropped with them.
- **A slight view from above is fine, and better.** Pixal3D sees the tops and models
  them. The split step undoes the tilt this causes (see step 3). An eye-level prompt was
  tried: tops it could not see came back invented, and less level (a stool seat 9.4
  degrees off, against 2.2).

Qwen's licence is a bit ambiguous, and the props made from its pictures share that.
Qwen says the pictures you generate are yours
([their statement](https://x.com/QwenDevs/status/2101917379785838660)), but the licence
still says the model is for non-commercial use. Our reading is that commercial work needs
a licence from Qwen; check it yourself if you plan to. The pipeline keeps those runs in
`research_only` and says why in the sidecar. Bring your own image and none of that applies.

## 2. Turn it into 3D

Pixal3D, default settings:

```bash
python scripts/pixal3d_generate.py sheet.png output/sheet/sheet.glb --res 1024
```

Keep the default field of view (20 degrees). A near-orthographic 5.7 degrees was tried,
to match how image models draw: the tilt came out more even across the grid, but box
corners came out less square (up to 5.3 degrees off, against 2.7). Neither helped
overall.

## 3. Split it into props

```bash
blender -b --factory-startup --python-exit-code 1 -P scripts/blender_split_props.py -- \
    output/sheet/sheet.glb output/sheet/split \
    --names barrel crate chest clay_pot bucket grain_sack stool anvil tree_stump
```

Names follow reading order: top row first, left to right. Each name becomes a file, so
two names that differ only in case are refused. Props past the end of the list are
numbered. Out come `props.blend` (everything lined up), one GLB per prop with its origin
at the bottom centre, and `props.json`, a record of what was done to each prop. It
takes seconds.

**Why the props need standing up.** Pixal3D's camera is level, and the sheet is drawn from
a little above, so Pixal3D tips each prop back to show its top to a level camera: 22 to
34 degrees on this sheet. That is a rotation, not a distortion. Once undone, tops and
bases sat within 3.3 degrees of level, box corners within 2.7 degrees of square, and
round props measured as deep as they are wide (0.97 to 1.03).

**Why some props get turned.** A crate drawn corner-on comes back rotated 41 degrees
about the vertical. It reads as warped until it is squared up. Box-like props are
turned to face the front. Round and soft ones are left alone, since turning them would
swing their painted front away.

**Check the ties.** A box drawn almost exactly corner-on is a coin toss between its
front and its side. The script flags any turn near 45 degrees. On this sheet the chest
came out with its lock facing sideways, and `--turn chest=90` fixed it. A `--turn` for a
name that isn't there stops the run, so a typo can't slip past.

## 4. Finish each prop

```bash
python scripts/finish_props.py output/sheet/split output/sheet/finished
```

Each prop gets three levels of detail (LODs), lighter copies a game swaps in with
distance: 5,000, 2,500 and 1,000 triangles by default (`--lods`). Each has its own
1024 texture, plus the two maps Finish bakes from the original: a normal map, which
carries the surface relief the triangles no longer have, and its metallic-roughness
map, so iron stays dark metal and wood stays matte. The nine props took 3 minutes.

Those counts are targets. On every prop here the bake landed on them exactly, but when
Blender's quad remesher (QuadriFlow) takes a prop, it makes quads, and a LOD can hold up
to twice as many triangles. `finish_props.json` records what each file really holds.

**Every LOD is baked from the original.** Simplifying LOD0 down to 1,000 triangles with
meshoptimizer stretched the texture across its seams, and the barrel's iron hoops came
out blotched. A normal map re-baked for that mesh did not help. Re-baking from the
original came out clean on all nine. The cost is one texture per LOD instead of one
shared.

**Then [gltfpack](https://github.com/zeux/meshoptimizer), for size.** When it is on PATH
(or passed with `--gltfpack`), each LOD also gets a compressed `.web.glb`, with the mesh
compressed and textures in WebP. The chest's LOD0 went from 3.7 MB to 495 KB. Those
files use `KHR_mesh_quantization`, `EXT_meshopt_compression` and `EXT_texture_webp`:
three.js reads all three, but check your engine. Use a native release build of
gltfpack, since the npm build cannot write WebP. Both kinds of file pass the Khronos
glTF validator with no errors and no warnings, and each LOD's node is named after it
(`chest_LOD0`).

`--resume` keeps the LODs already on disk, and refuses if they were baked with other
settings.

| Prop | LOD0 | LOD1 | LOD2 |
|---|---|---|---|
| barrel | 350 KB | 328 KB | 303 KB |
| crate | 285 KB | 253 KB | 233 KB |
| chest | 495 KB | 454 KB | 432 KB |
| anvil | 190 KB | 206 KB | 172 KB |

## From the viewer

The **Props** tab runs steps 3 and 4 in one go. Pick the sheet's GLB from what
Generate 3D made, or upload one; type the names one per line, in reading order; press
**Split & finish**. The panel shows the split, then one row per prop counting its LODs.

The results are a chip per prop. Clicking one shows its 3D view (the plain LOD0), a
table of its LODs with a download for each `.glb` and `.web.glb`, and **Turn 90°**. A
prop flagged as a tie gets a ⚠ on its chip and a note above the view. Turn 90° re-splits
the sheet with that turn and re-bakes only that prop, in the same run folder, so the
chest fix above is one click. Clicking twice turns it twice. A turn is built beside the
run and swapped in when it finishes, so one that fails or is cancelled leaves the run as
it was.

Above the chips, the run says which licence its props inherit, from the record kept
beside the generated model. With **Debug** off, Generate 3D deletes the run's manifest,
which is where Pixal3D keeps that record, and the tab says the record is missing rather
than guess.

Runs live in `output/props/<sheet>__props__<time>/` and stay listed under the form.
The tab says so when Blender or gltfpack is missing: without gltfpack the LODs are
written uncompressed. It looks for gltfpack on PATH, then at `vendor/gltfpack/gltfpack`.

## Limits

- **Some bend stays.** After standing up and squaring, what is left is 2 to 5 degrees,
  measured. It comes from how the sheet was drawn, and the field-of-view test above did
  not remove it.
- **Less detail per prop** than a run of its own, since the sheet shares one run's
  resolution between all of them.
- **The front of a round prop is wherever it was drawn.** Nothing turns it.
