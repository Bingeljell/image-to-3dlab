# How image-to-3D actually works, and the two places we broke it

A plain-language walkthrough. No prior 3D knowledge assumed; every technical word is
defined the first time it appears. Written 2026-08-12, the day we discovered that most of
what we had blamed on TRELLIS was our own doing.

---

## The vocabulary, once

- **Mesh** — a 3D model's skin. A shell of flat triangles, like a papier-mâché figure. It
  is hollow; there is nothing inside.
- **Vertex** (plural vertices) — a corner point. Three vertices make one triangle.
- **Face / triangle / polygon** — one flat patch of that shell. "300,000 faces" means the
  shell is built from 300,000 little triangles. More triangles = finer detail = bigger file.
- **Decimation / simplify** — deliberately deleting triangles to make the model lighter,
  trying to keep the shape. Like redrawing a picture with fewer brush strokes.
- **Texture** — the image painted onto the mesh, giving it colour. A flat picture wrapped
  around a 3D shape.
- **UV unwrapping** — flattening the 3D shell into a 2D pattern so a flat image can be
  painted onto it. Exactly like cutting a cardboard box along its edges and pressing it
  flat, or the flat panel layout on a sewing pattern.
- **UV atlas** — that flattened pattern, packed into one square image.
- **Island** — one connected piece of the flattened pattern. A good unwrap gives a few big
  islands (head, body, each leg). A bad one gives thousands of confetti scraps.
- **Baking** — computing the colours once and saving them into the texture image, so they
  don't have to be worked out every time the model is drawn.
- **Winding** — the order a triangle's three corners are listed in, which is how software
  decides which side of it is the *outside*. Get it backwards and the triangle faces
  inward.
- **Backface culling** — an optimisation where the renderer skips triangles facing away
  from the camera, because you shouldn't be able to see the inside of a solid object.
  Games do this always.

---

## What the demo does, stage by stage

The HuggingFace demo shows three stages. Here is what each one means.

### Stage 1 — Sparse Structure Generation

The model decides **where the object exists in space**, very coarsely. Imagine a 3D grid
of cubes filling the space around your creature, and the model ticks which cubes contain
any part of it. No detail, no surface, no colour — just a blocky occupancy map, like
Minecraft.

"Sparse" means it only stores the cubes that are occupied, not the empty ones, because
most of the space is empty.

### Stage 2 — Shape Generation

Now it refines that blocky map into an actual **surface**. This is where the creature's
form appears: the curve of an ear, the taper of a leg. The output is a very dense mesh —
on Flicker, **3.2 million triangles**.

That number matters for everything that follows.

### Stage 3 — Material Generation

Finally it works out what the surface **looks like** — colour, and how shiny or rough it
is. This gets baked into texture images.

### The fourth stage, which the demo page doesn't show

After those three, there is a **post-processing** step, and this is where all of our
problems lived. In order:

1. Clean up the raw mesh — fill small holes, fix bad geometry
2. **Decimate** it from millions of triangles to something usable
3. **UV unwrap** the result — flatten it into a 2D pattern
4. **Bake** the colours into a texture image using that pattern
5. Export as a `.glb` file

Every one of these steps operates on the *output of the previous one*. Damage introduced
early gets baked into everything after it. That's the whole story of what went wrong.

---

## Where we broke it

### Break 1 — we deleted 94% of the model before anyone could use it

Our Apple Silicon version had this line, sitting *before* step 1 above:

```python
target_faces = min(args.bake_target_faces, 200000, len(faces_np))
```

In plain terms: **"never let the mesh be bigger than 200,000 triangles."**

Stage 2 had just produced 3.2 million triangles. This line threw away 94% of them — and
did it with a crude, fast tool rather than the careful one that runs later.

An analogy: you commission a detailed pencil drawing, then photocopy it at 6% size on a
bad photocopier, then hand the photocopy to a restorer and ask them to clean it up. Every
later step — the cleanup, the unwrapping, the painting — was working from the photocopy.

**What it looked like:** the surface came out covered in fine cracks, like crazed pottery.
Markings on the creature turned into torn trenches in the geometry.

**Why it was there:** to avoid a crash in Apple's graphics library on very large meshes.
A real concern — but nobody ever measured what it cost, and it cost nearly everything.

**A second, sneakier consequence:** the setting we thought controlled mesh detail,
`bake_target_faces`, was being silently overruled. Ask for 300,000 or 3,000,000 and you
got the same ~197,000 either way. Every experiment tuning that number was measuring
nothing at all.

### Break 2 — our models were inside-out

Every mesh we produced had its triangles facing the wrong way — not all of them, but
enough that the model's measured "volume" came out **negative**, which is the mathematical
way of saying the shell is inverted.

Here is why this hid for so long. The `.glb` file format marks materials as
**double-sided** by default, meaning "draw this triangle from both sides." So in a normal
preview, everything looked fine.

But games don't do that. They use backface culling — skip anything facing away — because
it's faster and you shouldn't see the inside of a solid object anyway. Turn that on, and
our models were **hollow**: you looked straight through the chest and saw the inside of the
creature's back.

You spotted this immediately in Blender: *"the front is not there at all — we can see the
inside."*

**The damage beyond appearance:** this corrupted our own measuring tools. We had a "tear
metric" counting see-through gaps, and we steered the whole project by it for weeks. But a
backwards-facing triangle looks *exactly* like a missing one to that measurement. So we
were largely counting flipped triangles and calling them holes — which is why every repair
we tried failed to move the number, and why "recalculate the directions" kept seeming to
fix things at random.

---

## How we found out, and why it took so long

For over a week, every experiment compared **our output against our other output**. Higher
face count versus lower. This texture setting versus that one. With the fix versus without.

The flaw: *both sides of every comparison had the same damage*. A defect present in
everything is invisible — it stops looking like a defect and starts looking like "what this
tool does." That's precisely how we concluded TRELLIS had weak texturing and unusable UV
layouts.

Then you asked *"how come others are getting much better results?"*, pasted your artwork
into the official demo, and sent back the result. Ten seconds of looking at it:

| | Ours | Official demo |
|---|------|---------------|
| Triangles | 100,291 | **281,889** |
| Facing the right way | **no** | yes |
| See-through when culled | badly | **no** |
| Markings carved into the surface | yes, torn open | **none — smooth** |

Same input image. That is called a **control group**: a known-good reference from outside
your own system. We had never had one, and it invalidated more conclusions in ten minutes
than a week of internal work had produced.

---

## What to do differently

1. **Get a control group first.** If a tool has a public demo, run your real input through
   it before theorising about the tool's limits. Minutes, not days.
2. **Judge with backface culling on.** A double-sided preview will cheerfully hide a
   hollow model. It is not a real view of your asset.
3. **Check that a setting does what it says.** `bake_target_faces` was clamped for weeks
   and every experiment on it measured noise. Print what actually happened, not what you
   requested.
4. **Look before measuring.** A number can only see what it was built to see. Our hole
   metric could not distinguish "missing" from "backwards", so it confidently reported
   nonsense. The render showed it instantly.

Both breaks and the code that fixes them are in
[self-inflicted-damage.md](self-inflicted-damage.md).
