# Pixel Match: your picture's real pixels, on the model

![Close-up of a robot's chest: the generated model's lettering is garbled, the Pixel
Match model reads VANGUARD 07 exactly like the source picture](images/pixel-match-lettering-before-after.jpg)

Every image-to-3D model **redraws** your picture. The generator looks at the picture and
paints the model from what it understood, so anything small and exact — text, a logo, a
number, a face — comes back as a lookalike that is nearly, but not quite, right.

**Pixel Match** is the Finish step putting the picture's real pixels back. The photo
already holds the right answer for every surface it can see, so those surfaces take the
photo's pixel, and the model's own paint stays only where no photo looks.

## When it runs

It is part of **Finish** in the studio, on by default:

- **Pixal3D models made in this lab**: automatic. Every Pixal3D run saves the camera it
  used beside the model (`<run>.svviews/`), and Finish finds it by matching the model's
  content, so there is nothing to point at.
- **Anything else** (another generator, a model from elsewhere): skipped, with a note
  saying so. There is no saved camera to project through.
- **Turn it off** with the unticked **Pixel Match** box on the Finish step. Each step in
  the Steps panel has its own Download GLB, so the model from before Pixel Match is
  always one click away too.

## How it works

One picture, one camera. The Pixal3D run's camera is aimed at the model, and every point
on the surface is projected into the picture — like a slide beamed back onto the screen
it was photographed from. Then each texel (a pixel of the model's texture) is judged on
four things before the photo's pixel is trusted:

- **Can the photo see it?** A texel hidden behind another part gets nothing.
- **Does it face the camera?** A surface seen edge-on smears a few photo pixels over a
  large area, so grazing angles are faded out.
- **Is it away from the silhouette edge?** The outline is where a cut-out bleeds
  backdrop colour and a slightly-off camera misses, so a few pixels inward are faded.
- **Is it inside the photo's cut-out?** Pixels outside the cut-out's matte are ignored.

Where the photo is trusted it wins; where it is not, the model keeps its paint. The
model's own colours are also shifted toward the photo's palette, so the sides and back
the photo never saw match the front instead of the generator's guess.

Only the base colour texture changes. The mesh, the UVs and every other byte of the GLB
are left alone, and the run takes seconds.

## When it can look worse

- **Some assets come out worse than the raw model.** When that happens, untick Pixel
  Match or download the pre-Finish model (see [known issues](info_and_credits.md)).
- **The model has to still line up with its camera.** Finish's retopology keeps
  coordinates, so a normal Finish run lines up; a model you moved, edited or re-exported
  elsewhere may not, and the photo lands slightly off.
- **Only the front is real.** Sides and back keep the generated paint, colour-matched.
  A second photo (a turnaround sheet, a generated side view) can be added as another
  view — the code takes a list of views, not one photo.

## The trust map

Every run computes how much it trusted the photo at each texel. From the command line,
`--weights` writes that map as a PNG — white where the photo's pixels were used, black
where the model kept its own paint. When a result looks wrong, look at this first: it
tells you whether the camera missed the part you care about or the trust tests turned
it away.

## From the command line

The same step without the viewer:

```bash
# As part of a Finish run (a Pixal3D run's saved views live beside its GLB):
python scripts/retopo_repaint.py generated.glb source.png finished.glb \
    --faces 40000 --skip-paint --views output/run.svviews

# Or on its own, on a model Finish already produced:
python scripts/photo_paint.py finished.glb output/run.svviews out.glb \
    --weights weights.png
```

The model must be one mesh with one base-colour texture and no node transforms — what
Finish writes. The trust thresholds can be tuned on `photo_paint.py`:

| Flag | Meaning / first adjustment |
|---|---|
| `--facing-low` / `--facing-high` | How square to the camera a surface must be before the photo is trusted (0–1; full trust at the high value). Raise both if slightly angled surfaces smear. |
| `--edge-px` | How many pixels inward from the silhouette the photo fades in. Raise if a rim of backdrop colour shows. |

Pure numpy and Pillow: no Blender, no GPU, no model download. The projection uses the
same formula as pixal3d.cpp itself, and the coordinate frame was found by search and
checked on every asset tried.
