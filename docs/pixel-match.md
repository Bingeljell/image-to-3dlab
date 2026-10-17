# Pixel Match: your picture's real pixels, on the model

![Close-up of a robot's chest: the generated model's lettering is garbled, the Pixel
Match model reads VANGUARD 07 exactly like the source picture](images/pixel-match-lettering-before-after.jpg)

Every image-to-3D model **redraws** your picture. The generator looks at the picture and
paints the model from what it understood, so anything small and exact — text, a logo, a
number, a face — comes back as a lookalike that is nearly, but not quite, right.

**Pixel Match** is the Finish step putting the picture's real pixels back. The photo
already holds the right answer for every surface it can see, so those surfaces take the
photo's pixel, and the model's own paint stays only where no photo looks.

**Front + back.** With Front + back on (Create's default once its downloads are present),
Qwen draws the back of your picture, Pixal3D builds the shape from both, and Pixel Match
paints from both. The back stops being a guess: it is a picture too.

![A friar from the front, side and back: top, the generator's own paint, flat and darker,
with no hood on his back; bottom, front + back Pixel Match, the habit's real colour and
folds, hood included](images/front-back-pixel-match.jpg)

## When it runs

It is part of **Finish** in the studio, on by default:

- **Pixal3D models made in AssetFurnace**: automatic. Every Pixal3D run saves the cameras it
  used beside the model (`<run>.svviews/`: the front, plus the back for a Front + back
  run), and Finish finds them by matching the model's content, so there is nothing to
  point at.
- **Anything else** (another generator, a model from elsewhere): skipped, with a note
  saying so. There is no saved camera to project through.
- **Turn it off** by unticking the **Pixel Match** box on the Finish step. Each step in
  the Steps panel has its own Download GLB, so the model from before Pixel Match is
  always one click away too.

## How it works

Each picture has a camera. The Pixal3D run's cameras are aimed at the model, and every
point on the surface is projected into each picture, like a slide beamed back onto the
screen it was photographed from.

**First, each picture is nudged onto the model.** The model never matches its picture
exactly: the camera is a pixel or two off, and the model builds some features a little
away from where they are drawn. So before painting, Pixel Match fits the picture's
cut-out onto the model's outline (a pixel or two, about 1% in size), then moves detailed
spots, such as brows, eyes and ink lines, onto where the model built them, by comparing
the picture with the model's own paint from the same camera. It only moves what it is
sure of, a few pixels at most; plain cloth and skin stay where they are.

**Then each texel** (a pixel of the model's texture) is judged before a picture's pixel is
trusted:

- **Can the picture see it?** A texel hidden behind another part gets nothing.
- **Does it face the picture?** Only surfaces turned towards a picture take its paint, and
  grazing angles are faded out, since they smear a few pixels over a large area. (The far
  side of a thin cloak or ponytail faces away, so it keeps its own paint.)
- **Is it away from an edge?** Three kinds: the model's outline, the edge of the
  picture's cut-out (where pixels are half background), and inner edges, where one part
  sits in front of another (an ear on a head, a collar on a neck). The drawn edge and the
  model's edge are a few pixels apart, so the picture fades out near all three rather
  than drawing lines across them.
- **Is it inside the picture's cut-out?** Pixels outside it are ignored.

Where a picture is trusted it wins; where it is not, the model keeps its paint. **That
leftover paint is colour-matched spot by spot:** the gap between the model's paint and
the picture is measured wherever the picture is trusted and spread smoothly over the
surface, so the sides no picture faces meet the front and the back in colour instead of
showing darker bands.

Only the base colour texture changes. The mesh, the UVs and every other byte of the GLB
are left alone. A Finish mesh takes seconds; a million-triangle raw model under a minute.

## When it can look worse

- **Some assets come out worse than the raw model.** When that happens, untick Pixel
  Match or download the pre-Finish model (see [known issues](info_and_credits.md)).
- **Coloured specks in hair, or small eyes gone dark.** The inner-edge fade mistakes deep
  eye sockets and layered hair for "one part in front of another" and fades the picture out
  there, most often on Full detail. Turn off **Edge fade** (Finish > Advanced; agents pass
  `advanced={"edge_fade": false}`; the command line takes `--no-edge-fade`) and Finish again.
- **A feature lands twice (a ghost eye) or slides off its place.** Before painting, Pixel
  Match fits each picture's outline onto the model's. A better fitting outline does not
  always mean a better fitting face, most often on a turned head. Turn off **Nudge**
  (Finish > Advanced; agents pass `advanced={"nudge": false}`; the command line takes `--no-nudge`) and
  Finish again.
- **The model has to still line up with its camera.** Finish's retopology keeps
  coordinates, so a normal Finish run lines up; a model you moved, edited or re-exported
  elsewhere may not, and the photo lands slightly off.
- **The sides are still the generator's.** Neither the front nor the back picture looks at
  them squarely, so they keep the generated paint, colour-matched to both pictures.
  Painting from pictures the model was not built from (a turnaround sheet you drew, a
  side view) is not supported: they do not line up and paint features twice.
- **Shape mistakes are not paint mistakes.** When the generator builds a feature wrong
  (an eye carved as a slot, a lump on a hood rim, a floor under the feet), Pixel Match
  paints it faithfully. Fix the shape, or try another picture or seed.

## The trust map

Every run computes how much it trusted the photo at each texel. From the command line,
`--weights` writes that map as a PNG — white where the photo's pixels were used, black
where the model kept its own paint. When a result looks wrong, look at this first: it
tells you whether the camera missed the part you care about or the trust tests turned
it away.

## From the command line

The same step without the viewer:

```bash
# Front + back: draw the back, then build from both (the views land in output/run.svviews)
python scripts/make_back_view.py front.png back.png
python scripts/pixal3d_generate.py front.png output/run.glb --back back.png

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
| `--cutout-edge` | Fade from the picture's own cut-out edge, as a share of its width (8 px at 768). `0` turns it off. |
| `--inner-edge` | Fade beside inner edges (an ear on a head), as a share of the width. `0` turns it off. |
| `--no-local-colour` | Colour-match with one shift for the whole model, as before. |
| `--no-align` | Paint from the pictures as they are, without nudging them onto the model. |

Each run prints, per picture, what alignment did: how well the outlines matched before and
after, the scale and shift, and how many confident spots it matched.

Pure numpy and Pillow: no Blender, no GPU, no model download. The projection uses the
same formula as pixal3d.cpp itself.
