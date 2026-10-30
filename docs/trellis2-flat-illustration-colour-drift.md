# TRELLIS.2: picking a picture that keeps its colours

TRELLIS.2 does not paste your picture onto the model. It looks at the picture, then
decides for itself what the surface is made of. On some pictures it decides wrong, and
the colours come out dark, nearly black, or washed out.

## Pictures that work

Photographs, and pictures that look like a softly lit 3D render:

- light and shadow visible across the shape;
- enough shading that you can tell it is round, not flat;
- the colours you want visible over large areas;
- a transparent background.

## Pictures that go wrong

Flat or vector-style art: solid fills, no shading, hard outlines. A flat blue cartoon dog
can come back nearly black, while a shaded 3D-style render of the same dog keeps its blue.
Changing the seed or the texture size will not fix it. Try a shaded version of the
picture, or use **Pixal3D**, which keeps flat colours much better.

## Resolution changes the colours too

At `1024_cascade`, colours can come out more washed out than at `512`, even on a good
picture. The two settings use different texture models, so this is a real trade:

- `1024_cascade`: sharper shapes (separate leaves, fur strands), weaker colour;
- `512`: softer shapes, truer colour, and about five times faster.

![The same asset at 512 (green kept) and 1024 cascade (bleached), two seeds
each](images/pipeline-type-colour-shift.jpg)

If you need both, generate at 1024 and repaint it on the **Finish** page.

## The warning in Generate 3D

When you pick a picture for TRELLIS.2, a small local check (TinyCLIP) says whether it
looks flat. It is a hint, not a verdict: it never blocks a run, and your own eye is the
better judge. The checkpoint is already on disk; the check does not fetch one. The
picture never leaves your machine.
