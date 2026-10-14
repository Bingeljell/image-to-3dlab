# Picking a picture: what to ask for

Most of a bad 3D result is decided before the 3D step starts. The 3D model can only build
what the picture shows it, and Pixel Match can only put back pixels that land on the right
surface. Fix the picture, and most problems never happen.

This page is for Pixal3D, the default route. Each point below was tested by changing one
thing about the same character and comparing the results.

![Top: the same character drawn flat anime-style, where the paint lands off the face, and as
a soft 3D-style render, where it lands. Bottom: a T-pose hand that smears into claws, and an
A-pose hand that comes out right](images/prompt-guide-style-and-pose.jpg)

## What works

**1. A soft 3D-render look.** Ask for a "stylised 3D game character render, soft even
lighting". Shading tells the 3D model which parts stick out: the brow, nose, cheeks, chin.
Flat anime or cel art has lines but almost no shading, so the model guesses the face, and
the paint then lands slightly off it. Face size is not the cause; a small face in a shaded
picture comes out fine.

**2. A-pose, not T-pose.** Arms angled down about 45°. In a T-pose the hands sit near the edge
of the space the model builds in, and fingers smear outwards into claws or planks. In an
A-pose the same hands come out with the right length and shape.

**3. Plain, relaxed hands.** Open hands, palms facing down. Say as little about fingers as you
can: asking for them tends to make them curl, and curled fingers come out merged or doubled.

**4. One subject, plain background, no props.** The background is cut out first; anything
held or nearby gets fused into the model.

**5. Eyes open, neutral face.** Closed or squinting eyes get carved into the shape as slits.

**6. Realistic characters: ask for slightly larger, clearly defined eyes.** In a full-body
picture a realistic face is small, and its eyes come out as a shelf with a dark rim around
them. "Slightly larger, clearly defined eyes with visible eyelids" fixes most of that while the
character still looks realistic. Stylised characters with big eyes already come out well.

**7. Name what you want, not what you don't.** "No wrinkles" tends to add wrinkles; "smooth
fabric" doesn't. The same goes for props: describe empty hands rather than ruling a weapon out.

## Things that tend to go wrong

These have gone wrong more than once, but are less proven than the list above:

- **Hoods or hats deep over the face.** The model has to guess how deep the opening is, and
  the face inside it comes out off-model. Push the hood back.
- **Heavy ink outlines.** Strong lines can get carved into the surface as grooves.
- **Strands of hair across the face.** Thin fringe strands come out as stiff spikes.

## A prompt to start from

```
A [character] in A-pose facing the camera, full body, arms angled down about 45 degrees
with a clear gap from the body, hands open, palms facing down, eyes open, calm expression,
slightly larger, clearly defined eyes with visible eyelids, [outfit], stylised 3D game character render, soft even lighting, plain light grey background
```
