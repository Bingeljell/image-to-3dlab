---
workflow: product-launch-video
flow: automation
storyboard: no
message: "Turn a single image into a local, inspectable, reproducible 3D asset."
destination: website-embed
aspect: 1920x1080
language: en
audience: "technical artists, indie game developers, and local AI/3D builders"
length: 27.5s
angle: "Demo Loop"
narration: no
style_preset: code-editorial
---

## Intent

Create a high-impact product launch film for Image to 3D Lab. The piece should feel like
a research-grade tool crossing from experiment into something tangible: one image
ruptures into a real textured model, the full-screen result becomes the proof, and the
project opens into experimental multi-view, labelling, wind, rigging, and animation lanes.

## Assets

- ../../output/video/moss_fox_wind.mp4 — real wind-animation proof for the extension beat.
- ../../output/video/nikita_turntable.mp4 — real 3D turntable for the core hero reveal.
- ../../output/video/nikita_cheers.mp4 — real rigging/animation proof for the final capability beat.
- ../../assets_to_test/moss-fox-alpha_big.png — source image for the opening image-to-3D transformation.
- ../../assets_to_test/nikita_holding_beer.png — source image paired with the Nikita model proof.
- ../../output/conditional/nikita_holding_beer__trellis2__commercial-conditional__480361379a16.glb — hero GLB reference.
- ../../output/conditional/nikita_holding_beer__trellis2__commercial-conditional__480361379a16.provenance.json — provenance receipt for the hero run.

## Customizations

- Use terse kinetic on-screen copy instead of narration or sentence-length subtitles.
- Use the three approved videos as the visual spine: moss fox wind, Nikita turntable, Nikita cheers.
- Use local `audio-lab` only for music and SFX; do not use its procedural/synthesized route.
- Generate music with `stable-audio-3-medium` or `stable-audio-3-small-music`.
- Generate short SFX with `stable-audio-3-small-sfx`, each with duration exactly 1 second.
- Mark multi-view, labelled wind, and rigging/animation as experimental on screen.
- Keep the visual language warm-editorial and technical: paper/ink surfaces, dark code panels,
  amber/terracotta signal accents, forceful motion, hard editorial cuts, and real output evidence.

## Notes

- No website capture; the repository assets are the source of truth.
- The first pass is autonomous and should produce a reviewable draft plus contact sheet.
- Do not describe the output as a finished game-ready character; describe it as an inspectable,
  reproducible base asset that can be built on.
- V4 is intentionally unnarrated: a 155 BPM Stable Audio breakbeat drives a 27.5-second editorial
  cut. Energy comes from proof inserts, reframes, and sound hits while every primary claim remains
  readable for at least two seconds.
