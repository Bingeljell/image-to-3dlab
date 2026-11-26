# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The region split command accepts `--output-prefix` for the exported model and its materials.
- Animation commands accept rig names and bone maps for different characters.
- **Pixel Match: `--keep-top-paint`** leaves surfaces that face up and that no picture sees (a hat's brim, shoulders,
  the top of the hair) in the generator's own paint. The pictures' cameras stand level with the subject, so they never
  see those tops, and the local colour match shifted them like whatever the pictures see nearby: a wizard's dark violet
  brim beside a light hat band turned teal. Faces turned sideways still take the shift, so the head's sides under the
  brim keep their correction. Off by default.
- **Build from four pictures** (`scripts/pixal3d_generate.py --right --left`): front, right side, back and left side
  instead of front and back, every view at one shared scale. Front and back say nothing about how deep a subject is,
  so animals came out too short or with a second pair of ears or legs, and figures with boots got made-up shoe plates;
  from four views they come out right. **Pixel Match is off by default for a four-view build**: its paint can drift
  around a face. `--pixel-match` (on `pixal3d_generate.py`, or later on `retopo_repaint.py`) turns it on to try it,
  and so does the Finish setting `four_view_pixel_match`. Front-and-back builds keep Pixel Match as before.
  Skins currently support one- and two-view builds.
- **`scripts/make_side_views.py` draws those side pictures** with the studio's own picture engine: the right side from
  the front picture, the left one its mirror (or drawn too with `--draw-left`, for a tail curled to one side), so both
  agree on the length. `--four-legged` draws a side again until the legs stand apart (`image_to_3dlab/leg_check.py`):
  legs that touch in a picture build as one lump.
- **`pixal3d_generate.py --sides-paint safe`** for a figure's drawn sides: they paint only the ears, the side hair, the
  arms, the hands and the legs, never the face and never the body between the neck and a little below the hands, where
  a drawn hand or a drawn profile eye never sits exactly where the model has it (a drawn hand was painted onto a coat
  beside the holsters, a profile eye onto the brow as a second eye). The neck, the arms and the fingertips are read off
  the front picture's cut-out, so no rig is needed (`image_to_3dlab/side_paint.py`); `photo_paint.py --sides` sets it
  for a model painted again. `--sides-paint none` (or `--sides-shape-only`) lets the sides shape the model only;
  `all`, the default, is right for real side photos and a vehicle's doors and lettering.

### Fixed
- Model previews report a framing error for meshes without vertices.
- Live asset previews, turntables, and mesh baking restore the previous Blender scene after a failure.
