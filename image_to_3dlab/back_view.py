"""The back picture for a front + back build: what Qwen is asked, and what it is shown.

Qwen-Image draws the back in edit mode from the front picture. Two rules found the hard
way (2026-10-10 to 10-12, eight characters):

- **Name only what you want.** Naming a thing summons it, so the prompt never says what to
  leave out. That cuts both ways: naming parts summons them even when the subject has none
  (2026-10-29), so the prompt names no parts at all, only that the back holds what the
  reference holds, with the same outline. It works for any subject that way.
- **Show the subject on plain grey.** The reference is the front's cut-out laid on flat
  light grey, so Qwen redraws the subject and not a backdrop, and the result cuts out
  cleanly.

Same pose, size and place in the frame matter as much as the drawing: Pixal3D reads the
two pictures through cameras exactly half a turn apart.
"""

from __future__ import annotations

from PIL import Image

BACK_PROMPT = (
    "The same subject as in the reference image, seen from directly behind: the camera has "
    "moved 180 degrees around it. Same pose, same proportions, same size and same position "
    "in the frame. The view shows the back of exactly the parts that are in the reference "
    "and ends exactly where the reference ends. The outline matches the reference: the same "
    "width at every height, and every part at the same height and the same size as in the "
    "reference. Keep the colours, materials and design consistent with the reference. Flat "
    "even lighting, plain light grey background."
)

# The backdrop the reference is laid on: light grey, like the backgrounds our prompts ask for.
GREY = 236


def reference_on_grey(cut_out: Image.Image) -> Image.Image:
    """The cut-out front composited on plain grey, as an RGB picture for Qwen to read."""
    rgba = cut_out.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (GREY, GREY, GREY, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")
