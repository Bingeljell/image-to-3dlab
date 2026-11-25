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

# The side for a four-view build (`pixal3d_generate.py --right --left`), by the same rules: no part
# is named, only the turn, the way the subject faces, and that its whole depth shows. A front and a
# back picture say nothing about how long a subject is from front to back (an animal, a vehicle),
# so a build from them alone comes out short or doubles what it cannot place (a second pair of
# ears, a third pair of legs); a side picture says it. Tested 2026-11-15 on two pets, a person and a
# van against two other wordings: "seen from its right side, the camera moved 90 degrees" drew a fox
# almost from the front and the van at three quarters; asking for the side view of a turnaround
# sheet gave a true profile for all four, and kept a round creature round.
SIDE_PROMPT = (
    "The side view from a turnaround sheet of the same subject as in the reference image: seen "
    "exactly from its right side, in strict profile, its front pointing to the right edge of the "
    "picture and its rear to the left edge, its whole length from front to rear visible, nothing of "
    "its front turned to the camera. Same pose, same proportions, same size and same height in the "
    "frame. Keep the colours, materials and design consistent with the reference. Flat even "
    "lighting, plain light grey background."
)

# The backdrop the reference is laid on: light grey, like the backgrounds our prompts ask for.
GREY = 236


def mirror_side(right: Image.Image) -> Image.Image:
    """The left side of a subject that is the same on both sides: its right side, mirrored.

    Drawing both sides lets them disagree on the subject's length (two sides of one van came out
    508 and 691 px long, 2026-11-13); a mirror cannot.
    """
    return right.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def reference_on_grey(cut_out: Image.Image) -> Image.Image:
    """The cut-out front composited on plain grey, as an RGB picture for Qwen to read."""
    rgba = cut_out.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (GREY, GREY, GREY, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")
