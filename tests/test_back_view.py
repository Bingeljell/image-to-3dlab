"""Tests for image_to_3dlab.back_view: the prompt and reference for a back picture."""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from image_to_3dlab import back_view


def test_the_back_prompt_names_only_what_it_wants():
    # Naming a thing summons it: "no face" draws a face. The prompt says what to show.
    words = set(re.findall(r"[a-z']+", back_view.BACK_PROMPT.lower()))
    assert not words & {"no", "not", "never", "without", "don't", "avoid", "hidden"}


def test_the_back_prompt_keeps_pose_size_and_position():
    prompt = back_view.BACK_PROMPT.lower()
    for phrase in ("180 degrees", "same pose", "same size", "same position"):
        assert phrase in prompt


def test_the_back_prompt_names_no_body_parts():
    # Naming a part summons it, even on a subject that has no such part.
    words = set(re.findall(r"[a-z']+", back_view.BACK_PROMPT.lower()))
    assert not words & {"head", "body", "neck", "shoulders", "clothing", "arms", "legs", "ears"}


def test_the_back_prompt_holds_the_front_outline():
    # Pixal3D builds whatever the back adds, so the back keeps the front's outline.
    prompt = back_view.BACK_PROMPT.lower()
    for phrase in ("ends exactly where the reference ends", "same width at every height",
                   "same height and the same size"):
        assert phrase in prompt


def test_the_reference_is_the_cut_out_on_plain_grey():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:2, :, :3] = (200, 10, 10)
    rgba[:2, :, 3] = 255  # top half is the subject, bottom half see-through
    out = np.asarray(back_view.reference_on_grey(Image.fromarray(rgba, "RGBA")))
    assert out.shape == (4, 4, 3)
    assert tuple(out[0, 0]) == (200, 10, 10)
    assert tuple(out[3, 3]) == (back_view.GREY,) * 3
