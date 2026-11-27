"""MeshRender.inpaint falls back only when OpenCV is missing, never when OpenCV fails."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PAINT = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "paint"


@pytest.fixture
def mesh_render(monkeypatch):
    monkeypatch.syspath_prepend(str(PAINT))
    pytest.importorskip("scipy")
    from hy3dpaint_mlx import mesh_render as module

    return module


def _texture_with_hole():
    texture = np.zeros((4, 4, 3), np.float32)
    texture[:, :2] = 0.8
    mask = np.zeros((4, 4), bool)
    mask[:, :2] = True
    return texture, mask


def test_an_opencv_error_is_not_hidden_as_a_fallback_fill(monkeypatch, mesh_render):
    stub = types.ModuleType("cv2")
    stub.INPAINT_NS = 0

    def broken(*_args, **_kwargs):
        raise ValueError("opencv broke")

    stub.inpaint = broken
    monkeypatch.setitem(sys.modules, "cv2", stub)
    texture, mask = _texture_with_hole()
    with pytest.raises(ValueError, match="opencv broke"):
        mesh_render.MeshRender.inpaint(texture, mask)


def test_a_missing_opencv_keeps_the_nearest_fill(monkeypatch, mesh_render):
    monkeypatch.setitem(
        sys.modules, "cv2", None
    )  # makes `import cv2` raise ImportError
    texture, mask = _texture_with_hole()
    out = mesh_render.MeshRender.inpaint(texture, mask)
    assert np.allclose(out, 0.8)
