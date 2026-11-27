"""Exercise the Studio preview launcher with a real background Blender child."""

import json
import sys
from pathlib import Path

import bpy


def check_preview_wrapper(workspace, source, assert_not_blank):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from viewer import preview_api

    output = workspace / "preview with spaces"
    output.mkdir()
    original_scene = bpy.context.scene
    original_objects = set(bpy.data.objects)
    original_size = preview_api.PANEL_W, preview_api.PANEL_H
    try:
        preview_api.PANEL_W = preview_api.PANEL_H = 64
        panels = preview_api.render_shots(
            Path(bpy.app.binary_path), source, output, 0, "clay",
            [("First", 0, 0), ("Last", 90, 1)],
        )
    finally:
        preview_api.PANEL_W, preview_api.PANEL_H = original_size
    assert [label for label, _ in panels] == ["0.0 s", "0.0 s"]
    times = json.loads((output / "times.json").read_text())
    assert times["First"] == 0
    assert times["Last"] > times["First"]
    pixels = {path.name: assert_not_blank(path) for _, path in panels}
    assert bpy.context.scene == original_scene
    assert set(bpy.data.objects) == original_objects
    print("PASS Studio preview wrapper rendered both animation endpoints in its own Blender process")
    return {"panels": len(panels), "times_seconds": times, "subject_pixels": pixels}
