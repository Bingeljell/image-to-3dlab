"""Photo paint CLI reports when alignment and safe side views are active."""

import json

import numpy as np
import test_coverage_geometry_finish
from test_coverage_utilities_mesh_edges import invoke

from image_to_3dlab.photo_paint import read_glb
from scripts import photo_paint

photo_inputs = test_coverage_geometry_finish.photo_inputs


def test_photo_paint_cli_reports_align_and_safe_side_without_weights_file(photo_inputs, tmp_path, monkeypatch, capsys):
    source, image, views = photo_inputs
    metadata_path = views / "transforms.json"
    metadata = json.loads(metadata_path.read_text())
    side_camera = [[0, 0, 1, 4], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    side = views / "view_002.png"
    side.write_bytes(image.read_bytes())
    metadata["frames"].append({"file_path": side.name, "transform_matrix": side_camera})
    metadata_path.write_text(json.dumps(metadata))
    target = tmp_path / "painted.glb"
    invoke(photo_paint, monkeypatch, source, views, target, "--sides", "safe")
    report = capsys.readouterr().out
    assert "align view_" in report
    assert "safe side view_002.png" in report
    positions, uv, faces, atlas = read_glb(target)
    before = read_glb(source)
    for actual, expected in zip([positions, uv, faces], before[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert atlas.shape == before[3].shape
