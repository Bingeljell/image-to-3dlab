"""Check export paths for incomplete metadata and rigs that cannot be reoriented."""

import numpy as np
import pytest
from test_roblox_export import _model, _picture, _skinned

from image_to_3dlab import roblox_export as rx
from image_to_3dlab.glb_turn import build_glb, parse_glb


@pytest.mark.parametrize(("mode", "count"), [(4, 2), (5, 4), (6, 4), (1, 0)])
def test_export_counts_triangles_for_each_primitive_mode(mode, count):
    doc, data = parse_glb(_model())
    doc["meshes"][0]["primitives"][0]["mode"] = mode
    assert rx.export(build_glb(doc, data)).triangles == count


@pytest.mark.parametrize("dimension", ["height_studs", "longest_studs"])
def test_export_cannot_size_model_without_recorded_bounds(dimension):
    doc, data = parse_glb(_model())
    doc["accessors"][0].pop("max")
    with pytest.raises(rx.RobloxExportError, match="no (height|size) to measure"):
        rx.export(build_glb(doc, data), **{dimension: 5})


def test_export_external_picture_is_warned_about_and_left_linked():
    doc, data = parse_glb(_model(pictures=[("image/png", _picture("PNG"))]))
    doc["images"][0] = {"uri": "external.png"}
    report = rx.export(build_glb(doc, data))
    assert any(check["level"] == "warn" and "linked from outside" in check["text"] for check in report.checks)
    new, _ = parse_glb(report.glb)
    assert new["images"][0] == {"uri": "external.png"}


def test_export_dimension_limit_checks_the_width_after_sizing_height():
    report = rx.export(_model(height=.001), height_studs=5)
    assert report.size_studs[0] == 5000
    assert any("Roblox parts stop at 2,048" in c["text"] for c in report.checks)


@pytest.mark.parametrize(("change", "note"), [
    ("matrix", "scaled or matrix bones"),
    ("curve", "curved keys"),
    ("scale-track", "changes a bone's size"),
])
def test_rig_edit_skips_unsupported_transforms_without_mutation(change, note):
    doc, data = parse_glb(_skinned())
    bone = doc["skins"][0]["joints"][0]
    if change == "matrix":
        doc["nodes"][bone]["matrix"] = np.eye(4).ravel().tolist()
    else:
        doc["animations"] = [{
            "samplers": [{"interpolation": "CUBICSPLINE" if change == "curve" else "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": bone, "path": "scale"}}],
        }]
    before = build_glb(doc, data)
    out, explanation = rx.straighten_bones(doc, data)
    assert out == data and build_glb(doc, out) == before
    assert note in explanation


def test_export_records_why_rig_was_not_reoriented():
    doc, data = parse_glb(_skinned())
    doc["nodes"][doc["skins"][0]["joints"][0]]["scale"] = [2, 1, 1]
    report = rx.export(build_glb(doc, data))
    assert any("scaled or matrix bones" in note for note in report.notes)


def test_export_reports_invalid_size_and_texture_limit():
    with pytest.raises(ValueError, match="height_studs must"):
        rx.export(b"", height_studs=0)
    with pytest.raises(ValueError, match="max_texture must"):
        rx.export(b"", max_texture=0)
