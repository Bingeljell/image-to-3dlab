import io

import pytest
import trimesh

from image_to_3dlab.real_height import set_height


def test_finished_model_has_requested_height_and_bottom_centre():
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=(2, 4, 1))
    scene.add_geometry(box, transform=trimesh.transformations.translation_matrix((3, 7, -2)))
    result = set_height(scene.export(file_type="glb"), 1.8)
    loaded = trimesh.load(io.BytesIO(result), file_type="glb", force="scene")
    low, high = loaded.bounds
    assert high[1] - low[1] == pytest.approx(1.8)
    assert low[1] == pytest.approx(0)
    assert (low[[0, 2]] + high[[0, 2]]) / 2 == pytest.approx([0, 0])


@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf")])
def test_invalid_height_is_rejected(height):
    model = trimesh.Scene(trimesh.creation.box()).export(file_type="glb")
    with pytest.raises(ValueError):
        set_height(model, height)


def test_finish_accepts_an_optional_height():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
    import finish_api

    assert finish_api.normalise_settings({})["height_m"] is None
    assert finish_api.normalise_settings({"height_m": 1.8})["height_m"] == 1.8
    with pytest.raises(ValueError):
        finish_api.normalise_settings({"height_m": 0})


def test_height_is_baked_for_the_rig_importer():
    from image_to_3dlab.glb_turn import parse_glb

    model = trimesh.Scene(trimesh.creation.box(extents=(1, 4, 1))).export(file_type="glb")
    document, _ = parse_glb(set_height(model, 1.8))
    for node in document["nodes"]:
        assert not set(node) & {"matrix", "scale", "translation", "rotation"}
    for node in document["nodes"]:
        if "mesh" in node:
            primitive = document["meshes"][node["mesh"]]["primitives"][0]
            bounds = document["accessors"][primitive["attributes"]["POSITION"]]
            assert bounds["max"][1] - bounds["min"][1] == pytest.approx(1.8)
            assert bounds["min"][1] == pytest.approx(0)


def test_sized_file_keeps_licence_and_updates_its_hash(tmp_path):
    import hashlib
    import json

    from image_to_3dlab.real_height import size_file

    path = tmp_path / 'model.glb'
    path.write_bytes(trimesh.Scene(trimesh.creation.box()).export(file_type='glb'))
    sidecar = path.with_suffix('.provenance.json')
    sidecar.write_text(json.dumps({'license': {'name': 'test'}, 'output': {'sha256': 'old'}}))
    size_file(path, 1.8)
    record = json.loads(sidecar.read_text())
    assert 'height_m' not in record
    assert record['license'] == {'name': 'test'}
    assert record['output']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
