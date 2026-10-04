import numpy as np
import pytest

from image_to_3dlab import glb_turn
from image_to_3dlab import photo_paint as pp


def make_glb(positions, normals=None, uvs=None, node=None, extras=None) -> bytes:
    positions = np.asarray(positions, dtype="<f4")
    parts = [positions]
    attributes = {"POSITION": 0}
    accessors = [{"bufferView": 0, "componentType": 5126, "count": len(positions),
                  "type": "VEC3", "min": positions.min(0).tolist(),
                  "max": positions.max(0).tolist()}]
    for name, data, kind in (("NORMAL", normals, "VEC3"), ("TEXCOORD_0", uvs, "VEC2")):
        if data is not None:
            attributes[name] = len(accessors)
            accessors.append({"bufferView": len(accessors), "componentType": 5126,
                              "count": len(data), "type": kind})
            parts.append(np.asarray(data, dtype="<f4"))
    views, offset = [], 0
    for part in parts:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": part.nbytes})
        offset += part.nbytes
    document = {
        "asset": {"version": "2.0", **({"extras": extras} if extras else {})},
        "nodes": [{"mesh": 0, **(node or {})}],
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    return glb_turn.build_glb(document, b"".join(p.tobytes() for p in parts))


def read(data: bytes, accessor: int) -> np.ndarray:
    document, binary = glb_turn.parse_glb(data)
    a = document["accessors"][accessor]
    view = document["bufferViews"][a["bufferView"]]
    width = {"VEC2": 2, "VEC3": 3}[a["type"]]
    return np.frombuffer(binary, "<f4", a["count"] * width, view["byteOffset"]).reshape(-1, width)


POSITIONS = [[1.0, 2.0, 3.0], [-0.5, 0.0, -4.0]]


def test_half_a_turn_negates_x_and_z_of_positions_and_normals_only():
    glb = make_glb(POSITIONS, normals=[[0, 0, 1], [1, 0, 0]], uvs=[[0.1, 0.2], [0.3, 0.4]])
    turned = glb_turn.turn_glb(glb)
    assert read(turned, 0).tolist() == [[-1.0, 2.0, -3.0], [0.5, 0.0, 4.0]]
    assert read(turned, 1).tolist() == [[0, 0, -1], [-1, 0, 0]]
    assert np.allclose(read(turned, 2), [[0.1, 0.2], [0.3, 0.4]])


def test_bounds_follow_the_turn():
    document, _ = glb_turn.parse_glb(glb_turn.turn_glb(make_glb(POSITIONS)))
    assert document["accessors"][0]["min"] == [-1.0, 0.0, -3.0]
    assert document["accessors"][0]["max"] == [0.5, 2.0, 4.0]


def test_a_turned_glb_is_stamped_and_not_turned_twice():
    once = glb_turn.turn_glb(make_glb(POSITIONS))
    assert glb_turn.glb_faces_front(once)
    assert not glb_turn.glb_faces_front(make_glb(POSITIONS))
    assert glb_turn.turn_glb(once) == once


def test_other_asset_extras_survive():
    turned = glb_turn.turn_glb(make_glb(POSITIONS, extras={"seed": 42}))
    extras = glb_turn.parse_glb(turned)[0]["asset"]["extras"]
    assert extras == {"seed": 42, glb_turn.FACING_KEY: glb_turn.FRONT}


def test_node_transforms_are_refused_not_half_turned():
    with pytest.raises(ValueError, match="node transforms"):
        glb_turn.turn_glb(make_glb(POSITIONS, node={"rotation": [0, 1, 0, 0]}))


def test_turned_model_and_turned_camera_put_every_point_on_the_same_pixel():
    """The point of turning both: Pixel Match must still land the photo where it did."""
    camera = np.array([[1, 0, 0, 0], [0, 0, -1, -2.8], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    image = np.zeros((64, 64, 4), np.uint8)
    points = np.array([[0.1, 0.2, 0.05], [-0.3, -0.1, 0.2], [0.0, 0.4, -0.1]])

    def pixels(glb_points, matrix):
        view = pp.View(image, np.asarray(matrix, float), 0.35)
        x, y, depth = pp.project(pp.to_view_space(glb_points), view)
        return np.stack([x, y, depth], axis=1)

    turned_points = points * [-1, 1, -1]
    turned_camera = glb_turn.turn_cameras({"frames": [{"transform_matrix": camera.tolist()}]})
    assert np.allclose(pixels(turned_points, turned_camera["frames"][0]["transform_matrix"]),
                       pixels(points, camera))


def test_cameras_turn_once():
    meta = {"camera_angle_x": 0.35, "frames": [{"file_path": "input.png",
                                                "transform_matrix": np.eye(4).tolist()}]}
    once = glb_turn.turn_cameras(meta)
    assert once["frames"][0]["transform_matrix"][0][0] == -1.0
    assert once["camera_angle_x"] == 0.35 and once["frames"][0]["file_path"] == "input.png"
    assert glb_turn.turn_cameras(once) == once
