"""The weight-bleed check must read world height, not the mesh's local Z.

The Blender program is a string, so these tests run its verification section
(step 5) against small stand-ins for the mesh and the scene. No Blender needed.
"""

from __future__ import annotations

import json
import types

import blender_build_rig as rig
import blender_voxel_weights_code as weights


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def copy(self):
        return Vec(self.x, self.y, self.z)


class Matrix:
    """A 4x4 transform given as rows, applied with `@` like Blender's."""

    def __init__(self, rows):
        self.rows = rows

    def __matmul__(self, v):
        x, y, z = v.x, v.y, v.z
        r = self.rows
        return Vec(
            *(r[i][0] * x + r[i][1] * y + r[i][2] * z + r[i][3] for i in range(3))
        )


def _lift(dz):
    return Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, dz], [0, 0, 0, 1]])


def _tilt_up():
    """Local Y becomes world Z (a mesh turned on its back, as imports often are)."""
    return Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def _verdict(matrix, vertices, head_marker_world_z):
    """Run step 5 on a mesh with `vertices` as (local xyz, group name) pairs."""
    code = weights.build_code(0.012, 0.15, False)
    section = code[code.index("# --- 5. verification") :]
    names = sorted({name for _, name in vertices})
    mesh = types.SimpleNamespace(
        matrix_world=matrix,
        vertex_groups=[
            types.SimpleNamespace(index=i, name=n) for i, n in enumerate(names)
        ],
        data=types.SimpleNamespace(
            vertices=[
                types.SimpleNamespace(
                    co=Vec(*co),
                    groups=[types.SimpleNamespace(group=names.index(name), weight=1.0)],
                )
                for co, name in vertices
            ]
        ),
    )
    head = types.SimpleNamespace(
        name="JOINT_head",
        matrix_world=types.SimpleNamespace(translation=Vec(0, 0, head_marker_world_z)),
    )
    bpy = types.SimpleNamespace(data=types.SimpleNamespace(objects=[head]))
    printed = []
    scope = {
        "bpy": bpy,
        "json": json,
        "mesh": mesh,
        "THRESH": 0.15,
        "proxy_faces": 0,
        "proxy_error": None,
        "proxy_nonempty": 0,
        "KEEP_PROXY": False,
        "print": printed.append,
    }
    exec(section, scope)
    return json.loads(printed[-1])


def test_a_mesh_moved_down_is_not_called_bleeding():
    # Local: spine 0..1, front leg tops out at 0.3. World: everything sits 10 below,
    # so the head marker (world) is at -9. The leg is 0.7 under the head: clean.
    # Comparing local Z (0.3) with the world marker (-9) calls this a bleed.
    out = _verdict(
        _lift(-10.0),
        [((0, 0, 0.0), "spine"), ((0, 0, 1.0), "head"), ((0, 0, 0.3), "front_leg.L")],
        head_marker_world_z=-9.0,
    )
    assert out["legs_reaching_head"] == []
    assert out["verdict"] == "CLEAN"


def test_a_leg_that_reaches_the_head_is_found_on_a_moved_mesh():
    # World: mesh sits 10 up, head marker at 11. The leg vertex is at local 1.0 =
    # world 11.0, so it really reaches the head. Local Z (1.0) is far below the
    # marker (11), so a local-space check misses the bleed.
    out = _verdict(
        _lift(10.0),
        [((0, 0, 0.0), "spine"), ((0, 0, 1.0), "front_leg.L")],
        head_marker_world_z=11.0,
    )
    assert out["legs_reaching_head"] == ["front_leg.L"]
    assert out["verdict"] == "ENVELOPE-LIKE BLEED"


def test_a_rotated_mesh_is_judged_by_world_height():
    # Mesh lies on its back: local Y is world Z. The leg's local Z is 0 everywhere,
    # but its world height reaches the head (world Z 1.0).
    out = _verdict(
        _tilt_up(),
        [((0, 0.0, 0), "spine"), ((0, 1.0, 0), "front_leg.L")],
        head_marker_world_z=1.0,
    )
    assert out["legs_reaching_head"] == ["front_leg.L"]


def _rig_verdict(matrix, vertices, head_marker_world_z):
    """Run the build-rig verification section on the same stand-ins."""
    code = rig.build_code(0.09, 0.15)
    section = code[code.index("# Verification.") :]
    names = sorted({name for _, name in vertices})
    mesh = types.SimpleNamespace(
        matrix_world=matrix,
        vertex_groups=[
            types.SimpleNamespace(index=i, name=n) for i, n in enumerate(names)
        ],
        data=types.SimpleNamespace(
            vertices=[
                types.SimpleNamespace(
                    co=Vec(*co),
                    groups=[types.SimpleNamespace(group=names.index(name), weight=1.0)],
                )
                for co, name in vertices
            ]
        ),
    )
    printed = []
    scope = {
        "json": json,
        "mesh": mesh,
        "THRESH": 0.15,
        "skeleton": rig.SKELETON,
        "ear_bones": rig.EAR_BONES,
        "arm_data": types.SimpleNamespace(bones=[]),
        "heat_error": None,
        "markers": {"head": Vec(0, 0, head_marker_world_z)},
        "print": printed.append,
    }
    exec(section, scope)
    return json.loads(printed[-1])


def test_build_rig_does_not_call_a_moved_down_mesh_bleeding():
    out = _rig_verdict(
        _lift(-10.0),
        [
            ((0, 0, 0.0), "spine_01"),
            ((0, 0, 1.0), "head"),
            ((0, 0, 0.3), "frontL_upperarm"),
        ],
        head_marker_world_z=-9.0,
    )
    assert out["legs_reaching_head"] == []


def test_build_rig_finds_a_bleeding_leg_on_a_moved_up_mesh():
    out = _rig_verdict(
        _lift(10.0),
        [((0, 0, 0.0), "spine_01"), ((0, 0, 1.0), "frontL_upperarm")],
        head_marker_world_z=11.0,
    )
    assert out["legs_reaching_head"] == ["frontL_upperarm"]
    assert out["verdict"] == "ENVELOPE FALLBACK"
