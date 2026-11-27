"""Test shared rig decisions without loading Blender."""

import numpy as np
import pytest

from scripts import clean_skin_weights, kimodo_retarget


def test_cleaning_uses_bone_distance_and_preserves_weight_total():
    parents = {"upper": None, "lower": "upper", "end": "lower"}
    segments = {
        "upper": (np.array([0, 0, 0]), np.array([0, 1, 0])),
        "lower": (np.array([0, 1, 0]), np.array([0, 2, 0])),
        "end": (np.array([0, 2, 0]), np.array([0, 3, 0])),
    }
    clean = clean_skin_weights.build_vertex_cleaner(parents, segments, max_hops=0)
    assert clean(np.array([0, .5, 0]), {"end": .4, "upper": .2}) == {
        "upper": pytest.approx(.6)
    }
    close = {"upper": .1, "lower": .9}
    assert clean(np.array([0, 1, 0]), close) is close


def test_sole_weights_use_four_strongest_foot_influences():
    pairs = [
        ("toe1", .3), ("toe2", .4), ("toe3", .8), ("toe4", .5), ("toe5", .7),
    ]
    assert kimodo_retarget.sole_weights(pairs) == pytest.approx(
        {"toe3": .8 / 2.4, "toe5": .7 / 2.4, "toe4": .5 / 2.4, "toe2": .4 / 2.4}
    )
    assert kimodo_retarget.sole_weights([("toe1", .19)]) == {}
    assert kimodo_retarget.sole_weights([]) == {}
    assert kimodo_retarget.sole_weights([("toe1", .20)]) == {"toe1": 1.0}


def test_retarget_parser_accepts_paths_and_animation_options(tmp_path):
    rig, motion, output = (tmp_path / name for name in ("rig.glb", "walk.npz", "walk.glb"))
    args = kimodo_retarget.build_parser().parse_args([
        "--rig", str(rig), "--motion", str(motion), "--out", str(output),
        "--loop", "--keep-whole", "--speed", "1.5", "--in-place",
    ])
    assert (args.rig, args.motion, args.out) == (rig, motion, output)
    assert args.loop and args.keep_whole and args.in_place
    assert args.speed == 1.5
    assert not args.no_align
    assert not args.no_ground


@pytest.fixture
def cleanable_mesh():
    from types import SimpleNamespace as Obj

    class Identity:
        def inverted(self):
            return self

        def __matmul__(self, value):
            return value

    class Group:
        def __init__(self, name, index):
            self.name = name
            self.index = index
            self.assignments = {}
            self.removals = []

        def add(self, vertices, weight, mode):
            assert mode == "REPLACE"
            self.assignments.update(dict.fromkeys(vertices, weight))

        def remove(self, vertices):
            self.removals.extend(vertices)

    class Groups(list):
        def get(self, name):
            return next((group for group in self if group.name == name), None)

        def new(self, *, name):
            group = Group(name, len(self))
            self.append(group)
            return group

    upper = Obj(name="upper", parent=None, head_local=[0, 0, 0], tail_local=[0, 1, 0])
    lower = Obj(name="lower", parent=upper, head_local=[0, 1, 0], tail_local=[0, 2, 0])
    arm = Obj(matrix_world=Identity(), data=Obj(bones=[upper, lower]))
    lower_group = Group("lower", 0)
    unrelated = Group("mask", 1)
    vertex = Obj(index=5, co=[0, .5, 0], groups=[Obj(group=0, weight=.75), Obj(group=1, weight=.25)])
    unchanged = Obj(index=6, co=[0, 1.5, 0], groups=[Obj(group=0, weight=1), Obj(group=1, weight=0)])
    mesh = Obj(matrix_world=Identity(), vertex_groups=Groups([lower_group, unrelated]),
               data=Obj(vertices=[vertex, unchanged]))

    return mesh, arm, lower_group, unrelated


def test_clean_mesh_moves_remote_weight_to_nearby_bone(cleanable_mesh):
    mesh, arm, lower_group, unrelated = cleanable_mesh
    assert clean_skin_weights.clean_mesh(mesh, arm, max_hops=0) == 1
    assert mesh.vertex_groups.get("upper").assignments == {5: .75}
    assert lower_group.removals == [5]
    assert unrelated.removals == []


@pytest.fixture
def rig_scene(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace as Obj

    from test_kimodo_retarget import _humanoid, _rest_mats

    class Bones(dict):
        def __iter__(self):
            return iter(self.values())

    class Matrix:
        def inverted(self):
            return np.eye(4)

        def __array__(self, dtype=None, copy=None):
            return np.eye(4, dtype=dtype)

        def to_3x3(self):
            return self

        def normalized(self):
            return np.eye(3)

    class PoseBone:
        def __init__(self):
            self.keys = []

        def keyframe_insert(self, channel, *, frame):
            self.keys.append((channel, frame, np.array(self.matrix_basis)))

    class Objects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    parents, heads, roles = _humanoid(fingers=False)
    rest, order = _rest_mats(parents, heads)
    bones = Bones()
    for name in order:
        parent = bones[parents[name]] if parents[name] is not None else None
        bone = Obj(
            name=name, parent=parent, children=[], matrix_local=rest[name],
            tail_local=heads[name] + [0, .02, 0],
        )
        if parent is not None:
            parent.children.append(bone)
        bones[name] = bone

    arm = Obj(type="ARMATURE", data=Obj(bones=bones), matrix_world=Matrix(),
              pose=Obj(bones=Bones({name: PoseBone() for name in order})))
    helper = Obj(type="MESH", name="Icosphere_hidden")
    unrelated = Obj(type="EMPTY", name="Empty")
    objects = Objects([helper, arm, unrelated])
    scene = Obj(render=Obj())
    calls = []

    def reset(*, use_empty):
        assert use_empty
        calls.append("reset")

    def import_gltf(*, filepath):
        assert filepath == str(tmp_path / "rig.glb")
        calls.append("import")

    def export_gltf(*, filepath, export_format, export_animations):
        assert export_format == "GLB" and export_animations
        assert helper not in objects and unrelated in objects
        calls.append("export")
        Path(filepath).write_bytes(b"adapter")

    def save_blend(*, filepath):
        calls.append("blend")
        Path(filepath).write_bytes(b"blend")

    bpy = Obj(
        data=Obj(objects=objects),
        context=Obj(scene=scene),
        ops=Obj(
            wm=Obj(read_factory_settings=reset, save_as_mainfile=save_blend),
            import_scene=Obj(gltf=import_gltf),
            export_scene=Obj(gltf=export_gltf),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", Obj(Matrix=np.array))
    return bpy, arm, roles, calls


@pytest.mark.parametrize("flags,frame_count,with_skin", [
    (["--loop", "--in-place", "--no-align", "--no-clearance"], 3, False),
    (["--loop", "--no-ground", "--no-align", "--no-clearance"], 2, False),
    (["--loop", "--keep-whole", "--stiff-fingers"], 12, True),
    (["--loop", "--no-ground"], 12, False),
    (["--loop", "--no-ground", "--arm-spread", "10"], 9, False),
    ([], 4, False),
])
def test_retarget_cli_keys_motion_and_exports_without_blender(
    rig_scene, tmp_path, monkeypatch, capsys, flags, frame_count, with_skin
):
    import sys
    from runpy import run_path
    from types import SimpleNamespace as Obj

    bpy, arm, roles, calls = rig_scene
    if with_skin:
        groups = [Obj(index=i, name=roles[role]) for i, role in enumerate(("Lfoot", "Rfoot", "chest"))]
        vertices = [
            Obj(co=[-.1, 0, 0], groups=[Obj(group=0, weight=1)]),
            Obj(co=[.1, 0, 0], groups=[Obj(group=1, weight=1)]),
            Obj(co=[0, 0, 1.65], groups=[Obj(group=2, weight=1)]),
            Obj(co=[0, .1, 1.55], groups=[]),
        ]
        bpy.data.objects.append(Obj(
            name="Skinned", type="MESH", find_armature=lambda: arm, matrix_world=np.eye(4),
            data=Obj(vertices=vertices), vertex_groups=groups,
        ))
    hips = np.zeros((frame_count, 3))
    hips[:, 1] = .9
    hips[:, 2] = np.linspace(0, 1, frame_count)
    joints = np.zeros((frame_count, 77, 3))
    joints[:, 0] = hips
    g = np.broadcast_to(np.eye(3), (frame_count, 77, 3, 3))
    motion = tmp_path / "track.npz"
    np.savez(motion, global_rot_mats=g, posed_joints=joints)
    output, blend = tmp_path / "out" / "rig.glb", tmp_path / "preview.blend"
    argv = [
        "kimodo_retarget", "--rig", str(tmp_path / "rig.glb"), "--motion", str(motion),
        "--out", str(output), "--speed", "1.5", *flags,
    ]
    if flags:
        argv.extend(["--blend", str(blend)])
    monkeypatch.setattr(sys, "argv", argv)
    run_path(kimodo_retarget.__file__, run_name="__main__")
    reported = capsys.readouterr().out
    assert calls[:3] == ["reset", "import", "export"]
    assert output.read_bytes() == b"adapter"
    if flags:
        assert calls[3:] == ["blend"]
        assert blend.read_bytes() == b"blend"
    else:
        assert calls[3:] == []
        assert not blend.exists()
    assert bpy.context.scene.render.fps == 30
    root = arm.pose.bones[roles["hips"]]
    head = arm.pose.bones[roles["head"]]
    assert root.rotation_mode == head.rotation_mode == "QUATERNION"
    assert len(root.keys) == 2 * len(head.keys)
    assert root.keys[0][:2] == ("rotation_quaternion", 1)
    assert root.keys[1][:2] == ("location", 1)
    assert head.keys[1][:2] == ("rotation_quaternion", pytest.approx(1 + 1 / 1.5))
    if frame_count <= 3 and "--loop" in flags:
        assert len(head.keys) == frame_count
        assert bpy.context.scene.frame_end >= 2
        assert "no cycle found" in reported
    if "--in-place" in flags:
        assert "no cycle found" in reported
        assert len(head.keys) == frame_count
        assert root.keys[-1][2][1, 3] == pytest.approx(0)
    else:
        assert root.keys[-1][2][1, 3] > 0


def test_clean_cli_filters_meshes_and_exports_rebalanced_weights(cleanable_mesh, tmp_path, monkeypatch, capsys):
    import sys
    from runpy import run_path
    from types import SimpleNamespace as Obj

    mesh, arm, _lower_group, _unrelated = cleanable_mesh
    mesh.type, mesh.name = "MESH", "Body"
    mesh.find_armature = lambda: arm
    arm.type = "ARMATURE"

    class Objects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    helper = Obj(type="MESH", name="Icosphere_hidden")
    other_mesh = Obj(type="MESH", name="Unrigged", find_armature=lambda: None)
    objects = Objects([mesh, helper, other_mesh, arm])
    source, output = tmp_path / "rig.glb", tmp_path / "output" / "clean.glb"
    calls = []

    def reset(*, use_empty):
        assert use_empty
        calls.append("reset")

    def import_gltf(*, filepath):
        assert filepath == str(source)
        calls.append("import")

    def export_gltf(*, filepath, export_format):
        assert filepath == str(output)
        assert export_format == "GLB"
        assert helper not in objects and other_mesh in objects
        assert mesh.vertex_groups.get("upper").assignments == {5: .75}
        assert output.parent.is_dir()
        output.write_bytes(b"fixture")
        calls.append("export")

    bpy = Obj(data=Obj(objects=objects), ops=Obj(
        wm=Obj(read_factory_settings=reset), import_scene=Obj(gltf=import_gltf),
        export_scene=Obj(gltf=export_gltf),
    ))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", [
        "clean_skin_weights", "--in", str(source), "--out", str(output), "--max-hops", "0",
    ])
    run_path(clean_skin_weights.__file__, run_name="__main__")
    assert calls == ["reset", "import", "export"]
    assert output.read_bytes() == b"fixture"
    assert "cleaned skin weights on 1 vertices" in capsys.readouterr().out


@pytest.mark.parametrize("packed", [False, True])
def test_motion_reader_keeps_same_joints_and_tracks_across_formats(tmp_path, packed):
    path = tmp_path / "motion.npz"
    joints = np.zeros((2, 77, 3))
    joints[:, 0] = [[0, 1, 0], [0, 1, 2]]
    rotations = np.broadcast_to(np.eye(3), (2, 77, 3, 3))
    if packed:
        np.savez(path, quats=kimodo_retarget.matrices_to_quaternions(rotations),
                 joints0=joints[0], hips=joints[:, 0])
    else:
        np.savez(path, global_rot_mats=np.stack([rotations, rotations]),
                 posed_joints=np.stack([joints, joints + 10]))
    read_rotations, read_joints, read_hips = kimodo_retarget.load_motion(path)
    np.testing.assert_allclose(read_rotations, rotations)
    np.testing.assert_allclose(read_joints, joints[0])
    np.testing.assert_allclose(read_hips, joints[:, 0])


@pytest.mark.parametrize(
    "parents,heights,error",
    [
        ({"hips": None, "spine": "hips"}, {"hips": 1, "spine": 2}, "expected 2 legs"),
        (
            {"hips": None, "spine": "hips", "left": "hips", "right": "hips"},
            {"hips": 1, "spine": 2, "left": 0, "right": 0},
            "legs too short",
        ),
    ],
)
def test_mapping_rejects_incomplete_skeletons(parents, heights, error):
    heads = {name: np.array([0, 0, z]) for name, z in heights.items()}
    with pytest.raises(ValueError, match=error):
        kimodo_retarget.map_skintokens_to_soma(parents, heads)


def test_mapping_requires_a_chest_with_two_arms():
    from test_kimodo_retarget import _humanoid

    parents, heads, roles = _humanoid(fingers=False)
    for side in ("L", "R"):
        for part in ("sh", "arm", "fa", "hand"):
            key = roles[side + part]
            parents.pop(key)
            heads.pop(key)
    with pytest.raises(ValueError, match="no bone above the hips forks"):
        kimodo_retarget.map_skintokens_to_soma(parents, heads)


def test_vertical_foot_cannot_be_tilted_by_pitch():
    assert kimodo_retarget.foot_pitch([0, 0, 0], [0, 0, 0]) == 0
    assert kimodo_retarget.foot_pitch([0, 0, 0], [0, 0, 1]) == pytest.approx(np.pi / 2)
    np.testing.assert_array_equal(
        kimodo_retarget.tilt_to_pitch([0, 0, 0], [0, 0, 1], .3), np.eye(3)
    )


def test_quaternion_blend_does_not_turn_between_equivalent_orientations():
    orientation = np.array([.6, .8, 0, 0])
    np.testing.assert_allclose(kimodo_retarget._slerp(orientation, -orientation, .5), orientation)


def test_mixed_clip_grounds_planted_foot_and_leaves_airborne_toe_unchanged():
    from test_kimodo_retarget import _booted_rig

    parents, _heads, roles, rest, tails, order, mapping, up, planted, floor = _booted_rig()
    world, root = planted[0]
    airborne = (world, root + .6 * up)
    frames = [planted[0], airborne]
    out = kimodo_retarget.plant_feet(frames, rest, parents, order, tails, mapping, floor, 1.6, up)
    toe = roles["Ltoe"]
    np.testing.assert_allclose(out[1][0][toe], world[toe])
    np.testing.assert_allclose(out[1][1], airborne[1])
    before = kimodo_retarget.planted_tilt(frames, rest, parents, order, tails, mapping, floor, 1.6, up)
    after = kimodo_retarget.planted_tilt(out, rest, parents, order, tails, mapping, floor, 1.6, up)
    assert after < before / 2


def test_mapping_rejects_a_third_floor_reaching_limb():
    from test_kimodo_retarget import _humanoid

    parents, heads, roles = _humanoid(fingers=False)
    parent = roles["hips"]
    for i, z in enumerate([.95, .5, .02]):
        name = f"extra_leg_{i}"
        parents[name] = parent
        heads[name] = np.array([0, -.2, z])
        parent = name
    with pytest.raises(ValueError, match="expected 2 legs"):
        kimodo_retarget.map_skintokens_to_soma(parents, heads)


def test_loop_metrics_reject_invalid_height_and_keep_a_single_frame():
    frame = ({}, np.zeros(3))
    with pytest.raises(ValueError, match="height must be positive"):
        kimodo_retarget.loop_gap([frame], {}, {}, [], 0)
    assert kimodo_retarget.blend_loop([frame], 0, 1)[0] is frame
    assert kimodo_retarget.trim_duplicate_tail([frame], {}, {}, [], 1)[0] is frame
    assert kimodo_retarget.horizontal_travel([]) == 0
    assert kimodo_retarget.in_place_frames([]) == []
    assert kimodo_retarget.seam_blend_frames(.1, 0) == kimodo_retarget.LOOP_BLEND_FRAMES


@pytest.mark.parametrize("defect", ["short arms", "neck below chest"])
def test_mapping_rejects_false_chest_forks(defect):
    from test_kimodo_retarget import _humanoid

    parents, heads, roles = _humanoid(fingers=False)
    if defect == "short arms":
        for side in ("L", "R"):
            for suffix in ("hand", "fa"):
                name = roles[side + suffix]
                parents.pop(name)
                heads.pop(name)
    else:
        heads[roles["neck"]][2] = 1.23
        heads[roles["head"]][2] = 1.24
    with pytest.raises(ValueError, match="no bone above the hips forks"):
        kimodo_retarget.map_skintokens_to_soma(parents, heads)


def test_foot_measurement_rejects_rig_without_mapped_feet():
    mapping = {"head": "Head"}
    with pytest.raises(ValueError, match="rig maps no foot joints"):
        kimodo_retarget._foot_track([], {}, {}, [], {}, mapping)
    with pytest.raises(ValueError, match="rig maps no foot joints"):
        kimodo_retarget._rest_floor({}, {}, mapping)


def test_single_planted_frame_can_be_leveled():
    from test_kimodo_retarget import _booted_rig

    parents, _heads, _roles, rest, tails, order, mapping, up, frames, floor = _booted_rig(frames=1)
    before = kimodo_retarget.planted_tilt(frames, rest, parents, order, tails, mapping, floor, 1.6, up)
    out = kimodo_retarget.plant_feet(frames, rest, parents, order, tails, mapping, floor, 1.6, up)
    after = kimodo_retarget.planted_tilt(out, rest, parents, order, tails, mapping, floor, 1.6, up)
    assert before > .01
    assert after < .005
