"""Exercise geometry helpers with small local fixtures."""

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
    if "--in-place" in flags:
        assert "no cycle found" in reported
        assert len(head.keys) == frame_count - 1
        assert root.keys[-1][2][1, 3] == pytest.approx(0)
    else:
        assert root.keys[-1][2][1, 3] > 0
