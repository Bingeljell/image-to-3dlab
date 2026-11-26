"""Drive the RigNet wrapper without opening a model or GPU."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rignet_infer


@pytest.fixture
def rig_runtime(tmp_path, monkeypatch):
    quick_dir = tmp_path / "rig" / "quick_start"
    quick_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(rignet_infer, "RIGNET", quick_dir.parent)
    events = []

    def log(name, *args, **kwargs):
        events.append((name, args, kwargs))

    class Network:
        def __init__(self, name, **kwargs):
            self.name = name
            log("network", name, **kwargs)

        def to(self, device):
            log("model-device", self.name, device)

        def eval(self):
            log("eval", self.name)

        def load_state_dict(self, state):
            assert state == self.name
            log("state", self.name)

    def checkpoint(path, *, map_location):
        log("checkpoint", path, map_location)
        return {"state_dict": Path(path).parent.name}

    torch = SimpleNamespace(
        device=str,
        cuda=SimpleNamespace(is_available=lambda: False),
        load=checkpoint,
    )
    data = SimpleNamespace(to=lambda device: log("data-device", device))
    rig = SimpleNamespace(
        normalize=lambda scale, translation: log("normalize", scale, translation.tolist()),
        save=lambda path: Path(path).write_text("rig\n"),
    )

    def joints(*args, **kwargs):
        log("joints", *args, **kwargs)
        return data

    def skeleton(*args, **kwargs):
        log("skeleton", *args, **kwargs)
        return "skeleton"

    def skinning(*args, **kwargs):
        log("skinning", *args, **kwargs)
        return rig

    def transfer(*args):
        log("transfer", *args)
        return rig

    def input_data(path):
        log("data", path)
        return data, "voxel", "geodesic", np.array([1, -2, 3]), 2.5

    quick_start = SimpleNamespace(
        JOINTNET=lambda: Network("gcn_meanshift"),
        ROOTNET=lambda: Network("rootnet"),
        BONENET=lambda: Network("bonenet"),
        SKINNET=lambda **kw: Network("skinnet", **kw),
        create_single_data=input_data,
        predict_joints=joints,
        predict_skeleton=skeleton,
        predict_skinning=skinning,
        tranfer_to_ori_mesh=transfer,
    )

    def read_mesh(path):
        log("read", path)
        mesh = SimpleNamespace(vertices=[(0, 0, 0)] if Path(path).exists() else [])

        def simplify(face_count):
            log("simplify", face_count)
            return mesh

        mesh.simplify_quadric_decimation = simplify
        return mesh

    def write_mesh(path, _mesh):
        log("write", path)
        Path(path).write_text("remesh\n")

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(io=SimpleNamespace(
        read_triangle_mesh=read_mesh, write_triangle_mesh=write_mesh,
    )))
    monkeypatch.setitem(sys.modules, "quick_start", quick_start)
    monkeypatch.setattr(sys, "argv", ["rignet_infer.py", "owl", "--threshold", "0.03", "--bandwidth", "0.05"])
    return quick_dir, events, quick_start, torch


@pytest.mark.parametrize("cached,device", [(False, "cpu"), (True, "cuda:0")])
def test_rignet_selects_device_and_passes_normalized_mesh_across_stages(rig_runtime, cached, device):
    quick_dir, events, quick_start, torch = rig_runtime
    torch.cuda.is_available = lambda: device.startswith("cuda")
    (quick_dir / "owl_ori.obj").write_text("input\n")
    remesh = quick_dir / "owl_remesh.obj"
    if cached:
        remesh.write_text("cached\n")
    rignet_infer.main()
    assert quick_start.device == device
    assert remesh.read_text() == ("cached\n" if cached else "remesh\n")
    assert (quick_dir / "owl_ori_rig.txt").read_text() == "rig\n"
    assert [event[0] for event in events if event[0] in ("read", "simplify", "write")] == (
        [] if cached else ["read", "simplify", "write"]
    )
    if not cached:
        assert ("simplify", (4000,), {}) in events
    assert ("network", ("skinnet",), {"nearest_bone": 5, "use_Dg": True, "use_Lf": True}) in events
    assert [args for kind, args, _ in events if kind == "checkpoint"] == [
        (f"checkpoints/{name}/model_best.pth.tar", device)
        for name in ("gcn_meanshift", "rootnet", "bonenet", "skinnet")
    ]
    stages = [(kind, args, kw) for kind, args, kw in events
              if kind in ("data", "joints", "skeleton", "skinning", "normalize", "transfer")]
    assert [kind for kind, _, _ in stages] == ["data", "joints", "skeleton", "skinning", "normalize", "transfer"]
    normalized = str(Path("quick_start") / "owl_normalized.obj")
    original = str(Path("quick_start") / "owl_ori.obj")
    remeshed = str(Path("quick_start") / "owl_remesh.obj")
    assert stages[0][1] == (remeshed,)
    assert stages[1][1][-1] == 0.03
    assert stages[1][2] == {"bandwidth": 0.05, "mesh_filename": normalized}
    assert stages[2][2] == {"mesh_filename": normalized}
    assert stages[3][1][-1] == normalized and stages[3][2] == {"subsampling": True}
    assert stages[4][1] == (2.5, [-1, 2, -3])
    assert stages[5][1][:2] == (original, remeshed)


def test_rignet_empty_original_stops_before_predicting(rig_runtime, capsys):
    quick_dir, events, _, _ = rig_runtime
    with pytest.raises(SystemExit) as exc:
        rignet_infer.main()
    assert exc.value.code == 1
    assert "owl_ori.obj" in capsys.readouterr().out
    assert not any(event[0] == "data" for event in events)
    assert not list(quick_dir.iterdir())


def test_rignet_script_entrypoint_writes_rig_using_the_cached_remesh(rig_runtime, monkeypatch):
    import os
    import runpy

    quick_dir, events, _, _ = rig_runtime
    (quick_dir / "owl_ori.obj").write_text("input\n")
    (quick_dir / "owl_remesh.obj").write_text("cached mesh\n")
    original_chdir = os.chdir
    script = Path(rignet_infer.__file__)

    def enter_vendor(directory):
        assert directory == script.resolve().parents[1] / "vendor" / "rignet"
        original_chdir(quick_dir.parent)

    with monkeypatch.context() as patch:
        patch.setattr(os, "chdir", enter_vendor)
        runpy.run_path(str(script), run_name="__main__")
    assert (quick_dir / "owl_ori_rig.txt").read_text() == "rig\n"
    assert not any(event[0] in {"read", "simplify", "write"} for event in events)
