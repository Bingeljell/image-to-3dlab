"""Check cleanup replay order and persistence at the CUDA mesh boundary."""

import runpy
import sys
from types import SimpleNamespace

import pytest

from scripts import replay_cleanup_checkpoints


def test_replay_snapshots_every_cleanup_stage_in_order(monkeypatch, tmp_path):
    import torch

    original_file = tmp_path / "checkpoint.pt"
    prefix = tmp_path / "nested" / "snapshot"
    vertices = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]).T
    faces = torch.tensor([[0, 1, 2]])
    torch.save({"vertices": vertices, "faces": faces}, original_file)
    calls = []

    class Mesh:
        num_vertices = 3
        num_faces = 1

        def init(self, points, triangles):
            assert torch.equal(points, vertices)
            assert torch.equal(triangles, faces)
            self.vertices = points
            self.faces = triangles
            self.stage = 0

        def read(self):
            return self.vertices + self.stage, self.faces

        def remove_duplicate_faces(self):
            calls.append(("remove_duplicates",))
            self.stage += 1

        def repair_non_manifold_edges(self):
            calls.append(("repair_nonmanifold",))
            self.stage += 1

        def remove_small_connected_components(self, threshold):
            calls.append(("remove_small_components", threshold))
            self.stage += 1

        def fill_holes(self, *, max_hole_perimeter):
            calls.append(("fill_holes", max_hole_perimeter))
            self.stage += 1

    monkeypatch.setitem(sys.modules, "cumesh", SimpleNamespace(CuMesh=Mesh))
    monkeypatch.setattr(sys, "argv", ["replay_cleanup_checkpoints.py", str(original_file), str(prefix)])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(replay_cleanup_checkpoints.__file__, run_name="__main__")
    assert status.value.code == 0
    assert calls == [
        ("remove_duplicates",), ("repair_nonmanifold",),
        ("remove_small_components", 1e-5), ("fill_holes", 3e-2),
    ]
    for stage, (tag, *_) in enumerate(calls, start=1):
        saved = torch.load(prefix.with_name(f"{prefix.name}_{tag}.pt"), weights_only=True)
        assert torch.equal(saved["vertices"], vertices + stage)
        assert saved["vertices"].is_contiguous()
        assert torch.equal(saved["faces"], faces)
