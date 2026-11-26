"""Exercise TRELLIS with local files and fake tensor libraries."""

import runpy
import sys
from types import ModuleType

import numpy as np
import pytest

from scripts import trellis_space_generate as mac


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, index):
        return Tensor(self.values[index])


def test_mac_precap_worker_simplifies_and_saves_cpu_arrays(tmp_path, monkeypatch):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    src, dest = tmp_path / "in.pt", tmp_path / "out.pt"
    src.touch()
    simplified = []

    def load(path, *, map_location, weights_only):
        assert (path, map_location, weights_only) == (src, "cpu", False)
        return {"vertices": Tensor(vertices), "faces": Tensor(faces)}

    def simplify(v, f, ratio):
        assert v is vertices and f is faces and ratio == 0.5
        simplified.append(ratio)
        return v, f[:1]

    class OutputTensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

    def save(value, path):
        assert path == dest and simplified == [0.5]
        v, f = value
        assert v.dtype == np.float32 and f.dtype == np.int32
        np.testing.assert_array_equal(v, vertices)
        np.testing.assert_array_equal(f, [[0, 1, 2]])
        path.touch()

    torch = ModuleType("torch")
    torch.load = load
    torch.save = save
    torch.from_numpy = lambda data: data.view(OutputTensor)
    native = ModuleType("fast_simplification")
    native.simplify = simplify
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "fast_simplification", native)
    monkeypatch.setattr(sys, "argv", [mac.__file__, "--precap-worker", str(src), str(dest), "1"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(mac.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert dest.is_file()
