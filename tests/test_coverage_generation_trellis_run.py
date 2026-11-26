"""Exercise TRELLIS with local files and fake tensor libraries."""

import runpy
import subprocess
import sys
from pathlib import Path
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


@pytest.mark.parametrize("eventual_success", [False, True])
def test_mac_precap_retries_crash_empty_mesh_and_corrupt_output(
    eventual_success, monkeypatch, capsys
):
    vertices = Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = Tensor([[0, 1, 2], [0, 2, 1]])
    files = {}
    scratch = []

    def save(data, path):
        assert data == {"vertices": vertices, "faces": faces}
        path.write_bytes(b"fake serialized tensors")

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and not weights_only
        return files[path]

    def run(argv, *, check, capture_output, text):
        assert argv[:3] == [sys.executable, mac.__file__, "--precap-worker"]
        assert check and capture_output and text
        source, dest = map(Path, argv[3:5])
        assert source.is_file()
        assert argv[5] == "1"
        scratch.append(source.parent)
        if len(scratch) == 1:
            raise subprocess.CalledProcessError(11, argv)
        output_faces = {
            2: np.empty((0, 3), dtype=int),
            3: [[0, 1, 99]],
            4: [[0, 1, 2]],
        }[len(scratch)]
        files[dest] = (vertices, Tensor(output_faces))
        return subprocess.CompletedProcess(argv, 0)

    torch = ModuleType("torch")
    torch.save = save
    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(subprocess, "run", run)
    if eventual_success:
        actual_vertices, actual_faces = mac._precap_subprocess(vertices, faces, pre_cap=1)
        assert actual_vertices is vertices
        np.testing.assert_array_equal(actual_faces.numpy(), [[0, 1, 2]])
        assert len(scratch) == 4
    else:
        with pytest.raises(RuntimeError, match="after 3 attempts.*out-of-range faces"):
            mac._precap_subprocess(vertices, faces, pre_cap=1, max_attempts=3)
        assert len(scratch) == 3
    assert not any(path.exists() for path in scratch)
    out = capsys.readouterr().out
    assert "crash/failure" in out
    assert "no faces" in out
    assert "out-of-range faces" in out
    assert ("succeeded on attempt 4" in out) is eventual_success
