"""Exercise MPS diagnostic inputs and report decisions without a GPU."""

import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import measure_bvh_on_surface as surface
from scripts import measure_bvh_precision as precision


@pytest.fixture
def cpu_transport(monkeypatch):
    import torch

    to = torch.Tensor.to
    tensor = torch.tensor
    device_calls = []

    def to_device(value, target, *args, **kwargs):
        if target == "mps":
            device_calls.append(tuple(value.shape))
            target = "cpu"
        return to(value, target, *args, **kwargs)

    def new_tensor(value, *args, **kwargs):
        if kwargs.get("device") == "mps":
            device_calls.append(np.shape(value))
            kwargs["device"] = "cpu"
        return tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_device)
    monkeypatch.setattr(torch, "tensor", new_tensor)
    return torch, device_calls


@pytest.mark.parametrize("bad", [False, True])
def test_surface_cli_queries_triangle_centroids_and_flags_bad_distances(
    monkeypatch, tmp_path, capsys, cpu_transport, bad,
):
    torch, transfers = cpu_transport
    payload = {
        "vertices": torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 3.]]),
        "faces": torch.tensor([[0, 1, 2], [0, 1, 3]]),
    }
    source = tmp_path / "checkpoint.pt"
    torch.save(payload, source)

    class Bvh:
        def __init__(self, vertices, faces):
            assert torch.equal(vertices, payload["vertices"])
            assert faces.dtype == torch.int32

        def unsigned_distance(self, points):
            expected = torch.tensor([[1 / 3, 1 / 3, 0], [1 / 3, 0, 1]])
            torch.testing.assert_close(points, expected)
            return (torch.full((2,), 0.02 if bad else 0.0),)

    monkeypatch.setitem(sys.modules, "mtlbvh", SimpleNamespace(MtlBVH=Bvh))
    monkeypatch.setattr(sys, "argv", ["measure_bvh_on_surface.py", str(source),
                                     "--samples", "10", "--eps", "0.01"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(surface.__file__, run_name="__main__")
    assert status.value.code == int(bad)
    assert transfers == [(4, 3), (2, 3), (2, 3)]
    output = capsys.readouterr().out
    assert "querying 2 exact centroids" in output
    assert ("FAIL:" if bad else "PASS:") in output
    assert f"  over_eps     {float(bad)}" in output


@pytest.mark.parametrize("bad", [False, True])
def test_bvh_precision_cli_uses_thin_shell_and_reports_error_against_analytic_sphere(
    monkeypatch, capsys, cpu_transport, bad,
):
    torch, transfers = cpu_transport
    radius = 0.25

    class Bvh:
        def __init__(self, vertices, faces):
            assert len(vertices) == 12 and len(faces) == 20
            torch.testing.assert_close(vertices.norm(dim=1), torch.full((12,), radius))

        def unsigned_distance(self, points):
            adjusted_distances = (points.norm(dim=1) - radius).abs()
            assert float(adjusted_distances.max()) <= 0.01
            return (adjusted_distances + (0.02 if bad else 0.0),)

    monkeypatch.setitem(sys.modules, "mtlbvh", SimpleNamespace(MtlBVH=Bvh))
    monkeypatch.setattr(sys, "argv", ["measure_bvh_precision.py", "--subdivisions", "0",
                                     "--samples", "6", "--radius", str(radius), "--eps", "0.01"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(precision.__file__, run_name="__main__")
    assert status.value.code == 0
    assert transfers == [(12, 3), (20, 3), (6, 3)]
    output = capsys.readouterr().out
    assert "20 faces" in output
    assert ("Crossing tests will flip" if bad else "BVH is precise relative to eps") in output
