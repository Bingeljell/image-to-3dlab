"""Apply Metal backend patches to an upstream-shaped checkout on disk."""

import runpy
import sys

import patch_trellis_metal_backends as patch
import pytest

BVH_WRAPPER = '''"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""
import torch

class BVH:
    def unsigned_distance(self, positions, return_uvw=False, out_id=None, output_dtype=None):
        return self._impl.unsigned_distance(positions, return_uvw, out_id)
'''
BVH_STACK = '''// Depth 24 is sufficient: a balanced 4-ary BVH with 40K leaves has depth ~8,
// so 24 provides ample margin while cutting register pressure vs 32/48.

struct FixedStack24 {
    int elems[24];      // 96 bytes — down from 128/192
    int count = 0;

    void push(int val) {
        if (count < 24) elems[count++] = val;
    }
};
'''
BVH_DISTANCE = "\n".join(line + chr(92) for line in (
    "    float dist;                                                       ",
    "    int idx = closest_triangle(point, nodes, tris, dist);             ",
    "                                                                      ",
    "    distances[tid] = (OUT_T)dist;                                     ",
)) + "\n"
OLD_DISPATCH = '''    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];'''
OLD_LOOKUP = (
    "    uint idx = linear_probing_lookup_{width}(hashmap_keys, hashmap_vals, flat_idx, M);\n"
    "    return udf[idx];\n"
)
POSTPROCESS = '''def to_glb(mesh, decimation_target, verbose, alpha, mask):
    if mesh:
        # Clean up topology before simplification
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)

        # Simplify and clean the remeshed result
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    # Auto-detect transparency from baked alpha values
    alpha_valid = alpha[mask]
    if alpha_valid.size > 0 and alpha_valid.min() < 250:
        alpha_mode = 'BLEND'
        if verbose:
            print(f"Detected transparency (alpha min={alpha_valid.min()}), using BLEND mode")
    else:
        alpha_mode = 'OPAQUE'
    return dict(
        alphaMode=alpha_mode,
        doubleSided=True,
    )
'''
PREVIOUS_LOOKUP = '''class _CPUHashMap:
    """Sparse coordinate lookup for non-CUDA decoders."""
    def __init__(self, grid_size, device='cpu'):
        self._entries = {}
'''


@pytest.fixture
def checkout(tmp_path):
    layout = {
        "deps/mtlbvh/mtlbvh/bvh.py": BVH_WRAPPER,
        "deps/mtlbvh/src/metal/bvh.metal": BVH_STACK + "FixedStack24 stack;\n" * 3 + BVH_DISTANCE,
        "deps/mtlbvh/src/metal_bvh.mm": OLD_DISPATCH,
        "deps/mtlmesh/src/metal/remesh.metal": OLD_LOOKUP.format(width="u32") + OLD_LOOKUP.format(width="u64"),
        "deps/trellis2-apple/o-voxel/o_voxel/postprocess.py": POSTPROCESS,
        "deps/trellis2-apple/o-voxel/o_voxel/convert/flexible_dual_grid.py": PREVIOUS_LOOKUP,
    }
    for name, source in layout.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return tmp_path


def test_metal_apply_writes_all_production_fixes_and_is_idempotent(checkout, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["patch", "--root", str(checkout)])
    with pytest.raises(SystemExit) as entry:
        runpy.run_path(patch.__file__, run_name="__main__")
    assert entry.value.code == 0
    assert capsys.readouterr().out.count("applied :") == 6
    paths = [p for p in checkout.rglob("*") if p.is_file()]
    snapshot = {p: p.read_bytes() for p in paths}
    assert patch.apply(checkout) == []
    assert capsys.readouterr().out.count("present :") == 6
    assert {p: p.read_bytes() for p in paths} == snapshot

    bvh = (checkout / "deps/mtlbvh/src/metal/bvh.metal").read_text()
    assert bvh.count("FixedStack48 stack;") == 3
    assert "closest_triangle_stackless(" in bvh
    native = (checkout / "deps/mtlbvh/src/metal_bvh.mm").read_text()
    assert "@autoreleasepool" in native and "[cmdBuf waitUntilCompleted];" in native
    remesh = (checkout / "deps/mtlmesh/src/metal/remesh.metal").read_text()
    assert remesh.count("if (idx == 0xFFFFFFFFu) return 1.0e30f;") == 2
    output = checkout / "deps/trellis2-apple/o-voxel/o_voxel/postprocess.py"
    assert runpy.run_path(str(output))["to_glb"](None, 1000, False, None, None) == {
        "alphaMode": "OPAQUE", "doubleSided": False,
    }


def test_metal_rejects_missing_dependencies_or_different_stack_layout(checkout, tmp_path):
    with pytest.raises(RuntimeError, match="missing Metal source trees"):
        patch.apply(tmp_path / "absent")
    bvh = checkout / "deps/mtlbvh/src/metal/bvh.metal"
    changed = bvh.read_text().replace("FixedStack24 stack;", "using revised_stack = int;", 1)
    bvh.write_text(changed)
    with pytest.raises(RuntimeError, match="expected three source anchors"):
        patch.patch_mtlbvh_metal(checkout)
    assert bvh.read_text() == changed


def test_metal_bvh_keeps_an_existing_import_and_rejects_repeated_anchor(checkout):
    path = checkout / "deps/mtlbvh/mtlbvh/bvh.py"
    path.write_text("import os\n" + BVH_WRAPPER)
    assert patch.patch_mtlbvh_python(checkout)
    assert path.read_text().count("import os\n") == 1
    assert not patch.patch_mtlbvh_python(checkout)
    source = BVH_WRAPPER + BVH_WRAPPER
    path.write_text(source)
    with pytest.raises(RuntimeError, match="expected one source anchor, found 2"):
        patch.patch_mtlbvh_python(checkout)
    assert path.read_text() == source


@pytest.mark.parametrize(("text", "message"), [
    ('"""Sparse coordinate lookup for non-CUDA decoders."""\n', "missing previous sparse class"),
    ('class _CPUHashMap:\n    """Sparse coordinate lookup for non-CUDA decoders."""\n', "unknown previous sparse class"),
])
def test_metal_refuses_unknown_previous_lookup(checkout, text, message):
    path = checkout / "deps/trellis2-apple/o-voxel/o_voxel/convert/flexible_dual_grid.py"
    path.write_text(text)
    with pytest.raises(RuntimeError, match=message):
        patch.patch_ovoxel_sparse_hashmap(checkout)
    assert path.read_text() == text


@pytest.mark.parametrize(("chunk", "return_uvw", "sizes"), [
    ("0", True, [5]), ("8", False, [5]), ("2", True, [2, 2, 1]), ("2", False, [2, 2, 1]),
])
def test_chunked_bvh_uses_copied_helper_on_cpu(checkout, monkeypatch, chunk, return_uvw, sizes):
    import importlib

    import torch

    from scripts import trellis_bvh_compat as compat

    assert patch.patch_mtlbvh_python(checkout)
    package = checkout / "deps/mtlbvh/mtlbvh"
    (package / "__init__.py").touch()
    installed = package / "_assetfurnace_bvh.py"
    assert installed.read_bytes() == patch.BVH_HELPER_SOURCE.read_bytes()
    assert not patch.patch_mtlbvh_python(checkout)
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.setenv("MTLBVH_QUERY_CHUNK", chunk)
    calls = []

    def query(positions, uvw, out_id):
        calls.append(len(positions))
        assert uvw is return_uvw and out_id == "face-id"
        return positions[:, 0], positions[:, 0].long(), positions if uvw else None

    positions = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    from types import SimpleNamespace

    try:
        cls = importlib.import_module("mtlbvh.bvh").BVH
        bvh = cls()
        bvh._impl = SimpleNamespace(unsigned_distance=query)
        actual = bvh.unsigned_distance(positions, return_uvw, "face-id", torch.float64)
        assert calls == sizes
        calls.clear()
        tracked = compat.chunked_unsigned_distance(query, positions, return_uvw, "face-id", torch.float64)
        assert calls == sizes
        dtype = torch.float32 if sizes == [5] else torch.float64
        expected = (positions[:, 0].to(dtype), positions[:, 0].long(), positions if return_uvw else None)
        for result in (actual, tracked):
            torch.testing.assert_close(result, expected)
    finally:
        for name in ("mtlbvh.bvh", "mtlbvh._assetfurnace_bvh", "mtlbvh"):
            sys.modules.pop(name, None)


def test_bvh_synchronizes_each_completed_mps_chunk_without_gpu(monkeypatch):
    from types import SimpleNamespace

    import torch

    from scripts import trellis_bvh_compat as compat

    class Positions:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.device = SimpleNamespace(type="mps")

        def __getitem__(self, index):
            return self.values[index]

    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    events = []
    cpu_torch = SimpleNamespace(
        empty=lambda shape, *, dtype, device: torch.empty(shape, dtype=dtype),
        float32=torch.float32,
        int64=torch.int64,
        mps=SimpleNamespace(synchronize=lambda: events.append("synchronize")),
    )

    def query(points, return_uvw, out_id):
        assert not return_uvw and out_id is None
        events.append(len(points))
        return points[:, 0], points[:, 0].long(), None

    monkeypatch.setenv("MTLBVH_QUERY_CHUNK", "2")
    monkeypatch.setattr(compat, "torch", cpu_torch)
    result = compat.chunked_unsigned_distance(query, Positions(values), False, None)
    torch.testing.assert_close(result, (values[:, 0], values[:, 0].long(), None))
    assert events == [2, "synchronize", 2, "synchronize", 1, "synchronize"]
