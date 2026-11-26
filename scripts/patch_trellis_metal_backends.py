#!/usr/bin/env python3
"""Replay proven image-to-3dlab fixes over pinned Pedro Metal backends.

Pedro's repositories provide the required Metal implementations.  This layer keeps
their clean sources as provenance, then reapplies only production fixes established by
the Fox/Forest/Snag investigations.  Diagnostic logging and checkpoint instrumentation
are intentionally excluded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-space-mac"
BVH_HELPER_SOURCE = Path(__file__).with_name("trellis_bvh_compat.py")
SPARSE_HASHMAP_SOURCE = Path(__file__).with_name("trellis_sparse_hashmap_compat.py")


def replace_once(source: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in source:
        return source, False
    found = source.count(old)
    if found != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {found}")
    return source.replace(old, new, 1), True


def patch_file(path: Path, transform) -> bool:
    source = path.read_text()
    patched, changed = transform(source)
    if changed:
        path.write_text(patched)
    return changed


def patch_mtlbvh_python(root: Path) -> bool:
    path = root / "deps/mtlbvh/mtlbvh/bvh.py"
    return_anchor = "        return self._impl.unsigned_distance(positions, return_uvw, out_id)"
    chunked = '''        from ._assetfurnace_bvh import chunked_unsigned_distance
        return chunked_unsigned_distance(self._impl.unsigned_distance, positions,
                                         return_uvw, out_id, output_dtype)'''
    legacy_start = "        # image-to-3dlab: bound each command buffer on dense TRELLIS meshes."
    legacy_end = "        return distances, face_ids, uvws"

    def transform(source: str) -> tuple[str, bool]:
        if legacy_start in source:
            start = source.index(legacy_start)
            end = source.find(legacy_end, start)
            if end < 0:
                raise RuntimeError("chunked BVH queries: previous patch has changed")
            digest = hashlib.sha256(source[start:end + len(legacy_end)].encode()).hexdigest()
            if digest != "4d0f52499c206dcb3746ce82651c0eba5f7ceea2aa696808c5e78fffac1a6418":
                raise RuntimeError("chunked BVH queries: previous patch has changed")
            return source[:start] + chunked + source[end + len(legacy_end):], True
        return replace_once(source, return_anchor, chunked, "chunked BVH queries")

    source = path.read_text()
    patched, changed = transform(source)
    helper = path.with_name("_assetfurnace_bvh.py")
    expected = BVH_HELPER_SOURCE.read_bytes()
    helper_changed = not helper.is_file() or helper.read_bytes() != expected
    if helper_changed:
        helper.write_bytes(expected)
    if changed:
        path.write_text(patched)
    return changed or helper_changed


def patch_mtlbvh_metal(root: Path) -> bool:
    path = root / "deps/mtlbvh/src/metal/bvh.metal"
    old_stack = '''// Depth 24 is sufficient: a balanced 4-ary BVH with 40K leaves has depth ~8,
// so 24 provides ample margin while cutting register pressure vs 32/48.

struct FixedStack24 {
    int elems[24];      // 96 bytes — down from 128/192
    int count = 0;

    void push(int val) {
        if (count < 24) elems[count++] = val;
    }'''
    new_stack = '''// A depth-d 4-ary traversal can hold 1 + 3d pending nodes.  Production
// TRELLIS meshes exceed the range where 24 entries are safe; silent overflow loses
// branches and inflates nearest-surface distances.
struct FixedStack48 {
    int elems[48];
    int count = 0;

    void push(int val) {
        if (count < 48) elems[count++] = val;
    }'''
    old_distance = '''    float dist;                                                       <EOL>
    int idx = closest_triangle(point, nodes, tris, dist);             <EOL>
                                                                      <EOL>
    distances[tid] = (OUT_T)dist;                                     <EOL>
'''.replace("<EOL>", chr(92))
    new_distance = '''    float dist_sq;                                                    <EOL>
    int idx = closest_triangle_stackless(                             <EOL>
        point, nodes, tris, dist_sq, BVH_MAX_DIST_SQ                  <EOL>
    );                                                                <EOL>
    float dist = sqrt(dist_sq);                                       <EOL>
                                                                      <EOL>
    distances[tid] = (OUT_T)dist;                                     <EOL>
'''.replace("<EOL>", chr(92))

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_once(source, old_stack, new_stack, "48-entry BVH traversal stack")
        changed |= did
        if "FixedStack24 stack;" in source:
            if source.count("FixedStack24 stack;") != 3:
                raise RuntimeError("BVH stack uses: expected three source anchors")
            source = source.replace("FixedStack24 stack;", "FixedStack48 stack;")
            source = source.replace("stack=24", "stack=48")
            changed = True
        source, did = replace_once(
            source, old_distance, new_distance, "stackless unsigned-distance traversal"
        )
        return source, changed | did

    return patch_file(path, transform)


def patch_mtlbvh_autorelease(root: Path) -> bool:
    path = root / "deps/mtlbvh/src/metal_bvh.mm"
    old = '''    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];'''
    new = '''    // Chunked queries create many autoreleased driver objects.  Drain them per
    // dispatch so unified-memory baking does not accumulate command buffers.
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        encode(enc);
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }'''
    return patch_file(path, lambda source: replace_once(source, old, new, "BVH autorelease pool"))


def patch_mtlmesh_hashmap_miss(root: Path) -> bool:
    path = root / "deps/mtlmesh/src/metal/remesh.metal"
    u32_old = '''    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];'''
    u32_new = '''    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);
    // A miss is 0xFFFFFFFF.  Metal returns zero for that OOB read, which falsely
    // classifies the missing vertex as exactly on the surface.
    if (idx == 0xFFFFFFFFu) return 1.0e30f;
    return udf[idx];'''
    u64_old = '''    uint idx = linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];'''
    u64_new = '''    uint idx = linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, M);
    if (idx == 0xFFFFFFFFu) return 1.0e30f;
    return udf[idx];'''

    def transform(source: str) -> tuple[str, bool]:
        source, first = replace_once(source, u32_old, u32_new, "u32 remesh hashmap miss")
        source, second = replace_once(source, u64_old, u64_new, "u64 remesh hashmap miss")
        return source, first | second

    return patch_file(path, transform)


def patch_ovoxel_solid_export(root: Path) -> bool:
    path = root / "deps/trellis2-apple/o-voxel/o_voxel/postprocess.py"
    fill = '''        # Clean up topology before simplification
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)'''
    no_fill = '''        # Clean up topology before simplification.  Metal non-manifold repair
        # creates coincident seam vertices; filling those topological seams adds
        # overlapping caps and visible openings to an already-closed shell.
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)'''
    simplify = '''        # Simplify and clean the remeshed result
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")'''
    simplify_cleanup = '''        # Simplify and clean the remeshed result
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Remove torn fragments left near former non-manifold seams without adding caps.
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)'''
    alpha_old = '''    # Auto-detect transparency from baked alpha values
    alpha_valid = alpha[mask]
    if alpha_valid.size > 0 and alpha_valid.min() < 250:
        alpha_mode = 'BLEND'
        if verbose:
            print(f"Detected transparency (alpha min={alpha_valid.min()}), using BLEND mode")
    else:
        alpha_mode = 'OPAQUE'
    '''
    alpha_new = '''    # The Microsoft demo exports generated objects as opaque.  Padding and
    # unsampled texels legitimately contain alpha=0 and are not evidence that the
    # reconstructed shell should reveal its interior.
    alpha_mode = 'OPAQUE'
    '''

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        source, did = replace_once(source, fill, no_fill, "avoid Metal seam capping")
        changed |= did
        source, did = replace_once(
            source, simplify, simplify_cleanup, "post-simplification topology cleanup"
        )
        changed |= did
        source, did = replace_once(source, alpha_old, alpha_new, "opaque generated material")
        changed |= did
        source, did = replace_once(source, "        doubleSided=True,", "        doubleSided=False,", "backface-cullable material")
        return source, changed | did

    return patch_file(path, transform)


def patch_ovoxel_sparse_hashmap(root: Path) -> bool:
    path = root / "deps/trellis2-apple/o-voxel/o_voxel/convert/flexible_dual_grid.py"
    old = '''class _CPUHashMap:
    """Pure PyTorch hashmap replacement for CUDA _C.hashmap_* functions."""

    def __init__(self, grid_size, device='cpu'):
        D, H, W = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
        self.D, self.H, self.W = D, H, W
        self.table_size = D * H * W
        self.device = device
        # Use int64 flat lookup table
        self.lookup = torch.full((self.table_size,), -1, dtype=torch.long, device=device)

    def _flat_key(self, coords_3d):
        """coords_3d: (..., 3) int tensor of (x, y, z)"""
        return (coords_3d[..., 0].long() * self.H * self.W +
                coords_3d[..., 1].long() * self.W +
                coords_3d[..., 2].long())

    def insert(self, coords_4d):
        """coords_4d: (N, 4) with [batch, x, y, z]. batch is ignored (assumed 0), value = row index."""
        flat = self._flat_key(coords_4d[:, 1:4])
        self.lookup[flat] = torch.arange(coords_4d.shape[0], dtype=torch.long, device=self.device)

    def lookup_3d(self, coords_4d):
        """coords_4d: (M, 4) with [batch, x, y, z]. Returns (M,) indices, 0xffffffff for missing."""
        coords_3d = coords_4d[:, 1:4]
        flat = self._flat_key(coords_3d)
        # Bounds check
        valid = ((coords_3d[..., 0] >= 0) & (coords_3d[..., 0] < self.D) &
                 (coords_3d[..., 1] >= 0) & (coords_3d[..., 1] < self.H) &
                 (coords_3d[..., 2] >= 0) & (coords_3d[..., 2] < self.W))
        flat = flat.clamp(0, self.table_size - 1)
        result = self.lookup[flat]
        result[~valid] = -1
        # Convert -1 to 0xffffffff for compatibility
        result[result < 0] = 0xffffffff
        return result
'''
    replacement = "from .trellis_sparse_hashmap_compat import CPUHashMap as _CPUHashMap\n"
    source = path.read_text()
    if replacement in source:
        patched = source
    elif '"""Sparse coordinate lookup for non-CUDA decoders.' in source:
        # Previous releases embedded the class in the vendor file. The constructor
        # marker self._entries = {} distinguishes it from upstream's dense table.
        source_lines = source.splitlines(keepends=True)
        previous = next(
            (node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
             and node.name == "_CPUHashMap"
             and (ast.get_docstring(node) or "").startswith("Sparse coordinate lookup for non-CUDA decoders.")),
            None,
        )
        if previous is None:
            raise RuntimeError("sparse o_voxel coordinate map: missing previous sparse class")
        start, end = previous.lineno - 1, previous.end_lineno
        if "self._entries = {}" not in "".join(source_lines[start:end]):
            raise RuntimeError("sparse o_voxel coordinate map: unknown previous sparse class")
        patched = "".join([*source_lines[:start], replacement, *source_lines[end:]])
    else:
        patched, _ = replace_once(source, old, replacement, "sparse o_voxel coordinate map")

    helper = path.with_name(SPARSE_HASHMAP_SOURCE.name)
    desired = SPARSE_HASHMAP_SOURCE.read_bytes()
    helper_changed = not helper.is_file() or helper.read_bytes() != desired
    if helper_changed:
        helper.write_bytes(desired)
    if patched != source:
        path.write_text(patched)
    return helper_changed or patched != source


def apply(root: Path) -> list[str]:
    required = (
        root / "deps/mtlbvh",
        root / "deps/mtlmesh",
        root / "deps/trellis2-apple/o-voxel",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing Metal source trees: {', '.join(missing)}")
    steps = (
        ("chunked BVH wrapper", patch_mtlbvh_python),
        ("correct BVH traversal", patch_mtlbvh_metal),
        ("BVH autorelease pool", patch_mtlbvh_autorelease),
        ("remesh hashmap miss guard", patch_mtlmesh_hashmap_miss),
        ("sparse o_voxel coordinate map", patch_ovoxel_sparse_hashmap),
        ("solid o_voxel export", patch_ovoxel_solid_export),
    )
    changed = []
    for label, function in steps:
        if function(root):
            changed.append(label)
            print(f"applied : {label}")
        else:
            print(f"present : {label}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    apply(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
