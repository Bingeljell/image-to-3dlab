#!/usr/bin/env python3
"""Replay proven image-to-3dlab fixes over pinned Pedro Metal backends.

Pedro's repositories provide the required Metal implementations.  This layer keeps
their clean sources as provenance, then reapplies only production fixes established by
the Fox/Forest/Snag investigations.  Diagnostic logging and checkpoint instrumentation
are intentionally excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-space-mac"


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
    chunked = '''        # image-to-3dlab: bound each command buffer on dense TRELLIS meshes.
        # Queries are independent, so chunking preserves the exact result.
        chunk_size = int(os.environ.get("MTLBVH_QUERY_CHUNK", "65536"))
        if chunk_size <= 0 or positions.shape[0] <= chunk_size:
            return self._impl.unsigned_distance(positions, return_uvw, out_id)

        scalar_dtype = output_dtype or torch.float32
        distances = torch.empty(positions.shape[0], dtype=scalar_dtype, device=positions.device)
        face_ids = torch.empty(positions.shape[0], dtype=torch.int64, device=positions.device)
        uvws = (
            torch.empty((positions.shape[0], 3), dtype=torch.float32, device=positions.device)
            if return_uvw else None
        )
        for start in range(0, positions.shape[0], chunk_size):
            result = self._impl.unsigned_distance(
                positions[start:start + chunk_size], return_uvw, out_id
            )
            end = start + result[0].shape[0]
            distances[start:end].copy_(result[0])
            face_ids[start:end].copy_(result[1])
            if return_uvw:
                uvws[start:end].copy_(result[2])
            del result
            if positions.device.type == "mps":
                torch.mps.synchronize()
        return distances, face_ids, uvws'''

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        if "import os\n" not in source:
            source, did = replace_once(
                source,
                '"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""\nimport torch',
                '"""Python wrapper for MtlBVH — Metal-accelerated BVH queries."""\nimport os\n\nimport torch',
                "mtlbvh environment import",
            )
            changed |= did
        source, did = replace_once(source, return_anchor, chunked, "chunked BVH queries")
        return source, changed | did

    return patch_file(path, transform)


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
    new = '''class _CPUHashMap:
    """Sparse coordinate lookup for non-CUDA decoders.

    Pedro's dense fallback allocates D*H*W int64 entries, which is about 8 GiB
    for a 1024-cubed decode even though only active voxels are queried.  Keep only
    active coordinates while preserving the CUDA API's 0xFFFFFFFF miss sentinel.
    """

    def __init__(self, grid_size, device='cpu'):
        self.device = device
        self._entries = {}

    def insert(self, coords_4d):
        coords = coords_4d[:, 1:4].detach().cpu().tolist()
        self._entries = {tuple(coord): index for index, coord in enumerate(coords)}

    def lookup_3d(self, coords_4d):
        coords = coords_4d[:, 1:4].detach().cpu().tolist()
        values = [self._entries.get(tuple(coord), 0xFFFFFFFF) for coord in coords]
        return torch.tensor(values, dtype=torch.long, device=self.device)
'''
    return patch_file(path, lambda source: replace_once(source, old, new, "sparse o_voxel coordinate map"))


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
