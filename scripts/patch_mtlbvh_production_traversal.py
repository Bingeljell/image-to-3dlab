#!/usr/bin/env python3
"""Apply the production-scale MtlBVH traversal and dispatch-lifetime fixes.

The stock Metal traversal uses a 24-entry stack. Deep four-way BVHs built from TRELLIS
decodes can require roughly 34 pending nodes, and `push` silently discards the excess.
Unsigned-distance queries therefore use the BVH's escape indices (stackless); other query
paths retain child sorting with a 48-entry stack.

The CPU/unified-memory dispatch also needs a local autorelease pool because a remesh can
submit thousands of chunked command buffers in one Python call.

`vendor/` is ignored, so run this after bootstrap, then rebuild both the metallib and native
extension with `scripts/rebuild_mtlbvh_metallib.sh` and
`scripts/rebuild_mtlbvh_native.sh`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METAL = REPO / "vendor/trellis-mac/deps/mtlbvh/src/metal/bvh.metal"
NATIVE = REPO / "vendor/trellis-mac/deps/mtlbvh/src/metal_bvh.mm"

STACK_NOTE = """// A depth-d 4-ary traversal can have 1 + 3d pending nodes. The production TRELLIS
// decodes contain 12-28M triangles (roughly 1.5-3.5M leaves at 8 triangles/leaf),
// giving depth ~11 and a worst-case stack occupancy of 34. A 24-entry stack silently
// dropped live branches and returned inflated nearest-surface distances on those meshes.
// 48 covers the production depth with margin while keeping the fixed-size allocation.

"""

STACKLESS = """// Stackless closest triangle
inline int closest_triangle_stackless(
    float3 point,
    device const BvhNode* nodes,
    device const BvhTriangle* tris,
    thread float& out_dist_sq,
    float max_dist_sq
) {
    float shortest_dist_sq = max_dist_sq;
    int shortest_idx = -1;

    int idx = 0;
    while (idx != -1) {
        device const BvhNode& node = nodes[idx];
        float dbb = bb_distance_sq(node.bb, point);
        if (dbb > shortest_dist_sq) {
            idx = node.escape_idx;
            continue;
        }
        if (node.left_idx < 0) {
            int end = -node.right_idx - 1;
            for (int i = -node.left_idx - 1; i < end; ++i) {
                float dsq = tri_distance_sq(tris[i].a, tris[i].b, tris[i].c, point);
                if (dsq <= shortest_dist_sq) {
                    shortest_dist_sq = dsq;
                    shortest_idx = i;
                }
            }
            idx = node.escape_idx;
        } else {
            idx = node.left_idx;
        }
    }

    if (shortest_idx == -1) {
        shortest_idx = 0;
        shortest_dist_sq = 0.0f;
    }
    out_dist_sq = shortest_dist_sq;
    return shortest_idx;
}

"""

UNSIGNED_OLD = """    float dist;                                                       \\
    int idx = closest_triangle(point, nodes, tris, dist);             \\
"""
UNSIGNED_NEW = """    float dist_sq;                                                    \\
    int idx = closest_triangle_stackless(                             \\
        point, nodes, tris, dist_sq, BVH_MAX_DIST_SQ                  \\
    );                                                                \\
    float dist = sqrt(dist_sq);                                       \\
"""

CPU_OLD = """#endif
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}"""
CPU_NEW = """#endif
    // `to_glb` supplies CPU tensors backed by unified memory, so production remeshing
    // takes this path. Chunked multi-million-point queries call it hundreds of times.
    // `commandBuffer` and `computeCommandEncoder` are autoreleased; without a local pool
    // those driver objects accumulate for the lifetime of the Python call and Metal kills
    // the process part-way through the resolution-512 grid level.
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        encode(enc);
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}"""


def apply_metal(source: str) -> str:
    if "closest_triangle_stackless" in source:
        return source
    anchor = "// ─── BVH traversal ──────────────────────────────────────────────────────────\n\n"
    if anchor not in source or UNSIGNED_OLD not in source:
        raise RuntimeError("bvh.metal anchors changed; inspect before patching")
    source = source.replace("struct FixedStack24 {", "struct FixedStack48 {", 1)
    source = source.replace("    int elems[24];", "    int elems[48];", 1)
    source = source.replace("        if (count < 24)", "        if (count < 48)", 1)
    source = source.replace("FixedStack24 stack;", "FixedStack48 stack;")
    source = source.replace(
        "// ─── Fixed-size stack ────────────────────────────────────────────────────────\n",
        "// ─── Fixed-size stack ────────────────────────────────────────────────────────\n" + STACK_NOTE,
        1,
    )
    source = source.replace(anchor, anchor + STACKLESS, 1)
    source = source.replace(
        "// Stack-based closest triangle (with sorting network)",
        "// Stack-based closest triangle (with sorting network, stack=48)",
        1,
    )
    return source.replace(UNSIGNED_OLD, UNSIGNED_NEW, 1)


def revert_metal(source: str) -> str:
    if "closest_triangle_stackless" not in source:
        return source
    if STACKLESS not in source or UNSIGNED_NEW not in source:
        raise RuntimeError("bvh.metal patch differs from the recorded version")
    source = source.replace(STACKLESS, "", 1).replace(UNSIGNED_NEW, UNSIGNED_OLD, 1)
    source = source.replace(STACK_NOTE, "", 1)
    source = source.replace("struct FixedStack48 {", "struct FixedStack24 {", 1)
    source = source.replace("    int elems[48];", "    int elems[24];", 1)
    source = source.replace("        if (count < 48)", "        if (count < 24)", 1)
    source = source.replace("FixedStack48 stack;", "FixedStack24 stack;")
    return source.replace(
        "// Stack-based closest triangle (with sorting network, stack=48)",
        "// Stack-based closest triangle (with sorting network)",
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    if not METAL.is_file() or not NATIVE.is_file():
        print("mtlbvh source not found; bootstrap vendor/trellis-mac first", file=sys.stderr)
        return 2

    metal = METAL.read_text()
    native = NATIVE.read_text()
    applied = "closest_triangle_stackless" in metal and CPU_NEW in native
    absent = "closest_triangle_stackless" not in metal and CPU_OLD in native
    if not (applied or absent):
        raise RuntimeError("MtlBVH is partially patched or its anchors changed")

    if args.check:
        print("APPLIED" if applied else "ABSENT")
        return 0

    if args.revert:
        METAL.write_text(revert_metal(metal))
        NATIVE.write_text(native.replace(CPU_NEW, CPU_OLD, 1))
        print("reverted MtlBVH production traversal fixes")
    else:
        METAL.write_text(apply_metal(metal))
        NATIVE.write_text(native.replace(CPU_OLD, CPU_NEW, 1) if absent else native)
        print("applied MtlBVH production traversal fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
