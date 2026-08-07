#!/usr/bin/env python3
"""Export TRELLIS' full-resolution mesh before it is simplified away.

TRELLIS generates on the order of six million triangles and the Mac port immediately
decimates to at most 200,000 before texture baking, to avoid an `mtlbvh` crash. The
discarded 98% is not noise — it is real surface detail we already spent the generation
budget producing, and it is exactly what a **normal map** needs as its bake source.

The usual normal-map workflow requires sculpting a high-poly model to bake from. Here it
already exists; it is simply thrown away. This patch writes it to disk first.

Binary PLY, written by hand with numpy: the generator's environment is not guaranteed to
have trimesh, and PLY at this vertex count is far smaller and faster to write than OBJ.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "vendor" / "trellis-mac" / "generate.py"

OPTION_ANCHOR = '    parser.add_argument(\n        "--no-texture", action="store_true",\n'
OPTION = (
    "    parser.add_argument(\n"
    '        "--dump-highpoly", default=None,\n'
    '        help="Write the unsimplified mesh here as binary PLY, before decimation. '
    'This is the bake source for a normal map",\n'
    "    )\n"
)

NEEDLE = (
    '    print(f"\\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")'
)

REPLACEMENT = '''    print(f"\\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")

    # image-to-3dlab: keep the full-resolution mesh before decimation. It is the bake
    # source for a normal map, and the only moment it exists on disk.
    if getattr(args, "dump_highpoly", None):
        import numpy as _np

        _hp = Path(args.dump_highpoly)
        _hp.parent.mkdir(parents=True, exist_ok=True)
        _v = _np.ascontiguousarray(verts, dtype="<f4")
        _f = _np.ascontiguousarray(faces, dtype="<i4")
        _header = (
            "ply\\n"
            "format binary_little_endian 1.0\\n"
            f"element vertex {_v.shape[0]}\\n"
            "property float x\\nproperty float y\\nproperty float z\\n"
            f"element face {_f.shape[0]}\\n"
            "property list uchar int vertex_indices\\n"
            "end_header\\n"
        ).encode("ascii")
        # PLY stores a per-face vertex count, so each triangle is prefixed with a 3.
        _counts = _np.full((_f.shape[0], 1), 3, dtype="<u1")
        _face_rows = _np.hstack(
            [_counts.view("<u1"), _f.view("<u1").reshape(_f.shape[0], -1)]
        )
        with open(_hp, "wb") as _fh:
            _fh.write(_header)
            _fh.write(_v.tobytes())
            _fh.write(_face_rows.tobytes())
        print(f"  high-poly written: {_hp} ({_v.shape[0]:,} verts, {_f.shape[0]:,} tris)")'''


def patch(path: Path) -> None:
    source = path.read_text()

    if '"--dump-highpoly"' not in source:
        if OPTION_ANCHOR not in source:
            raise RuntimeError(f"expected TRELLIS CLI option anchor not found in {path}")
        source = source.replace(OPTION_ANCHOR, OPTION + OPTION_ANCHOR)

    if "image-to-3dlab: keep the full-resolution mesh" not in source:
        if NEEDLE not in source:
            raise RuntimeError(f"expected mesh-report line not found in {path}")
        source = source.replace(NEEDLE, REPLACEMENT)

    path.write_text(source)


patch(GENERATOR)
print("Patched TRELLIS to optionally dump the unsimplified mesh (--dump-highpoly).")
