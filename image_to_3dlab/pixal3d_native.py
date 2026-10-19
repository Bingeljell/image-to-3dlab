"""Pixal3D's own clean-up at a game face count: the Pixal3D Finish route.

Every Pixal3D run ends with a clean-up (weld, remesh, simplify, unwrap, bake its paint from
its own colour data) aimed at 1,000,000 triangles. Aimed at 20,000 instead it gives the
cleanest game model we have made: no holes, shards or blotches, because it simplifies
Pixal3D's own clean surface and bakes from Pixal3D's own colours rather than copying them
across from the exported model. Measured 2026-10-17 on a gator, a mech and a tank;
written up in journal/instant-meshes-and-ram-2026-10-17.md.

`pixal3d_generate.py` saves that clean-up's input beside the model (`<run>.post.bin`) on
builds that carry scripts/patch_pixal3d_dump_continue.py. Here `post-replay` reruns it at
any face count in seconds, and the result is brought into the pipeline's frame and given
smooth shading. Pixel Match, the normal-map bake and compression then run as for any
Finish.

**The frame.** post-replay writes its mesh as pixal3d.cpp's postprocess holds it, which is
neither glTF's Y-up nor the pipeline's front-facing turn. (x, y, z) -> (x, -z, y) lands it
exactly on the model `pixal3d_generate.py` saved (turned, `glb_turn`); the other obvious
candidate, (-x, -z, -y), lands it on trellis-cli's raw, unturned output instead, and Pixel
Match then paints the front picture on the back. Because a half turn about the vertical
leaves a symmetric silhouette unchanged, `frame_problem` compares every bound, not sizes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from image_to_3dlab.glb_turn import (
    FACING_KEY,
    FLOAT,
    FRONT,
    NODE_TRANSFORMS,
    build_glb,
    parse_glb,
)
from image_to_3dlab.host import executable

REPO = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
POST_REPLAY = executable(PIXAL3D_ROOT / "build", "post-replay")
DUMP_SUFFIX = ".post.bin"

# post-replay's frame -> the pipeline's: a proper rotation (determinant +1), so triangle
# winding and normals stay outward.
PIPELINE_FRAME = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]])

# The frame check asks which way round the clean-up fits the raw model best, not whether it
# fits within a fixed limit: the clean-up smooths tips off (0.9% on the 2026-10-19 gnoll's
# mane) while a wrong half turn can miss by as little as 1.3%. The mistakes it guards
# against, as maps from the converted model's bounds to where the raw model would be:
WRONG_FRAMES = {
    "turned round": np.array([-1.0, 1.0, -1.0]),    # a half turn about the up axis
    "mirrored": np.array([-1.0, 1.0, 1.0]),          # left and right swapped
}
# Only a clear win counts: a symmetric model fits about as well either way round.
CLEAR_WIN = 0.8
# Beyond this, whichever way round, the clean-up is not the raw model's shape at all.
FRAME_TOLERANCE = 0.03


class FrameMismatch(Exception):
    """The clean-up's model does not sit where the raw model does."""

INDEX_TYPES = {5121: "<u1", 5123: "<u2", 5125: "<u4"}


def dump_beside(views: Path) -> Path | None:
    """The clean-up dump saved by the run whose cameras are `views` (`<run>.svviews`)."""
    dump = views.with_suffix(DUMP_SUFFIX)
    return dump if dump.is_file() else None


def replay_available(post_replay: Path = POST_REPLAY) -> bool:
    """Upstream's prebuilt downloads ship trellis-cli without post-replay."""
    return post_replay.is_file()


def replay_command(dump: Path, output: Path, faces: int, atlas: int,
                   post_replay: Path = POST_REPLAY) -> list[str]:
    """The clean-up from `dump`, aimed at `faces` triangles. `--no-remesh`: the dump holds
    the already remeshed surface, written by the same run."""
    return [str(post_replay), str(dump), str(output), "--no-remesh",
            "--faces", str(int(faces)), "--atlas", str(int(atlas))]


def to_pipeline_frame(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ PIPELINE_FRAME.T


def smooth_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, shared by every vertex at the same position.

    post-replay splits vertices along UV seams and gives each its own face's normal, which
    renders faceted. Pooling by position smooths across those splits too, so seams do not
    show as creases. Unit length; a vertex on no face keeps a zero normal.
    """
    positions = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = positions[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 1.0
    keys = np.round(positions / max(extent, 1e-12) * 1e6).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.reshape(-1)
    pooled = np.zeros((group.max() + 1 if len(group) else 0, 3))
    for k in range(3):
        np.add.at(pooled, group[faces[:, k]], face_normals)
    normals = pooled[group]
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-20)


def frame_problem(raw_bounds: np.ndarray, out_bounds: np.ndarray,
                  tolerance: float = FRAME_TOLERANCE) -> str | None:
    """Why the converted model does not sit where the raw model does, or None.

    Every bound is compared, low and high per axis: sizes alone match after a wrong half
    turn, and that is exactly the mistake that paints the front picture on the back. The
    model is wrong when turned round or mirrored it would fit clearly better.
    """
    raw, out = np.asarray(raw_bounds, float), np.asarray(out_bounds, float)
    size = float(np.ptp(raw, axis=0).max()) or 1.0

    def miss(bounds: np.ndarray) -> float:
        return float(np.abs(raw - bounds).max()) / size

    here = miss(out)
    for name, flip in WRONG_FRAMES.items():
        moved = np.sort(out * flip, axis=0)            # a flipped axis swaps its low and high
        if miss(moved) < CLEAR_WIN * here:
            return (f"the clean-up's model fits {miss(moved):.1%} {name} against {here:.1%} as "
                    "it is; its frame is wrong, so Pixel Match would paint the wrong side")
    if here > tolerance:
        return (f"the clean-up's model sits {here:.1%} away from the raw model "
                f"(allowed {tolerance:.1%}); it is not the same shape")
    return None


def _rows(binary: bytearray, document: dict, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if accessor.get("componentType") != FLOAT or accessor.get("type") != "VEC3" or "sparse" in accessor:
        raise ValueError(f"accessor {index} is not plain float VEC3 data")
    view = document["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride", 12)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return np.ndarray((accessor["count"], 3), dtype="<f4", buffer=binary, offset=start,
                      strides=(stride, 4))


def _indices(binary: bytes, document: dict, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    dtype = INDEX_TYPES[accessor["componentType"]]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return np.frombuffer(binary, dtype=dtype, count=accessor["count"], offset=start).reshape(-1, 3)


def convert_replay_glb(data: bytes) -> bytes:
    """post-replay's GLB in the pipeline frame, smooth-shaded and stamped as facing front.

    Edited in place: positions and normals are rewritten where they sit, so textures,
    materials and every other byte are kept exactly. Refuses what it cannot convert
    correctly (node transforms, non-indexed or non-float geometry).
    """
    document, binary = parse_glb(data)
    for node in document.get("nodes", []):
        if any(key in node for key in NODE_TRANSFORMS):
            raise ValueError("GLB has node transforms; expected post-replay's raw output")
    out = bytearray(binary)
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes or "indices" not in primitive:
                raise ValueError("expected an indexed mesh with positions")
            positions = _rows(out, document, attributes["POSITION"])
            moved = to_pipeline_frame(positions)
            positions[:] = moved
            accessor = document["accessors"][attributes["POSITION"]]
            accessor["min"], accessor["max"] = moved.min(0).tolist(), moved.max(0).tolist()
            if "NORMAL" in attributes:
                faces = _indices(bytes(out), document, primitive["indices"])
                _rows(out, document, attributes["NORMAL"])[:] = smooth_normals(moved, faces)
    asset = document.setdefault("asset", {"version": "2.0"})
    asset.setdefault("extras", {})[FACING_KEY] = FRONT
    return build_glb(document, bytes(out))


def glb_bounds(data: bytes) -> np.ndarray:
    """[[min x, y, z], [max x, y, z]] over every primitive's POSITION accessor."""
    document, _ = parse_glb(data)
    lows, highs = [], []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor = document["accessors"][primitive["attributes"]["POSITION"]]
            lows.append(accessor["min"]); highs.append(accessor["max"])
    return np.array([np.min(lows, axis=0), np.max(highs, axis=0)])


def native_retopo(dump: Path, raw_glb: Path, output: Path, faces: int, atlas: int,
                  log: Path, post_replay: Path = POST_REPLAY, runner=subprocess.run) -> dict:
    """Run the clean-up at `faces`, convert it, check it against the raw model, write it.

    Returns what it did, for the run record. Raises FrameMismatch rather than handing
    Pixel Match a model it would paint backwards.
    """
    replayed = output.with_name(output.stem + ".replay.glb")
    with open(log, "w") as handle:
        runner(replay_command(dump, replayed, faces, atlas, post_replay),
               stdout=handle, stderr=subprocess.STDOUT, check=True)
    converted = convert_replay_glb(replayed.read_bytes())
    problem = frame_problem(glb_bounds(raw_glb.read_bytes()), glb_bounds(converted))
    if problem:
        raise FrameMismatch(f"Pixal3D clean-up: {problem}")
    output.write_bytes(converted)
    replayed.unlink(missing_ok=True)
    document, _ = parse_glb(converted)
    triangles = sum(document["accessors"][p["indices"]]["count"] // 3
                    for m in document["meshes"] for p in m["primitives"])
    return {"route": "pixal3d", "dump": dump.name, "faces_requested": int(faces),
            "triangles": int(triangles)}
