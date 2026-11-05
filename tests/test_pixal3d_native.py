"""Tests for the Pixal3D Finish route: Pixal3D's own clean-up at a game face count.

What matters: the model lands in the pipeline's frame (the wrong half turn is the mistake
that paints the front picture on the back, and it is caught), shading is smooth across
UV seams, the GLB is edited in place without losing anything, and the replay is asked for
the right face count.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import pixal3d_native as pn
from image_to_3dlab.glb_turn import FACING_KEY, FRONT, build_glb, parse_glb


def _glb(positions, faces, normals=None, extra_node=None):
    """A minimal indexed GLB: POSITION (+ NORMAL) float VEC3 and uint32 indices."""
    positions = np.asarray(positions, "<f4")
    normals = np.asarray(normals if normals is not None else np.zeros_like(positions), "<f4")
    indices = np.asarray(faces, "<u4").reshape(-1)
    binary = positions.tobytes() + normals.tobytes() + indices.tobytes()
    n = len(positions)
    document = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, **(extra_node or {})}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": n * 12},
            {"buffer": 0, "byteOffset": n * 12, "byteLength": n * 12},
            {"buffer": 0, "byteOffset": n * 24, "byteLength": len(indices) * 4},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
        "images": [{"bufferView": 0, "mimeType": "image/webp"}],
    }
    return build_glb(document, binary)


def _attribute(data, name):
    document, binary = parse_glb(data)
    accessor = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"][name]]
    start = document["bufferViews"][accessor["bufferView"]]["byteOffset"]
    return np.frombuffer(binary, "<f4", accessor["count"] * 3, start).reshape(-1, 3)


# A lopsided wedge: no axis is symmetric, so every wrong frame shows in its bounds.
WEDGE = [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]]
WEDGE_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]


def test_the_frame_is_a_proper_rotation():
    """Determinant +1: no mirror, so triangle winding and normals stay outward."""
    assert np.isclose(np.linalg.det(pn.PIPELINE_FRAME), 1.0)


def test_the_frame_maps_post_replay_axes_onto_the_pipeline():
    """(x, y, z) -> (x, -z, y), measured against pixal3d_generate's turned model."""
    assert np.allclose(pn.to_pipeline_frame([[1, 2, 3]]), [[1, -3, 2]])


def test_a_matching_model_passes_the_frame_check():
    bounds = np.array([[-0.41, -0.49, -0.26], [0.42, 0.45, 0.25]])
    assert pn.frame_problem(bounds, bounds + 0.001) is None


def test_the_wrong_half_turn_is_caught_even_though_sizes_match():
    """2026-10-18: the gnome's first 20k was turned 180 degrees; same sizes, swapped sides."""
    raw = np.array([[-0.407, -0.493, -0.257], [0.424, 0.454, 0.251]])
    turned = np.array([[-0.424, -0.493, -0.251], [0.407, 0.454, 0.257]])
    assert np.allclose(np.ptp(raw, axis=0), np.ptp(turned, axis=0))
    assert "wrong side" in pn.frame_problem(raw, turned)


def test_a_clean_up_that_trims_every_side_still_passes():
    """2026-10-19 gnoll: the clean-up smoothed the back of the mane off by 0.88% and the old
    fixed 0.5% limit failed the Finish. Turned round it would be off by 1.32%: the right
    way round fits better, so it passes."""
    raw = np.array([[-0.4088, -0.4405, -0.1922], [0.4095, 0.4418, 0.1822]])
    cleaned = np.array([[-0.4065, -0.4384, -0.1844], [0.4064, 0.4368, 0.1805]])
    assert pn.frame_problem(raw, cleaned) is None


def test_a_model_far_from_the_raw_one_is_refused_whichever_way_round():
    raw = np.array([[-0.4, -0.4, -0.2], [0.4, 0.4, 0.2]])
    assert "not the same shape" in pn.frame_problem(raw, raw * 1.2)


def test_normals_are_shared_across_a_uv_seam():
    """Two triangles of one flat quad, split at the seam like post-replay writes them."""
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [3, 4, 5]]
    normals = pn.smooth_normals(positions, faces)
    assert np.allclose(normals, [[0, 0, 1]] * 6)


def test_a_corner_gets_one_averaged_normal():
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]]
    faces = [[0, 1, 2], [3, 4, 5]]
    normals = pn.smooth_normals(positions, faces)
    assert np.allclose(normals[0], normals[3])
    assert np.isclose(np.linalg.norm(normals[0]), 1.0)


def test_a_vertex_on_no_face_keeps_a_zero_normal():
    normals = pn.smooth_normals([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], [[0, 1, 2]])
    assert np.allclose(normals[3], 0)


def test_the_conversion_moves_positions_and_updates_bounds():
    data = pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES))
    assert np.allclose(_attribute(data, "POSITION"), pn.to_pipeline_frame(WEDGE))
    assert np.allclose(pn.glb_bounds(data), [[0, -3, 0], [2, 0, 1]])


def test_the_conversion_writes_unit_smooth_normals():
    normals = _attribute(pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES)), "NORMAL")
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_the_converted_model_is_stamped_as_facing_front():
    document, _ = parse_glb(pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES)))
    assert document["asset"]["extras"][FACING_KEY] == FRONT


def test_everything_else_is_kept():
    before, _ = parse_glb(_glb(WEDGE, WEDGE_FACES))
    after, _ = parse_glb(pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES)))
    assert after["images"] == before["images"]
    assert after["bufferViews"] == before["bufferViews"]


def test_node_transforms_are_refused():
    with pytest.raises(ValueError):
        pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES, extra_node={"scale": [2, 2, 2]}))


def test_the_replay_asks_for_the_face_count_and_keeps_the_remesh():
    command = pn.replay_command(Path("run.post.bin"), Path("out.glb"), 20000, 2048,
                                post_replay=Path("post-replay"))
    assert command[:3] == ["post-replay", "run.post.bin", "out.glb"]
    assert command[command.index("--faces") + 1] == "20000"
    assert command[command.index("--atlas") + 1] == "2048"
    assert "--no-remesh" in command


def test_the_dump_is_found_beside_its_cameras(tmp_path):
    views = tmp_path / "run.svviews"
    views.mkdir()
    assert pn.dump_beside(views) is None
    (tmp_path / "run.post.bin").write_bytes(b"dump")
    assert pn.dump_beside(views) == tmp_path / "run.post.bin"


def test_native_retopo_refuses_a_model_in_the_wrong_frame(tmp_path):
    """The safety net: a frame mismatch stops the stage instead of painting backwards."""
    raw = tmp_path / "raw.glb"
    raw.write_bytes(pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES)))
    mirrored = [[-x, y, z] for x, y, z in WEDGE]

    def runner(command, **kwargs):
        Path(command[2]).write_bytes(_glb(mirrored, WEDGE_FACES))

    with pytest.raises(pn.FrameMismatch):
        pn.native_retopo(tmp_path / "run.post.bin", raw, tmp_path / "out.glb", 100, 1024,
                         tmp_path / "replay.log", post_replay=tmp_path / "post-replay",
                         runner=runner)
    assert not (tmp_path / "out.glb").exists()


def test_native_retopo_writes_the_converted_model_and_reports_it(tmp_path):
    raw = tmp_path / "raw.glb"
    raw.write_bytes(pn.convert_replay_glb(_glb(WEDGE, WEDGE_FACES)))

    def runner(command, **kwargs):
        Path(command[2]).write_bytes(_glb(WEDGE, WEDGE_FACES))

    record = pn.native_retopo(tmp_path / "run.post.bin", raw, tmp_path / "out.glb", 100, 1024,
                              tmp_path / "replay.log", post_replay=tmp_path / "post-replay",
                              runner=runner)
    assert (tmp_path / "out.glb").is_file()
    assert record["route"] == "pixal3d" and record["triangles"] == 4
    assert not list(tmp_path.glob("*.replay.glb"))


def test_a_failed_clean_up_puts_its_log_in_the_error(tmp_path):
    """A user's bug report only carries the error, not the log file beside it."""
    def runner(command, stdout=None, **kwargs):
        stdout.write("loading dump\nerror: dump was written by another build\n")
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(pn.ReplayFailed, match="written by another build") as caught:
        pn.native_retopo(tmp_path / "run.post.bin", tmp_path / "raw.glb", tmp_path / "out.glb",
                         100, 1024, tmp_path / "replay.log", post_replay=tmp_path / "post-replay",
                         runner=runner)
    assert "code 1" in str(caught.value)


def test_log_tail_keeps_only_the_last_lines(tmp_path):
    log = tmp_path / "replay.log"
    log.write_text("\n".join(f"line {i}" for i in range(40)))
    tail = pn.log_tail(log, lines=3)
    assert tail == "line 37\nline 38\nline 39"
    assert pn.log_tail(tmp_path / "missing.log") == "(no log written)"


# --- the dump preflight: post-replay says nothing when a read falls short (#192) -----------

def _dump(tmp_path, vertices=3, faces=1, voxels=2, res=512, extra=0, name="run.post.bin"):
    """A clean-up dump with the layout trellis-cli writes: header, verts, faces, coords, pbr6."""
    payload = 12 * vertices + 12 * faces + 36 * voxels
    data = struct.pack("<4i", vertices, faces, voxels, res) + b"\0" * (payload + extra)
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_a_complete_dump_passes_the_preflight(tmp_path):
    assert pn.dump_problem(_dump(tmp_path)) is None


def test_a_dump_short_of_the_size_its_header_asks_for_is_named(tmp_path):
    dump = _dump(tmp_path)
    dump.write_bytes(dump.read_bytes()[:-100])
    problem = pn.dump_problem(dump)
    assert "run.post.bin" in problem and "not written completely" in problem
    assert "V=3 F=1 Mv=2" in problem and "needs 136" in problem


def test_a_file_too_short_for_its_header_is_named(tmp_path):
    dump = tmp_path / "run.post.bin"
    dump.write_bytes(b"dump")
    assert "too short to hold the 16-byte header" in pn.dump_problem(dump)


def test_a_file_that_is_not_a_dump_is_named(tmp_path):
    dump = tmp_path / "run.post.bin"
    dump.write_bytes(struct.pack("<4i", 0, -1, 0, 0) + b"\0" * 64)
    assert "which is not a clean-up dump" in pn.dump_problem(dump)


def test_a_missing_dump_is_left_to_the_runner(tmp_path):
    """Only a dump that exists reaches native_retopo in a run; the tests pass a fake path."""
    assert pn.dump_problem(tmp_path / "absent.post.bin") is None


def test_a_short_dump_is_reported_without_running_post_replay(tmp_path):
    """The whole point: say the file is short instead of surfacing a bare exit code 1."""
    dump = _dump(tmp_path)
    dump.write_bytes(dump.read_bytes()[:-100])

    def runner(*args, **kwargs):
        raise AssertionError("post-replay must not run on a dump it cannot read")

    with pytest.raises(pn.ReplayFailed, match="not written completely"):
        pn.native_retopo(dump, tmp_path / "raw.glb", tmp_path / "out.glb", 100, 1024,
                         tmp_path / "replay.log", post_replay=tmp_path / "post-replay",
                         runner=runner)
    assert not (tmp_path / "out.glb").exists()


def test_an_empty_log_says_the_exit_code_is_all_there_is(tmp_path):
    def runner(command, stdout=None, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(pn.ReplayFailed, match="printed nothing") as caught:
        pn.native_retopo(tmp_path / "run.post.bin", tmp_path / "raw.glb", tmp_path / "out.glb",
                         100, 1024, tmp_path / "replay.log", post_replay=tmp_path / "post-replay",
                         runner=runner)
    assert "(log is empty)" in str(caught.value)
