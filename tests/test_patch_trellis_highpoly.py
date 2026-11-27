"""Tests for the high-poly patch and the binary PLY writer it deploys."""

from __future__ import annotations

import ast
import struct
import textwrap

import pytest

from scripts import patch_trellis_highpoly as patch_module
from scripts.trellis_export_compat import write_highpoly


def _injected_tree():
    """Parse the injected block. It is function-body code, so it arrives indented."""
    return ast.parse(textwrap.dedent(patch_module.REPLACEMENT))


def test_injected_block_is_valid_python():
    """The replacement must parse. A syntax error here surfaces 16 minutes downstream."""
    _injected_tree()


def test_injected_block_imports_everything_it_uses():
    """The regression that cost a generation run.

    `generate.py` imports neither pathlib nor numpy at module scope, so the injected
    code must import both itself. Assert on the parsed tree rather than on substrings,
    so renaming an alias cannot silently pass.
    """
    tree = _injected_tree()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update(a.asname or a.name for a in node.names)

    # Every bare name the block calls, minus builtins and names generate.py provides.
    provided = {"args", "verts", "faces", "getattr", "open", "print", "f"}
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    undefined = used - imported - provided - assigned
    assert not undefined, f"injected block uses undefined names: {sorted(undefined)}"


def test_ply_writer_round_trips(tmp_path):
    """Write a small mesh through the same byte layout and read it back by hand.

    Checks the part most likely to be silently wrong: PLY prefixes each face with a
    vertex count, so a face row is 1 + 3*4 = 13 bytes, not 12.
    """
    import numpy as np

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype="<i4")

    dest = tmp_path / "t.ply"
    write_highpoly(dest, verts, faces)

    raw = dest.read_bytes()
    split = raw.index(b"end_header\n") + len(b"end_header\n")
    assert raw.startswith(b"ply\n")

    body = raw[split:]
    vbytes = verts.shape[0] * 3 * 4
    got_verts = np.frombuffer(body[:vbytes], dtype="<f4").reshape(-1, 3)
    assert np.allclose(got_verts, verts)

    face_body = body[vbytes:]
    assert len(face_body) == faces.shape[0] * 13, "a face row is 1 count byte + 3 int32s"
    for i in range(faces.shape[0]):
        row = face_body[i * 13 : (i + 1) * 13]
        assert row[0] == 3
        assert struct.unpack("<3i", row[1:]) == tuple(faces[i])


def test_patch_is_idempotent(tmp_path):
    """Bootstrap re-runs the patch scripts, so applying twice must not duplicate."""
    stub = tmp_path / "generate.py"
    stub.write_text(
        patch_module.OPTION_ANCHOR
        + '        help="skip texture baking",\n    )\n'
        + patch_module.NEEDLE
        + "\n"
    )

    patch_module.patch(stub)
    once = stub.read_text()
    patch_module.patch(stub)
    twice = stub.read_text()

    assert once == twice
    assert once.count('"--dump-highpoly"') == 1
    assert once.count("image-to-3dlab: keep the full-resolution mesh") == 1


def test_patch_refuses_an_unrecognised_file(tmp_path):
    """Fail loudly if upstream moves the anchor, rather than silently no-op."""
    stub = tmp_path / "generate.py"
    stub.write_text("print('nothing to anchor to')\n")
    with pytest.raises(RuntimeError):
        patch_module.patch(stub)
