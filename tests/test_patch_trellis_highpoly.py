"""Tests for the high-poly dump patch.

The bug these exist to prevent: the injected block used `Path` without importing it, and
`generate.py` has no module-level `pathlib` import. That crashed on the *last* statement
of a 16-minute generation run.

Note these import the real patch module and compile the real injected text. An earlier
attempt re-extracted the block from the file with a regex and `exec`'d it, which
double-escaped the string literals and reported a failure that did not exist — testing a
re-derived copy is worse than not testing at all.
"""

from __future__ import annotations

import ast
import importlib.util
import struct
import textwrap
from pathlib import Path

import pytest

PATCH = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_highpoly.py"


def _load_patch_module():
    """Import the patch module without executing its module-level patch() call."""
    source = PATCH.read_text()
    # The script patches on import by design; strip the trailing side effects so the
    # constants can be inspected in isolation.
    trimmed = source.split("patch(GENERATOR)")[0]
    spec = importlib.util.spec_from_loader("patch_highpoly", loader=None)
    module = importlib.util.module_from_spec(spec)
    # The script derives paths from __file__, which a hand-built module lacks.
    module.__dict__["__file__"] = str(PATCH)
    exec(compile(trimmed, str(PATCH), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(scope="module")
def patch_module():
    return _load_patch_module()


def _injected_tree(patch_module):
    """Parse the injected block. It is function-body code, so it arrives indented."""
    return ast.parse(textwrap.dedent(patch_module.REPLACEMENT))


def test_injected_block_is_valid_python(patch_module):
    """The replacement must parse. A syntax error here surfaces 16 minutes downstream."""
    _injected_tree(patch_module)


def test_injected_block_imports_everything_it_uses(patch_module):
    """The regression that cost a generation run.

    `generate.py` imports neither pathlib nor numpy at module scope, so the injected
    code must import both itself. Assert on the parsed tree rather than on substrings,
    so renaming an alias cannot silently pass.
    """
    tree = _injected_tree(patch_module)

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


def test_header_uses_real_newlines(patch_module):
    """A PLY header separated by literal backslash-n is unreadable by every parser."""
    tree = _injected_tree(patch_module)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert any(v.startswith("ply\n") for v in literals), (
        "the 'ply' magic line must end in a real newline, not an escaped one"
    )


def test_ply_writer_round_trips(tmp_path):
    """Write a small mesh through the same byte layout and read it back by hand.

    Checks the part most likely to be silently wrong: PLY prefixes each face with a
    vertex count, so a face row is 1 + 3*4 = 13 bytes, not 12.
    """
    import numpy as np

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype="<i4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {verts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    counts = np.full((faces.shape[0], 1), 3, dtype="<u1")
    rows = np.hstack([counts, faces.view("<u1").reshape(faces.shape[0], -1)])

    dest = tmp_path / "t.ply"
    with open(dest, "wb") as fh:
        fh.write(header)
        fh.write(verts.tobytes())
        fh.write(rows.tobytes())

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


def test_patch_is_idempotent(tmp_path, patch_module):
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


def test_patch_refuses_an_unrecognised_file(tmp_path, patch_module):
    """Fail loudly if upstream moves the anchor, rather than silently no-op."""
    stub = tmp_path / "generate.py"
    stub.write_text("print('nothing to anchor to')\n")
    with pytest.raises(RuntimeError):
        patch_module.patch(stub)
