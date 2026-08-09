"""Tests for the brute-force packing patches.

Two files have to agree for this to work: `patch_ovoxel_pack_options.py` teaches
`o_voxel.postprocess.to_glb` to forward `xatlas_pack_charts_kwargs`, and
`patch_trellis_quality.py` teaches `generate.py` to pass it. If they disagree, the
failure lands at bake time — sixteen minutes into a generation run. So these tests
compile the injected text and check the two sides name the same keyword.

As with `test_patch_trellis_highpoly.py`: import the real patch modules, never a
re-derived copy of their constants.
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OVOXEL_PATCH = SCRIPTS / "patch_ovoxel_pack_options.py"
QUALITY_PATCH = SCRIPTS / "patch_trellis_quality.py"


def _load(path: Path, name: str, cut: str):
    """Import a patch module without executing its module-level patch() call."""
    trimmed = path.read_text().split(cut)[0]
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    exec(compile(trimmed, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(scope="module")
def ovoxel():
    return _load(OVOXEL_PATCH, "patch_ovoxel", 'if __name__ == "__main__":')


@pytest.fixture(scope="module")
def quality():
    return _load(QUALITY_PATCH, "patch_quality", "patch(GENERATOR)")


# --- the two sides must name the same keyword ------------------------------------

KEYWORD = "xatlas_pack_charts_kwargs"


def test_both_patches_use_the_same_keyword(ovoxel, quality):
    """The whole point of the pair. A rename on one side must fail here, not at bake."""
    assert KEYWORD in ovoxel.SIGNATURE_REPLACEMENT
    assert KEYWORD in ovoxel.CALL_REPLACEMENT
    assert KEYWORD in quality.PACK_PREAMBLE


def test_forwarded_keyword_is_what_cumesh_accepts(ovoxel):
    """`MtlMesh.uv_unwrap` names it `xatlas_pack_charts_kwargs`; so must we.

    cumesh lives in the TRELLIS venv and is not importable here, so this pins the name
    the vendored source was read from rather than re-deriving it.
    """
    call = ovoxel.CALL_REPLACEMENT
    assert f"{KEYWORD}=dict({KEYWORD} or {{}})" in call


def test_brute_force_flag_matches_xatlas_pack_options(quality):
    """xatlas' PackOptions field is `brute_force`, not `bruteForce`, via cumesh."""
    assert '"brute_force": args.uv_brute_force' in quality.PACK_PREAMBLE
    assert "bruteForce" not in quality.PACK_PREAMBLE


# --- the injected generator code must be valid and self-sufficient ---------------


def _tree(text: str) -> ast.Module:
    return ast.parse(textwrap.dedent(text))


def test_injected_preamble_is_valid_python(quality):
    _tree(quality.PACK_PREAMBLE)


def test_injected_preamble_imports_what_it_uses(quality):
    """generate.py has no module-level `inspect`; the block must import its own."""
    tree = _tree(quality.PACK_PREAMBLE)
    imported = {
        a.asname or a.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for a in node.names
    }
    assert "_inspect" in imported


def test_injected_preamble_guards_against_an_unpatched_ovoxel(quality):
    """Without the guard an unpatched o_voxel raises TypeError at bake time."""
    tree = _tree(quality.PACK_PREAMBLE)
    tests = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert any(isinstance(t, ast.Compare) and isinstance(t.ops[0], ast.In) for t in tests)


def test_new_option_parses_and_defaults_on(quality):
    """Build the real argparse options and check both spellings of the flag."""
    import argparse

    parser = argparse.ArgumentParser()
    namespace = {"parser": parser}
    exec(compile(textwrap.dedent(quality.PACK_OPTIONS), "<opts>", "exec"), namespace)  # noqa: S102

    assert parser.parse_args([]).uv_brute_force is True
    assert parser.parse_args(["--no-uv-brute-force-packing"]).uv_brute_force is False
    assert parser.parse_args(["--uv-brute-force-packing"]).uv_brute_force is True


# --- patching behaviour -----------------------------------------------------------


def _ovoxel_stub(ovoxel) -> str:
    return (
        "def to_glb(\n"
        + ovoxel.SIGNATURE_NEEDLE
        + "    use_tqdm: bool = False,\n):\n"
        + "    out = mesh.uv_unwrap(\n"
        + "        compute_charts_kwargs={},\n"
        + ovoxel.CALL_NEEDLE
        + "\n"
    )


def test_ovoxel_patch_applies_and_still_parses(tmp_path, ovoxel):
    stub = tmp_path / "postprocess.py"
    stub.write_text(_ovoxel_stub(ovoxel))
    ovoxel.patch(stub)
    patched = stub.read_text()

    ast.parse(patched)
    assert f"{KEYWORD}: dict = None" in patched
    assert f"{KEYWORD}=dict(" in patched


def test_ovoxel_patch_is_idempotent(tmp_path, ovoxel):
    stub = tmp_path / "postprocess.py"
    stub.write_text(_ovoxel_stub(ovoxel))
    ovoxel.patch(stub)
    once = stub.read_text()
    ovoxel.patch(stub)
    assert stub.read_text() == once
    assert once.count(KEYWORD) == 3  # parameter, forwarded name, forwarded value


def test_ovoxel_patch_refuses_an_unrecognised_file(tmp_path, ovoxel):
    stub = tmp_path / "postprocess.py"
    stub.write_text("def to_glb():\n    pass\n")
    with pytest.raises(RuntimeError):
        ovoxel.patch(stub)


def test_ovoxel_patch_reports_a_missing_install(tmp_path, ovoxel):
    with pytest.raises(RuntimeError, match="not installed"):
        ovoxel.find_postprocess(tmp_path)


def _generator_stub(quality) -> str:
    """A generate.py already carrying the earlier quality patch."""
    return (
        quality.OPTION_ANCHOR
        + '        help="skip texture baking",\n    )\n'
        + "                "
        + quality.PACK_ANCHOR.strip()
        + "\n"
        + quality.CALL_REPLACEMENT
        + "\n"
    )


def test_quality_patch_adds_packing_to_an_already_patched_generator(tmp_path, quality):
    """The regression this ordering exists for.

    The vendored generate.py already has the first quality patch applied, so a needle
    keyed on the *unpatched* call would silently skip — leaving the flag unreachable
    while the script reported success.
    """
    stub = tmp_path / "generate.py"
    stub.write_text(_generator_stub(quality))

    quality.patch(stub)
    patched = stub.read_text()

    assert '"--uv-brute-force-packing"' in patched
    assert "**_pack_kwargs," in patched


def test_quality_patch_is_idempotent(tmp_path, quality):
    stub = tmp_path / "generate.py"
    stub.write_text(_generator_stub(quality))

    quality.patch(stub)
    once = stub.read_text()
    quality.patch(stub)

    assert stub.read_text() == once
    assert once.count('"--uv-brute-force-packing"') == 1
    assert once.count("**_pack_kwargs,") == 1


def test_quality_patch_refuses_an_unrecognised_file(tmp_path, quality):
    stub = tmp_path / "generate.py"
    stub.write_text("print('nothing to anchor to')\n")
    with pytest.raises(RuntimeError):
        quality.patch(stub)
