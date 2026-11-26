"""Check the packing option passed between TRELLIS and o_voxel."""

from __future__ import annotations

import ast
import textwrap

import pytest

from scripts import patch_ovoxel_pack_options as ovoxel_patch
from scripts import patch_trellis_quality as quality_patch
from scripts.trellis_export_compat import packing_kwargs


@pytest.fixture(scope="module")
def ovoxel():
    return ovoxel_patch


@pytest.fixture(scope="module")
def quality():
    return quality_patch


# --- the two sides must name the same keyword ------------------------------------

KEYWORD = "xatlas_pack_charts_kwargs"


def test_both_patches_use_the_same_keyword(ovoxel):
    """The whole point of the pair. A rename on one side must fail here, not at bake."""
    assert KEYWORD in ovoxel.SIGNATURE_REPLACEMENT
    assert KEYWORD in ovoxel.CALL_REPLACEMENT
    def to_glb(*, xatlas_pack_charts_kwargs):
        return xatlas_pack_charts_kwargs

    assert packing_kwargs(to_glb, True) == {KEYWORD: {"brute_force": True}}
    assert packing_kwargs(to_glb, False) == {KEYWORD: {"brute_force": False}}


def test_forwarded_keyword_is_what_cumesh_accepts(ovoxel):
    """`MtlMesh.uv_unwrap` names it `xatlas_pack_charts_kwargs`; so must we.

    cumesh lives in the TRELLIS venv and is not importable here, so this pins the name
    the vendored source was read from rather than re-deriving it.
    """
    call = ovoxel.CALL_REPLACEMENT
    assert f"{KEYWORD}=dict({KEYWORD} or {{}})" in call


def test_injected_preamble_is_valid_python(quality):
    ast.parse(textwrap.dedent(quality.PACK_PREAMBLE))


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
