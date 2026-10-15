"""scripts/patch_trellis_rocm.py: what it changes, and what it refuses to do.

Two separate things go wrong on this route and both are silent if unhandled. The attention
patch half-applied leaves a checkout that imports fine and then dies minutes into a
generation; the build patch half-applied leaves an extension that compiles for the wrong
device, or not at all. So the tests cover the anchors actually being there, a second run
changing nothing, and the patched text still being valid Python.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_trellis_rocm as trellis_rocm

# The upstream lines, as they stand at the pinned commit microsoft/TRELLIS.2 75fbf01 and
# at the ROCm forks' HEAD, so the anchors are exercised against real text rather than a
# shape that happens to match itself.

SPARSE_CONFIG_SOURCE = """\
from typing import Literal
import os


def __from_env():
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    ATTN = 'flash_attn'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']:
        ATTN = env_sparse_attn_backend
    print(f"[SPARSE] Attention backend: {ATTN}")


def set_attn_backend(backend: Literal['xformers', 'flash_attn']):
    global ATTN
    ATTN = backend
"""

DENSE_CONFIG_SOURCE = """\
from typing import Literal
import os


def __from_env():
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    BACKEND = 'flash_attn'
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    print(f"Attention backend: {BACKEND}")


def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend
"""

DENSE_ATTN_SOURCE = """\
import torch
from .. import config


def scaled_dot_product_attention(qkv, kv=None, *args, **kwargs):
    num_all_args = 1 if kv is None else 2
    if config.BACKEND == 'xformers':
        out = xops.memory_efficient_attention(q, k, v)
    elif config.BACKEND == 'sdpa':
        out = sdpa(q, k, v)
    elif config.BACKEND == 'naive':
        out = _naive_sdpa(q, k, v)
    else:
        raise ValueError(f"Unknown attention module: {config.BACKEND}")

    return out
"""

SPARSE_ATTN_SOURCE = """\
import torch
from ... import config


def scaled_dot_product_attention(qkv, kv=None, q_seqlen=None, kv_seqlen=None, *args, **kwargs):
    num_all_args = 1 if kv is None else 2
    if config.ATTN == 'xformers':
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
    elif config.ATTN == 'flash_attn_3':
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
"""

# nvdiffrast nests this flag one level deeper than the other three: it is inside `if _rocm:`.
NVDIFFRAST_SETUP_SOURCE = '''\
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

IS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None

if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    _rocm = os.getenv("ROCM_PATH", "")
    _nvcc = ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__", "-fms-extensions"]
    _nvcc += [f"--offload-arch={a}" for a in archs]
    if _rocm:
        _nvcc += [
            f"--hip-path={_rocm}",
            f"--rocm-path={_rocm}",
            f"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}",
        ]
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"],
        "nvcc": _nvcc,
    }
'''

CUMESH_SETUP_SOURCE = '''\
import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION


def src(path):
    if IS_HIP and path.endswith(".cu"):
        return path[:-3] + ".hip"
    return path


if not IS_HIP:
    cc_flag = ["--use_fast_math"]
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    _rocm = os.getenv("ROCM_PATH", "")
    cc_flag = [f"--offload-arch={a}" for a in archs] + [
        f"--hip-path={_rocm}", f"--rocm-path={_rocm}",
        f"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}",
        "-fms-extensions", "-Wno-c++11-narrowing",
    ]
'''

UPSTREAM_SOURCES = {
    "TRELLIS.2/trellis2/modules/sparse/config.py": SPARSE_CONFIG_SOURCE,
    "TRELLIS.2/trellis2/modules/attention/config.py": DENSE_CONFIG_SOURCE,
    "TRELLIS.2/trellis2/modules/attention/full_attn.py": DENSE_ATTN_SOURCE,
    "TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py": SPARSE_ATTN_SOURCE,
    "extensions/nvdiffrast/setup.py": NVDIFFRAST_SETUP_SOURCE,
    "extensions/CuMesh/setup.py": CUMESH_SETUP_SOURCE,
    "extensions/FlexGEMM/setup.py": CUMESH_SETUP_SOURCE,
    "extensions/o-voxel/setup.py": CUMESH_SETUP_SOURCE,
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A vendor root holding unpatched copies of every file the patch touches."""
    for name, text in UPSTREAM_SOURCES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def read(root: Path, name: str) -> str:
    return (root / name).read_text()


# --- the patch does what it says ------------------------------------------------------------

def test_every_patch_finds_a_real_anchor():
    """No patch may be declared against text that is not in the upstream files.

    A patch whose anchor is a typo would otherwise sit in PATCHES forever, never fire, and
    be discovered only when a build failed on this card hours later.
    """
    for spec in trellis_rocm.PATCHES:
        source = UPSTREAM_SOURCES[spec.file]
        assert source.count(spec.needle) == spec.expected, (
            f"{spec.file}: anchor appears {source.count(spec.needle)} time(s), expected "
            f"{spec.expected}")
        assert spec.proof not in source, f"{spec.file}: the proof phrase is already upstream"
        assert spec.proof, f"{spec.file}: a patch with no proof cannot be checked for"


def test_apply_patches_everything_and_reports_it(root: Path):
    changed = trellis_rocm.apply(root)
    assert len(changed) == len(trellis_rocm.PATCHES)
    assert all(was for _, was in changed), "nothing changed: apply() reported no work done"
    assert trellis_rocm.is_patched(root)


def test_apply_reports_one_entry_per_patch_not_per_file(root: Path):
    """The two config files take two patches each; keying by filename loses one of each."""
    per_file = {}
    for spec in trellis_rocm.PATCHES:
        per_file[spec.file] = per_file.get(spec.file, 0) + 1
    assert max(per_file.values()) > 1, "no file takes two patches, so this test proves nothing"
    assert len(trellis_rocm.apply(root)) == sum(per_file.values())


def test_aule_is_selectable_in_both_attention_paths(root: Path):
    trellis_rocm.apply(root)
    sparse = read(root, "TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py")
    dense = read(root, "TRELLIS.2/trellis2/modules/attention/full_attn.py")

    assert "config.ATTN == 'aule'" in sparse
    assert "config.BACKEND == 'aule'" in dense
    # Both call AuleAttention by name, and both from inside the branch rather than at import
    # time: the package is optional and CUDA installs do not have it.
    assert sparse.count("from aule import flash_attention_triton") == 1
    assert dense.count("from aule import flash_attention_triton") == 1
    assert "flash_attention_triton" not in UPSTREAM_SOURCES[
        "TRELLIS.2/trellis2/modules/attention/full_attn.py"]


def test_aule_config_lists_accept_the_name(root: Path):
    """The name has to be in the allow-list, or it is silently ignored and flash_attn stands.

    That failure is late and confusing: the run starts, and dies inside the first attention
    with a flash-attn import error, never having been asked for flash-attn at all.
    """
    trellis_rocm.apply(root)
    for name in ("TRELLIS.2/trellis2/modules/sparse/config.py",
                 "TRELLIS.2/trellis2/modules/attention/config.py"):
        assert "'aule'" in read(root, name), f"{name}: aule is not in the backend list"


def test_sparse_aule_branch_walks_the_packed_sequences(root: Path):
    """The sparse branch must advance q and k by their own lengths, not q alone."""
    trellis_rocm.apply(root)
    sparse = read(root, "TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py")
    assert "for qn, kn in zip(q_seqlen, kv_seqlen):" in sparse
    assert "q_off += qn" in sparse and "kv_off += kn" in sparse
    assert "torch.cat(outs, dim=0)" in sparse


def test_unknown_backend_still_raises(root: Path):
    """The community forks replace this with a silent SDPA fallback. We must not.

    A backend that fails quietly produces a worse mesh and nobody finds out.
    """
    trellis_rocm.apply(root)
    for name in ("TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py",
                 "TRELLIS.2/trellis2/modules/attention/full_attn.py"):
        assert "raise ValueError(f\"Unknown attention module" in read(root, name), (
            f"{name}: an unknown backend would no longer raise")


def test_patched_text_is_still_valid_python(root: Path):
    """A patch that does not parse is the worst outcome: the file is left unreadable."""
    trellis_rocm.apply(root)
    for name in UPSTREAM_SOURCES:
        path = root / name
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - only on a bad patch
            pytest.fail(f"{name} no longer parses after patching: {exc}")


def test_indentation_follows_the_anchor(root: Path):
    """nvdiffrast nests the flag inside `if _rocm:`; the others do not.

    One fixed indentation is a syntax error in three of the four files, which is exactly
    what the first version of this patch did. The invariant is that the patched line sits at
    the same depth the anchor did, not a guessed number.
    """
    before = {name: read(root, name) for name in UPSTREAM_SOURCES}
    trellis_rocm.apply(root)
    needle = trellis_rocm.DEVICE_LIB_NEEDLE
    depths = set()
    for name in UPSTREAM_SOURCES:
        if not name.endswith("setup.py"):
            continue
        original = next(l for l in before[name].splitlines() if needle in l)
        original_indent = original[: original.index(needle)]
        patched = next(l for l in read(root, name).splitlines()
                       if '"--rocm-device-lib-path=" +' in l)
        assert patched.startswith(original_indent), (
            f"{name}: patched to {patched!r}, anchor was at {original_indent!r}")
        depths.add(len(original_indent))
    assert len(depths) > 1, "all four forks nest it the same way, so this proves nothing"


# --- the device-bitcode expression resolves ------------------------------------------------

def device_lib_expression() -> str:
    """The replacement as it is written into setup.py, as one evaluable expression.

    The trailing comma is the list-element separator, not part of the value, so it comes off
    before eval — otherwise the whole thing evaluates to a one-element tuple.
    """
    block = (trellis_rocm.DEVICE_LIB_REPLACEMENT.replace("%%MARK%%", trellis_rocm.MARKER)
             .replace("%%INDENT%%", " " * 8).replace("%%DEEP%%", " " * 4))
    code = textwrap.dedent("\n".join(line for line in block.splitlines()
                                     if line.strip() and not line.strip().startswith("#")))
    return code.rstrip().rstrip(",")


def test_device_lib_finds_the_rocm7_layout(tmp_path: Path, monkeypatch):
    """ROCm 7.x keeps the bitcode at <root>/amdgcn/bitcode; the forks only knew about 10."""
    root = tmp_path / "rocm"
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    got = eval(device_lib_expression(), {"os": os, "_rocm": str(root)})
    assert got == f"--rocm-device-lib-path={root / 'amdgcn' / 'bitcode'}"


def test_device_lib_finds_the_rocm10_layout(tmp_path: Path, monkeypatch):
    """The forks' original path is still right on ROCm 10, so it must not regress."""
    root = tmp_path / "rocm"
    (root / "lib" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    got = eval(device_lib_expression(), {"os": os, "_rocm": str(root)})
    assert got == f"--rocm-device-lib-path={root / 'lib' / 'llvm' / 'amdgcn' / 'bitcode'}"


def test_device_lib_prefers_the_rocm10_path_when_both_exist(tmp_path: Path, monkeypatch):
    """Both layouts present: the order must not depend on the filesystem."""
    root = tmp_path / "rocm"
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    (root / "lib" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    got = eval(device_lib_expression(), {"os": os, "_rocm": str(root)})
    assert "lib/llvm/amdgcn/bitcode" in got


def test_device_lib_honours_hip_device_lib_path(tmp_path: Path, monkeypatch):
    """HIP_DEVICE_LIB_PATH is hipcc's own variable; if it is set, it wins."""
    root = tmp_path / "rocm"
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", "/somewhere/else")
    got = eval(device_lib_expression(), {"os": os, "_rocm": str(root)})
    assert got == "--rocm-device-lib-path=/somewhere/else"


def test_device_lib_is_empty_when_neither_layout_exists(tmp_path: Path, monkeypatch):
    """No layout found yields an empty path, and clang says so. Loud beats guessing."""
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    got = eval(device_lib_expression(), {"os": os, "_rocm": str(tmp_path / "nope")})
    assert got == "--rocm-device-lib-path="


# --- second runs, and refusals -------------------------------------------------------------

def test_second_run_changes_nothing(root: Path):
    trellis_rocm.apply(root)
    before = {name: read(root, name) for name in UPSTREAM_SOURCES}
    assert not any(was for _, was in trellis_rocm.apply(root)), "a second run re-applied a patch"
    assert before == {name: read(root, name) for name in UPSTREAM_SOURCES}


def test_a_moved_anchor_is_refused_not_skipped():
    """Silently doing nothing is how a route ends up half-built and only noticed at runtime."""
    spec = trellis_rocm.PATCHES[0]
    with pytest.raises(RuntimeError, match="anchor"):
        trellis_rocm.patch_text("def something_else():\n    pass\n", spec, "fixture")


def test_a_duplicated_anchor_is_refused(root: Path):
    """A refactor that duplicates the line means the patch wants re-reading, not doubling."""
    name = "TRELLIS.2/trellis2/modules/sparse/config.py"
    path = root / name
    path.write_text(path.read_text() + "\n" + SPARSE_CONFIG_SOURCE)
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        trellis_rocm.apply(root)


def test_missing_files_are_named_rather_than_silently_skipped(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert trellis_rocm.missing_targets(empty) == sorted(set(UPSTREAM_SOURCES))
    assert not trellis_rocm.is_patched(empty)
    with pytest.raises(RuntimeError, match="cannot patch"):
        trellis_rocm.apply(empty)


def test_is_patched_is_false_when_one_file_is_reverted(root: Path):
    trellis_rocm.apply(root)
    name = "extensions/FlexGEMM/setup.py"
    (root / name).write_text(CUMESH_SETUP_SOURCE)
    assert not trellis_rocm.is_patched(root)
    assert "NOT applied" in trellis_rocm.describe(root)


# --- the command line ----------------------------------------------------------------------

def test_main_exits_nonzero_and_explains_when_the_checkout_is_absent(tmp_path: Path, capsys):
    code = trellis_rocm.main(["--root", str(tmp_path / "absent")])
    assert code == 1
    assert "bootstrap_trellis_rocm.py" in capsys.readouterr().out


def test_check_exits_zero_once_applied(root: Path, capsys):
    trellis_rocm.apply(root)
    assert trellis_rocm.main(["--check", "--root", str(root)]) == 0
    assert "all patches applied" in capsys.readouterr().out


def test_main_reports_a_checkout_it_did_not_patch(root: Path, capsys):
    assert trellis_rocm.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert f"{len(trellis_rocm.PATCHES)} of {len(trellis_rocm.PATCHES)} changed" in out
    assert "already" not in out