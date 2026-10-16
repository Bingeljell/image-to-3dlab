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

# nvdiffrast nests the device-bitcode flag one level deeper than a plain list, because it is
# inside `if _rocm:`. It sets no C++ standard at all, so torch's default applies.
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

# dmonkman/CuMesh-ROCm, which this route builds. Unlike upstream it *does* hardcode the
# device-bitcode path and already compiles as C++20, so it needs the device patch but not the
# C++20 one. Kept faithful on both counts: 3 device flags is not the point, the anchors are.
CUMESH_SETUP_SOURCE = '''\
import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION

IS_HIP = bool(IS_HIP_EXTENSION)

if not IS_HIP:
    hip_cxx = ["-O3", "-std=c++20"]
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    _rocm = os.getenv("ROCM_PATH", "")
    cc_flag = [f"--offload-arch={a}" for a in archs] + [
        f"--hip-path={_rocm}", f"--rocm-path={_rocm}",
        f"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}",
        "-fms-extensions", "-Wno-c++11-narrowing",
    ]

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++20", "-EHsc"],
    }
else:
    cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++20", "-fopenmp", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
        "nvcc": ["-O3", "-std=c++20"] + cc_flag,
    }
'''

# Upstream FlexGEMM, at the commit the NVIDIA route pins and this route also uses. It reads
# IS_HIP_EXTENSION and passes --offload-arch already, and it does NOT hardcode the
# device-bitcode path, so the C++20 change is all it needs.
FLEXGEMM_SETUP_SOURCE = '''\
import os
import platform
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
else:
    IS_HIP = BUILD_TARGET == "rocm"

if not IS_HIP:
    cc_flag = ["--use_fast_math", "-allow-unsupported-compiler"]
else:
    archs = os.getenv("GPU_ARCHS", "gfx1030").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc"],
        "nvcc": ["-O3", "-std=c++17", "-Xcompiler=/EHsc"] + cc_flag,
    }
else:
    cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17", "-fopenmp", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }
'''

# Upstream o-voxel, from the TRELLIS.2 checkout rather than extensions/. Two occurrences,
# not three: it has no Windows branch.
OVOXEL_SETUP_SOURCE = '''\
import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION

if IS_HIP_EXTENSION:
    IS_HIP = True
else:
    IS_HIP = False

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup(
    name="o_voxel",
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=["csrc/*.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"] + cc_flag,
            },
        ),
    ],
)
'''

NVDFRAST_PYPROJECT = '''\
[build-system]
requires = ["setuptools>=77", "wheel", "ninja"]
build-backend = "setuptools.build_meta"

[project]
name = "nvdiffrast"
version = "0.4.0+rocm10.0"
description = "nvdiffrast (AMD ROCm fork)"
readme = "README.md"
requires-python = ">=3.12"
license = "LicenseRef-NSCL"
dependencies = ["numpy"]
'''

# The three places windowed_attn.py dispatches on the backend, cut down to the lines around
# each anchor. No `else` in any of them upstream, so an unknown backend is an
# UnboundLocalError rather than a message.
WINDOWED_ATTN_SOURCE = """\
import torch
from .. import config


def calc_window_partition(tensor, window_size, shift_window=0):
    seq_lens = torch.bincount(shifted_indices)
    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        attn_func_args = {
            'attn_bias': xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        }
    elif config.ATTN == 'flash_attn':
        attn_func_args = {
            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),
            'max_seqlen': torch.max(seq_lens)
        }

    return fwd_indices, bwd_indices, seq_lens, attn_func_args


def sparse_windowed_scaled_dot_product_self_attention(qkv, window_size, shift_window=(0, 0, 0)):
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]
    if config.ATTN == 'xformers':
        out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]             # [M, H, C]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]
    return qkv.replace(out)


def sparse_windowed_scaled_dot_product_cross_attention(q, kv, q_window_size, kv_window_size):
    q_feats = q.feats[q_fwd_indices]      # [M, H, C]
    kv_feats = kv.feats[kv_fwd_indices]    # [M, 2, H, C]
    if config.ATTN == 'xformers':
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)[0]               # [M, H, C]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]

    out = out[q_bwd_indices]      # [T, H, C]

    return q.replace(out)
"""

UPSTREAM_SOURCES = {
    "TRELLIS.2/trellis2/modules/sparse/config.py": SPARSE_CONFIG_SOURCE,
    "TRELLIS.2/trellis2/modules/attention/config.py": DENSE_CONFIG_SOURCE,
    "TRELLIS.2/trellis2/modules/attention/full_attn.py": DENSE_ATTN_SOURCE,
    "TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py": SPARSE_ATTN_SOURCE,
    "TRELLIS.2/trellis2/modules/sparse/attention/windowed_attn.py": WINDOWED_ATTN_SOURCE,
    "TRELLIS.2/o-voxel/setup.py": OVOXEL_SETUP_SOURCE,
    "extensions/nvdiffrast/setup.py": NVDIFFRAST_SETUP_SOURCE,
    "extensions/nvdiffrast/pyproject.toml": NVDFRAST_PYPROJECT,
    "extensions/CuMesh/setup.py": CUMESH_SETUP_SOURCE,
    "extensions/CuMesh/pyproject.toml": NVDFRAST_PYPROJECT,
    "extensions/FlexGEMM/setup.py": FLEXGEMM_SETUP_SOURCE,
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


WINDOWED = "TRELLIS.2/trellis2/modules/sparse/attention/windowed_attn.py"


def test_windowed_attention_has_an_aule_branch_in_all_three_places(root: Path):
    """Without it, aule dies on an UnboundLocalError that names neither backend nor file."""
    trellis_rocm.apply(root)
    text = read(root, WINDOWED)
    assert text.count("config.ATTN == 'aule'") == 3
    assert text.count("from aule import flash_attention_triton") == 2
    assert "for qn, kn in zip(q_seq_lens.tolist(), kv_seq_lens.tolist()):" in text


def test_windowed_attention_names_an_unknown_backend(root: Path):
    trellis_rocm.apply(root)
    assert read(root, WINDOWED).count('raise ValueError(f"Unknown attention module') == 3


def test_windowed_patches_are_judged_separately(root: Path):
    """Three patches share one file, so each needs its own proof: a shared one would make
    the second and third read as already applied the moment the first went in."""
    proofs = [spec.proof for spec in trellis_rocm.PATCHES if spec.file == WINDOWED]
    assert len(proofs) == 3 and len(set(proofs)) == 3
    trellis_rocm.apply(root)
    (root / WINDOWED).write_text(WINDOWED_ATTN_SOURCE)
    assert not trellis_rocm.is_patched(root)


def test_the_windowed_anchors_match_the_real_checkout_when_there_is_one():
    """The excerpt above is a copy; when the vendored file exists, check against it too."""
    real = trellis_rocm.DEFAULT_ROOT / WINDOWED
    if not real.is_file():
        pytest.skip("no vendored TRELLIS.2 checkout here")
    text = real.read_text()
    for spec in trellis_rocm.PATCHES:
        if spec.file == WINDOWED:
            assert spec.needle in text or spec.proof in text, spec.proof


def test_patched_text_is_still_valid_python(root: Path):
    """A patch that does not parse is the worst outcome: the file is left unreadable."""
    trellis_rocm.apply(root)
    for name in UPSTREAM_SOURCES:
        if not name.endswith(".py"):
            continue
        path = root / name
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - only on a bad patch
            pytest.fail(f"{name} no longer parses after patching: {exc}")


# --- the Python floor the nvdiffrast port invented ------------------------------------------

def test_the_python_floor_is_lowered(root: Path):
    """uv refuses >=3.12 on this repo's 3.11, and nothing in nvdiffrast needs 3.12."""
    trellis_rocm.apply(root)
    text = read(root, "extensions/nvdiffrast/pyproject.toml")
    assert '">=3.12"' not in text
    assert trellis_rocm.REQUIRES_PYTHON_PROOF in text


def test_the_python_floor_allows_this_interpreter():
    """Guards against a floor that is lowered but still above 3.11."""
    floor = trellis_rocm.REQUIRES_PYTHON_REPLACEMENT.split('"')[1].lstrip(">=")
    major, minor = (int(part) for part in floor.split(".")[:2])
    assert (major, minor) <= (3, 11), f"{floor} still excludes this repo's Python"


def test_indentation_follows_the_anchor(root: Path):
    """nvdiffrast nests the flag inside `if _rocm:`; the others do not.

    One fixed indentation is a syntax error in three of the four files, which is exactly
    what the first version of this patch did. The invariant is that the patched line sits at
    the same depth the anchor did, not a guessed number.
    """
    needle = trellis_rocm.DEVICE_LIB_NEEDLE
    # And the one this route actually patches lands at its own anchor's depth.
    trellis_rocm.apply(root)
    name = "extensions/nvdiffrast/setup.py"
    original = next(l for l in NVDIFFRAST_SETUP_SOURCE.splitlines() if needle in l)
    patched = next(l for l in read(root, name).splitlines()
                   if '"--rocm-device-lib-path=" +' in l)
    assert patched.startswith(original[: original.index(needle)]), repr(patched)


# --- the C++ standard torch 2.14 requires --------------------------------------------------

def test_every_cxx20_occurrence_is_replaced(root: Path):
    """torch 2.14's headers #error below C++20, so a single missed flag fails the build."""
    trellis_rocm.apply(root)
    for path, count in trellis_rocm.CXX20_FILES:
        text = read(root, path)
        assert "-std=c++17" not in text, f"{path}: a C++17 flag survived"
        assert text.count("-std=c++20") == count, (
            f"{path}: {text.count('-std=c++20')} of {count} flags were changed")


def test_nvdiffrast_is_left_alone_by_the_cxx20_patch(root: Path):
    """It sets no standard at all, so torch's own default applies and there is nothing to do."""
    assert "extensions/nvdiffrast/setup.py" not in {p for p, _ in trellis_rocm.CXX20_FILES}
    trellis_rocm.apply(root)
    assert "-std=c++" not in read(root, "extensions/nvdiffrast/setup.py")


def test_the_cxx20_anchor_count_is_what_upstream_has():
    """A refactor that adds or drops a flag means the count wants re-reading, not guessing."""
    for path, count in trellis_rocm.CXX20_FILES:
        assert UPSTREAM_SOURCES[path].count("-std=c++17") == count, path


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


def test_device_lib_refuses_when_neither_layout_exists(tmp_path: Path, monkeypatch):
    """No layout found used to give a bare `--rocm-device-lib-path=` and a compile error that
    never mentioned the bitcode. Now setup.py stops and says where it looked."""
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    with pytest.raises(RuntimeError, match="HIP_DEVICE_LIB_PATH") as refused:
        eval(device_lib_expression(), {"os": os, "_rocm": str(tmp_path / "nope")})
    assert "amdgcn/bitcode" in str(refused.value) and "nope" in str(refused.value)


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
    name = "extensions/nvdiffrast/setup.py"
    (root / name).write_text(NVDIFFRAST_SETUP_SOURCE)
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