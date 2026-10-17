"""scripts/patch_sf3d_linux.py: the one line that decides whether SF3D has GPU kernels.

A build that succeeds while quietly producing a CPU-only texture baker is the failure this
covers: `CUDA_HOME` is None on a ROCm build of PyTorch, so upstream's `use_cuda` test goes
false and `baker_kernel.cu` is dropped from the source list with nothing to say so.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_sf3d_linux as gen

# The two anchors as they stand at the pinned commit.
BAKER_SETUP = """\
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension

def get_extensions():
    use_cuda = os.getenv("USE_CUDA", "1" if torch.cuda.is_available() else "0") == "1"
    use_cuda = use_cuda and CUDA_HOME is not None
    return extension
"""

REQUIREMENTS = """\
numpy==1.26.4
huggingface-hub==0.23.4
rembg[gpu]==2.0.57; sys_platform != 'darwin'
rembg==2.0.57; sys_platform == 'darwin'
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "stable-fast-3d"
    for relative, text in ((gen.BAKER_FILE, BAKER_SETUP), (gen.REQUIREMENTS_FILE, REQUIREMENTS)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_a_fresh_checkout_is_patched(checkout: Path):
    assert gen.is_patched(checkout) is False
    assert gen.patch(checkout) == 2
    assert gen.is_patched(checkout) is True


def test_patching_twice_changes_nothing(checkout: Path):
    gen.patch(checkout)
    assert gen.patch(checkout) == 0
    assert gen.is_patched(checkout) is True


def test_patching_ten_times_does_not_stack(checkout: Path):
    """The baker patch keeps its own anchor in its replacement, so checking the anchor
    before the proof would append to it on every run."""
    for _ in range(10):
        gen.patch(checkout)
    text = (checkout / gen.BAKER_FILE).read_text()
    assert text.count("or torch.version.hip is not None") == 1
    assert text.count("use_cuda = use_cuda and") == 1
    assert (checkout / gen.REQUIREMENTS_FILE).read_text().count(gen.REMBG_PIN) == 1


def test_the_gpu_kernels_are_no_longer_dropped(checkout: Path):
    """The point of the patch: on ROCm, CUDA_HOME is None, and that must no longer be
    enough to leave baker_kernel.cu out of the build."""
    gen.patch(checkout)
    text = (checkout / gen.BAKER_FILE).read_text()
    assert gen.BAKER_NEEDLE not in text
    assert "torch.version.hip is not None" in text


def test_the_patched_setup_is_still_valid_python(checkout: Path):
    """A syntax error here fails at pip time with a line number pointing into upstream."""
    gen.patch(checkout)
    ast.parse((checkout / gen.BAKER_FILE).read_text())


def test_the_cuda_only_onnxruntime_extra_is_dropped(checkout: Path):
    """`rembg[gpu]` pulls onnxruntime-gpu, a CUDA wheel that reports no device here.
    SF3D imports rembg but this repo never calls it -- sf3d_backend.py cuts the subject
    out with image_to_3dlab/matte.py first."""
    gen.patch(checkout)
    text = (checkout / gen.REQUIREMENTS_FILE).read_text()
    assert "rembg[gpu]" not in text
    assert gen.REMBG_PIN in text
    # The macOS line is untouched: it was already the CPU wheel.
    assert "rembg==2.0.57; sys_platform == 'darwin'" in text


def test_the_requirements_stay_parseable(checkout: Path):
    gen.patch(checkout)
    lines = [line for line in (checkout / gen.REQUIREMENTS_FILE).read_text().splitlines()
             if line and not line.startswith("#")]
    assert lines, "the patch must not comment out the whole file"
    assert all("rembg[gpu]" not in line for line in lines)


def test_an_anchor_that_moved_fails_loudly(checkout: Path):
    path = checkout / gen.BAKER_FILE
    path.write_text(path.read_text().replace("use_cuda and CUDA_HOME", "use_cuda and _cuda_home"))
    with pytest.raises(RuntimeError, match="anchor for this patch is gone"):
        gen.patch(checkout)
    assert gen.is_patched(checkout) is False


def test_each_patch_names_itself_when_it_fails(checkout: Path):
    (checkout / gen.BAKER_FILE).unlink()
    with pytest.raises(SystemExit, match=gen.BAKER_FILE):
        gen.patch(checkout)


def test_a_folder_that_is_not_a_checkout_says_so(tmp_path: Path):
    with pytest.raises(SystemExit, match="No stable-fast-3d checkout"):
        gen.patch(tmp_path / "elsewhere")


def test_check_reports_without_changing_anything(checkout: Path, capsys):
    assert gen.main(["--check", "--root", str(checkout)]) == 1
    assert "not patched" in capsys.readouterr().out
    gen.patch(checkout)
    assert gen.main(["--check", "--root", str(checkout)]) == 0
    assert "patched" in capsys.readouterr().out

def test_the_linux_rembg_can_make_a_birefnet_session(checkout: Path):
    """rembg 2.0.57 has no BiRefNet: matte.py's request quietly became u2net, fetched
    unasked, while the provenance still named BiRefNet."""
    gen.patch(checkout)
    text = (checkout / gen.REQUIREMENTS_FILE).read_text()
    assert "rembg[cpu]==2.0.69; sys_platform != 'darwin'" in text
    assert "rembg==2.0.57; sys_platform != 'darwin'" not in text


def test_a_checkout_patched_before_the_bump_is_moved_on(checkout: Path):
    path = checkout / gen.REQUIREMENTS_FILE
    path.write_text(path.read_text().replace(
        gen.REQUIREMENTS_NEEDLE,
        "# image-to-3dlab: plain rembg, not [gpu]: old comment\n" + gen.OLD_PATCHED_PIN))
    gen.patch(checkout)
    gen.patch(checkout)
    text = path.read_text()
    assert text.count(gen.REMBG_PIN) == 1 and gen.OLD_PATCHED_PIN not in text
