"""install.ps1: what can be read without running it.

Not in test_install_script.py, whose module is skipped on Windows, the one system that
runs install.ps1.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "install.ps1"


def test_the_windows_installer_exports_the_lock_as_install_sh_syncs_it():
    ps1 = PS1.read_text(encoding="utf-8")
    sh = (REPO / "install.sh").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev --inexact --python 3.11" in sh
    assert "uv export --frozen --no-dev" in ps1
    assert ps1.index("uv export --frozen --no-dev") < ps1.index("& uv pip install --quiet --python $Py -r $Reqs")
    assert "requirements.lock" not in ps1
    assert "requirements.txt" not in ps1


def test_cuda_pytorch_is_installed_at_the_locks_versions():
    """Otherwise PyTorch's index hands over its newest torch, and the lock then swaps it
    for PyPI's CPU-only wheel of the pinned version."""
    ps1 = PS1.read_text(encoding="utf-8")
    patterns = re.findall(r"-Pattern '(\^torch(?:vision)?==\(\[\^ ;\]\+\))'", ps1)
    assert len(patterns) == 2
    assert '"torch==$TorchVersion", "torchvision==$VisionVersion"' in ps1
    assert ps1.index("@Torch") < ps1.index("-r $Reqs")
    assert "uv export --frozen --no-dev" in ps1


def test_a_missing_torch_pin_keeps_the_unpinned_install():
    """Reading a pin from no match crashed under ErrorActionPreference Stop. The match is
    checked first, and PyTorch is installed unpinned, only when it lacks CUDA."""
    ps1 = PS1.read_text(encoding="utf-8")
    assert "Select-Object -First 1).Matches" not in ps1
    assert "if ($TorchPin) { $TorchPin.Matches[0]" in ps1 and "if ($VisionPin) { $VisionPin.Matches[0]" in ps1
    assert "$Pinned = [bool]($TorchVersion -and $VisionVersion)" in ps1
    unpinned = "--index-url $TorchIndex torch torchvision"
    assert ps1.count(unpinned) == 1
    assert "if ((-not $Pinned -or $PinFailed) -and -not $TorchHasCuda)" in ps1[:ps1.index(unpinned)]


def test_a_failed_pinned_pytorch_warns_and_drops_torch_from_the_lock():
    """The pinned CUDA build can be missing from the driver's index. Then the install goes
    on with the newest CUDA build, and the lock minus torch so it cannot swap in the CPU
    wheel."""
    ps1 = PS1.read_text(encoding="utf-8")
    torch_block = ps1[ps1.index("$TorchIndex = ("):ps1.index("-r $Reqs")]
    assert "Die " not in torch_block
    failed = torch_block.index("$PinFailed = ($LASTEXITCODE -ne 0)")
    fallback = torch_block[failed:]
    assert 'Say "Warning: PyTorch' in fallback
    assert "without-cuda" in fallback
    assert fallback.index("without-cuda") < fallback.index("--index-url $TorchIndex torch torchvision")
