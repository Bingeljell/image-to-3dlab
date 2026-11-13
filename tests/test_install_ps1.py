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


def test_a_release_from_before_uv_lock_installs_from_its_requirements_lock():
    """The installer is served from main but checks out the newest release. On 2026-11-08
    main moved to uv.lock while v0.5.7 still had requirements.lock: `uv export` found no
    pyproject.toml and the next line crashed reading a file it never wrote."""
    ps1 = PS1.read_text(encoding="utf-8")
    has_lock = 'if (Test-Path -LiteralPath (Join-Path $Dir "uv.lock"))'
    assert has_lock in ps1
    branch = ps1[ps1.index(has_lock):ps1.index("$TorchPin = ")]
    assert branch.index("uv export --frozen --no-dev") < branch.index("} else {")
    old = branch[branch.index("} else {"):]
    assert '$Reqs = Join-Path $Dir "requirements.lock"' in old
    assert '$Reqs = Join-Path $Dir "requirements.txt"' in old


def test_a_failed_export_stops_with_a_message_not_a_powershell_trace():
    ps1 = PS1.read_text(encoding="utf-8")
    after = ps1[ps1.index("& uv export --frozen --no-dev"):]
    assert after.index("if ($LASTEXITCODE -ne 0) { Pop-Location; Die ") < after.index("$TorchPin = ")


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


# --- anonymous install events, in step with install.sh -------------------------------------

def test_the_windows_installer_says_the_same_sentence_as_the_studio():
    """One promise, one wording, every surface: README, About, and both installers."""
    ps1 = PS1.read_text(encoding="utf-8")
    assert "We collect anonymous data to make AssetFurnace better: which steps you run" in ps1
    assert "Never your pictures, prompts, files or anything private" in ps1
    assert "Turn it off any time on the About page" in ps1
    assert "https://assetfurnace.com/privacy#usage-counts" in ps1


def test_it_posts_the_same_events_as_the_shell_installer():
    """One endpoint, one ceiling, one closed list of reasons, from either installer."""
    ps1 = PS1.read_text(encoding="utf-8")
    sh = (REPO / "install.sh").read_text(encoding="utf-8")
    closed = {"intel_mac", "unsupported_os", "no_git", "old_nvidia", "old_amd", "no_gpu",
              "user_said_no", "other"}
    for text in (ps1, sh):
        assert "assetfurnace.com/api/install-event" in text
        for event in ('"started"', "stopped", "finished"):
            assert event in text
        assert "I3D_NO_USAGE" in text and "DO_NOT_TRACK" in text
        used = set(re.findall(r'\b(?:intel_mac|unsupported_os|no_git|old_nvidia|old_amd|no_gpu|user_said_no)\b', text))
        assert used and used <= closed
    # 3 second ceiling, in each one's own spelling.
    assert '"-m", "3"' in ps1 and "-m 3" in sh
    # The reasons each machine can actually give: a Windows box is never an Intel Mac.
    assert {"intel_mac", "unsupported_os", "no_git", "old_nvidia", "user_said_no"} <= (
        set(re.findall(r'\b(intel_mac|unsupported_os|no_git|old_nvidia|user_said_no)\b', sh)))
    assert {"unsupported_os", "no_git", "old_nvidia", "user_said_no"} <= (
        set(re.findall(r'\b(unsupported_os|no_git|old_nvidia|user_said_no)\b', ps1)))

def test_an_old_nvidia_card_is_refused_before_the_download():
    """The same wall Rig hit, met before two gigabytes of packages are fetched."""
    ps1 = PS1.read_text(encoding="utf-8")
    assert "--query-gpu=compute_cap" in ps1
    assert "-lt 75" in ps1
    assert "older than AssetFurnace supports" in ps1
    assert "RTX 20-series" in ps1
    before = ps1[ps1.index("--query-gpu=compute_cap"):]
    assert before.index("old_nvidia") < before.index("uv venv")


def test_an_event_is_never_able_to_fail_the_install():
    """A missing curl, a blocked port, a dead site: the install carries on either way."""
    ps1 = PS1.read_text(encoding="utf-8")
    body = ps1[ps1.index("function Send-Event"):ps1.index("# 1. This machine")]
    assert "Get-Command curl.exe -ErrorAction SilentlyContinue" in body
    assert "if (-not $curl) { return }" in body
    assert "catch { }" in body
