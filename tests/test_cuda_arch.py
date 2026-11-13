"""Tests for the "can this torch run on this card" decision.

The bug this answers (2026-11-13): a Quadro P5000, compute capability 6.1, and our torch
carrying sm_75 and newer. Every kernel died with `CUDA error: no kernel image is available
for execution on the device` after the rig had loaded its model. The comparison is cheap;
the sentence is what a person gets instead of the traceback.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from image_to_3dlab import cuda_arch as ca

# What our NVIDIA installs report, and the card that crashed.
OURS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
QUADRO = (6, 1)
RTX_20 = (7, 5)
RTX_50 = (12, 0)


@pytest.mark.parametrize("entry, number", [
    ("sm_75", 75), ("sm_80", 80), ("sm_90a", 90), ("sm_120", 120), ("sm_75+PTX", 75),
    ("compute_75", None), ("", None), ("sm_", None),
])
def test_an_architecture_entry_becomes_its_number(entry, number):
    assert ca.sm_number(entry) == number


@pytest.mark.parametrize("archs, capability, expected", [
    (OURS, RTX_20, True),                 # sm_75 is in the list
    (OURS, QUADRO, False),                # sm_61 is not, and nothing below sm_75 is carried
    (OURS, RTX_50, True),                 # sm_120 is in the list
    (["sm_75+PTX"], RTX_50, True),        # PTX for 75 compiles up to 120 at load time
    (["sm_75+PTX"], QUADRO, False),       # ... but never down to an older card
    (["sm_90a"], (9, 0), True),           # the accelerated variant still runs sm_90 code
    ([], RTX_20, False),
])
def test_torch_supports_asks_whether_its_code_covers_the_card(archs, capability, expected):
    assert ca.torch_supports(archs, capability) is expected


def test_an_older_card_gets_the_plain_sentence():
    assert ca.problem(OURS, QUADRO, "NVIDIA Quadro P5000") == (
        "Your graphics card (NVIDIA Quadro P5000) is older than Rig supports: it needs an "
        "NVIDIA RTX 20-series or newer.")


def test_the_sentence_still_reads_without_a_card_name():
    assert ca.problem(OURS, QUADRO) == (
        "Your graphics card is older than Rig supports: it needs an NVIDIA RTX 20-series or newer.")


def test_a_supported_card_is_told_nothing():
    assert ca.problem(OURS, RTX_20, "NVIDIA GeForce RTX 2080") is None
    assert ca.problem(["sm_75+PTX"], RTX_50, "NVIDIA GeForce RTX 5090") is None


def test_a_card_newer_than_the_install_is_not_called_old():
    """An RTX 50 on a torch built up to sm_90 needs a newer Rig, not a newer card."""
    message = ca.problem(["sm_75", "sm_80", "sm_90"], RTX_50, "NVIDIA GeForce RTX 5090")
    assert "older than your graphics card" in message
    assert "Set up the auto-rig again" in message


def test_a_rocm_or_metal_build_is_not_judged_by_cuda_architectures():
    """SkinTokens runs on ROCm and Metal too; gfx1030 is not an sm_ number and never will be."""
    assert ca.problem(["gfx1030", "gfx1100"], (8, 6), "AMD Radeon RX 7900 XTX") is None


def test_missing_facts_are_never_a_refusal():
    """A machine that cannot tell us is left to fail or work on its own, not refused on a guess."""
    assert ca.problem([], QUADRO) is None
    assert ca.problem(OURS, None) is None


@pytest.mark.parametrize("capability, blocked", [
    ("61", True), (61, True), ("60", True), ("75", False), ("86", False), ("120", False),
    (None, False), ("", False), ("lots", False),
])
def test_setup_refuses_before_the_download_only_below_the_floor(capability, blocked):
    """The driver knows the card long before a venv exists, so setup can stop first."""
    assert (ca.setup_problem(capability, "NVIDIA Quadro P5000") is not None) is blocked


def test_setup_names_the_card_it_refuses():
    assert "Quadro P5000" in ca.setup_problem("61", "NVIDIA Quadro P5000")


def _run(stdout: str = "", code: int = 0, raises: Exception | None = None):
    def runner(command, **kwargs):
        if raises:
            raise raises
        return types.SimpleNamespace(returncode=code, stdout=stdout, stderr="")

    return runner


def test_the_probe_reads_the_venvs_own_torch(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    facts = {"archs": OURS, "capability": [6, 1], "device": "NVIDIA Quadro P5000"}
    seen = []

    def runner(command, **kwargs):
        seen.append(command)
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(facts) + "\n", stderr="")

    assert ca.probe(python, runner) == facts
    assert seen[0][:1] == [str(python)] and seen[0][1] == "-c"
    assert "torch.cuda.get_arch_list" in seen[0][2]


def test_the_probe_speaks_only_when_it_can_be_believed(tmp_path):
    """No venv, a torch that will not import, a hanging import: all "cannot tell"."""
    assert ca.probe(tmp_path / "missing") is None
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    assert ca.probe(python, _run("not json at all\n")) is None
    assert ca.probe(python, _run("", code=1)) is None
    assert ca.probe(python, _run(raises=OSError("no python"))) is None
    assert ca.probe(python, _run(raises=TimeoutError("too slow"))) is None


def test_the_card_check_says_the_sentence_for_the_real_card(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    facts = {"archs": OURS, "capability": [6, 1], "device": "NVIDIA Quadro P5000"}
    message = ca.card_problem(python, run=_run(json.dumps(facts) + "\n"))
    assert message.startswith("Your graphics card (NVIDIA Quadro P5000) is older")


def test_the_card_check_passes_a_torch_that_covers_the_card(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    facts = {"archs": OURS, "capability": [8, 6], "device": "NVIDIA GeForce RTX 3090"}
    assert ca.card_problem(python, run=_run(json.dumps(facts) + "\n")) is None


def test_a_probe_that_could_not_ask_stops_nothing(tmp_path):
    """None from the probe means "cannot tell", so the rig is not blocked on a guess."""
    assert ca.card_problem(Path("/nowhere/python"), run=_run()) is None
