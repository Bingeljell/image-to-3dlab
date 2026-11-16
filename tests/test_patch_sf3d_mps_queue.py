"""scripts/patch_sf3d_mps_queue.py: SF3D's Metal baker fetches its command buffer on the MPS queue.

At the pinned commit both baker kernels call `torch::mps::get_command_buffer()` before
`dispatch_sync` onto the MPS stream's queue. PyTorch 2.15 asserts that fetch happens on that
queue (`dispatch_assert_queue`), so every SF3D run on a Mac died with SIGTRAP while baking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_sf3d_mps_queue as mps

# One kernel's block as it stands at the pinned commit; the file has two of them.
BLOCK = """\
            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
                // Start a compute pass.
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
                torch::mps::commit();
            });
"""
SOURCE = "torch::Tensor rasterize_gpu() {\n" + BLOCK + "}\n\ntorch::Tensor interpolate_gpu() {\n" + BLOCK + "}\n"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    path = tmp_path / mps.BAKER_FILE
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    return tmp_path


def _blocks(text: str) -> list[str]:
    return [part for part in text.split("torch::Tensor ") if "dispatch_sync" in part]


def test_both_kernels_fetch_the_command_buffer_inside_the_mps_queue(checkout):
    assert mps.patch(checkout) == 1
    text = (checkout / mps.BAKER_FILE).read_text()
    blocks = _blocks(text)
    assert len(blocks) == 2
    for block in blocks:
        assert block.count("torch::mps::get_command_buffer()") == 1
        assert block.index("dispatch_sync(serialQueue") < block.index("torch::mps::get_command_buffer()")
        # still used where it was: inside the block, before the encoder
        assert block.index("get_command_buffer()") < block.index("[commandBuffer computeCommandEncoder]")


def test_running_it_twice_changes_nothing(checkout):
    mps.patch(checkout)
    once = (checkout / mps.BAKER_FILE).read_text()
    assert mps.patch(checkout) == 0
    assert (checkout / mps.BAKER_FILE).read_text() == once
    assert mps.is_patched(checkout)


def test_an_unpatched_checkout_is_reported_as_such(checkout):
    assert not mps.is_patched(checkout)


def test_a_changed_upstream_fails_loudly_instead_of_half_patching(checkout):
    (checkout / mps.BAKER_FILE).write_text(SOURCE.replace("get_command_buffer();", "getCommandBuffer();"))
    with pytest.raises(RuntimeError, match="anchor"):
        mps.patch(checkout)


def test_one_kernel_patched_by_hand_is_refused(checkout):
    """The anchor must appear exactly as often as upstream has it, or the file is not the one we know."""
    half = SOURCE.replace(BLOCK, "            // edited by hand\n", 1)
    (checkout / mps.BAKER_FILE).write_text(half)
    with pytest.raises(RuntimeError, match="expected 2"):
        mps.patch(checkout)
