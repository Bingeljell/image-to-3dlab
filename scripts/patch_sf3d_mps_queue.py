#!/usr/bin/env python3
"""Let Stable Fast 3D's Metal texture baker run under current PyTorch on a Mac.

Both kernels in `texture_baker/csrc/baker_kernel.mm` take the MPS stream's command buffer
with `torch::mps::get_command_buffer()` and only then `dispatch_sync` onto that stream's
queue to encode. PyTorch 2.15 asserts the command buffer is fetched on the stream's own
queue (`MPSHooks::getCommandBuffer` -> `dispatch_assert_queue`), so on an Apple Silicon Mac
every SF3D run died with SIGTRAP ("Trace/BPT trap: 5", exit 133) the moment it began to
bake, with no Python traceback to say why. SF3D's upstream has not moved since 2025-01.

The fix moves the fetch one block down, inside `dispatch_sync`. The buffer is only used
inside that block, so nothing else changes. The baker must be rebuilt afterwards;
`bootstrap_sf3d.install_code` does that when this changes the source.

Safe to run twice: a patched file is recognised and left alone, and the anchor is asserted
(twice, once per kernel) so an upstream change fails loudly rather than half-applying.

    python scripts/patch_sf3d_mps_queue.py
    python scripts/patch_sf3d_mps_queue.py --root <stable-fast-3d checkout>
    python scripts/patch_sf3d_mps_queue.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "stable-fast-3d"

BAKER_FILE = "texture_baker/texture_baker/csrc/baker_kernel.mm"
# Once per kernel (rasterize and interpolate), at the pinned commit.
EXPECTED = 2
NEEDLE = """\
            // Get a reference to the command buffer for the MPS stream.
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){
"""
PROOF = "image-to-3dlab: fetch the command buffer on the MPS stream's own queue"
REPLACEMENT = f"""\
            // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
            dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

            dispatch_sync(serialQueue, ^(){{
                // {PROOF}.
                // PyTorch asserts it there (dispatch_assert_queue); fetched outside, the run died with SIGTRAP.
                id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
                TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");
"""


def patch_text(source: str, label: str = BAKER_FILE) -> tuple[str, bool]:
    """Return the patched text and whether it changed."""
    if PROOF in source:
        return source, False
    found = source.count(NEEDLE)
    if found == 0:
        raise RuntimeError(f"{label}: the anchor for this patch is gone; upstream changed, "
                           "re-check the patch before running it")
    if found != EXPECTED:
        raise RuntimeError(f"{label}: expected {EXPECTED} occurrences of this patch's anchor, "
                           f"found {found}; re-check the patch before running it")
    return source.replace(NEEDLE, REPLACEMENT), True


def is_patched(root: Path = DEFAULT_ROOT) -> bool:
    try:
        return PROOF in (root / BAKER_FILE).read_text()
    except OSError:
        return False


def patch(root: Path = DEFAULT_ROOT) -> int:
    """Apply the patch. Returns how many files changed (0 or 1), and is safe to run twice."""
    path = root / BAKER_FILE
    if not path.is_file():
        raise SystemExit(f"{path} is missing; this is not a stable-fast-3d checkout.")
    patched, changed = patch_text(path.read_text(), str(path))
    if changed:
        path.write_text(patched)
    return int(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the stable-fast-3d checkout (default: vendor/stable-fast-3d)")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether the patch is in place.")
    args = parser.parse_args(argv)
    if args.check:
        done = is_patched(args.root)
        print("patched" if done else "not patched")
        return 0 if done else 1
    changed = patch(args.root)
    print("patched the Metal baker" if changed else "already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
