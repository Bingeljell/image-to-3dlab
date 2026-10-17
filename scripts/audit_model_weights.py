#!/usr/bin/env python3
"""Find model weights stored twice, and optionally reclaim the duplicates.

    python scripts/audit_model_weights.py               # dry run, prints a report
    python scripts/audit_model_weights.py --apply       # actually delete
    python scripts/audit_model_weights.py --rule ckpt   # one rule only

**Why this exists.** An audit on 2026-09-21 found a 104 GB Hugging Face cache holding
roughly 65 GB that nothing needed: a rejected PyTorch port still cached in full, a model
kept in two formats because the converter never deleted its input, and two mirrors of the
same image encoder. None of it was visible without walking the disk, and none of it would
have been noticed by eye.

**It deletes nothing by default.** `--apply` is required, every candidate is printed with
its size first, and the rules below only ever propose a path under a root passed in.

Three rules, in descending order of how sure they are:

1. ``ckpt`` — a ``.ckpt`` sitting beside a ``.safetensors`` of the same stem. The
   checkpoint was the download; the safetensors is the converted result the loader
   actually opens. Provable from the filesystem, so safe to delete.
2. ``cached`` — a Hugging Face cache entry whose files are also present as real (non
   symlinked) files under a local weights directory. The cache copy is then a second
   physical copy, not the one in use. Matched on file name and exact size.
3. ``unreferenced`` — a cache entry whose repo id appears in no tracked source file.
   **Reported only, never deleted**, because "nothing greps for it" is evidence about our
   code, not about whether something on this machine needs it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

from backend_catalog import GB, HF_HUB_DIR, human_bytes

# Where real-file copies of weights live in this repo. A cache entry duplicated by one of
# these is the `cached` rule's target.
LOCAL_WEIGHT_ROOTS = (
    REPO / "hunyuan_mlx" / "shape" / "weights",
    REPO / "hunyuan_mlx" / "paint" / "weights",
    REPO / "vendor" / "pixal3d-cpp" / "models",
)

# Searched for a cache entry's repo id. Deliberately runtime code only: `tests/` asserts
# string literals and `docs/` discusses candidates we evaluated and rejected, so grepping
# either reports a 43 GB dead cache as "referenced".
SOURCE_PATHS = ("scripts", "viewer", "hunyuan_mlx", "image_to_3dlab", "workflows",
                "pipeline.py")

RULES = ("ckpt", "cached", "unreferenced")


@dataclass
class Candidate:
    rule: str
    path: Path
    bytes_: int
    reason: str
    deletable: bool = True


def directory_size(path: Path) -> int:
    """Bytes under `path`, counting each file once.

    Symlinks are skipped: the Hugging Face cache keeps one copy in `blobs/` and links to
    it from `snapshots/`, so following both doubles every figure.
    """
    if path.is_file():
        return 0 if path.is_symlink() else path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_symlink() or not item.is_file():
                continue
            total += item.stat().st_size
        except OSError:
            continue
    return total


def find_superseded_checkpoints(roots=LOCAL_WEIGHT_ROOTS) -> list[Candidate]:
    """A `.ckpt` whose converted `.safetensors` sits beside it is dead weight."""
    found: list[Candidate] = []
    for root in roots:
        if not root.is_dir():
            continue
        for ckpt in root.rglob("*.ckpt"):
            converted = ckpt.with_suffix(".safetensors")
            if converted.is_file() and not ckpt.is_symlink():
                found.append(Candidate(
                    "ckpt", ckpt, directory_size(ckpt),
                    f"converted already: {converted.name} ({human_bytes(directory_size(converted))})",
                ))
    return found


def _local_files(roots=LOCAL_WEIGHT_ROOTS) -> dict[tuple[str, int], Path]:
    """Real files under the local weight roots, keyed by (name, size)."""
    index: dict[tuple[str, int], Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for item in root.rglob("*"):
            try:
                if item.is_symlink() or not item.is_file():
                    continue
                index[(item.name, item.stat().st_size)] = item
            except OSError:
                continue
    return index


def cache_repos(hub: Path = HF_HUB_DIR) -> list[tuple[str, Path]]:
    """Every `models--org--name` directory in the hub cache, with its repo id."""
    if not hub.is_dir():
        return []
    out = []
    for path in sorted(hub.glob("models--*")):
        if path.is_dir():
            out.append((path.name.removeprefix("models--").replace("--", "/", 1), path))
    return out


def find_cached_duplicates(hub: Path = HF_HUB_DIR, roots=LOCAL_WEIGHT_ROOTS) -> list[Candidate]:
    """Cache entries whose payload also exists as real files in the repo.

    Matched on name and exact byte size rather than a hash: these are multi-gigabyte
    files, hashing them all would take longer than the download did, and a collision on
    both name and exact size is not a realistic accident here.
    """
    local = _local_files(roots)
    if not local:
        return []
    found: list[Candidate] = []
    for repo_id, path in cache_repos(hub):
        blobs = path / "blobs"
        snapshots = path / "snapshots"
        if not snapshots.is_dir():
            continue
        names = [f for f in snapshots.rglob("*") if f.is_symlink() or f.is_file()]
        matched = [f for f in names if (f.name, _resolved_size(f)) in local]
        if not matched:
            continue
        # By bytes, not by file count. A snapshot is mostly small source files wrapped
        # around a couple of huge weights, so "half the files" never fires on the very
        # entries worth reclaiming.
        matched_bytes = sum(_resolved_size(f) for f in matched)
        total = directory_size(blobs) if blobs.is_dir() else directory_size(path)
        # Surface anything meaningfully duplicated, but see `complete` below before
        # offering to delete. A conversion changes the file name (`.bin` becomes
        # `.safetensors`), so name matching systematically under-counts and a strict
        # threshold hides the very entries worth looking at.
        if total <= 0 or matched_bytes < GB:
            continue
        biggest = max(matched, key=_resolved_size)
        example = _short(local[(biggest.name, _resolved_size(biggest))])
        # Only offer to delete when essentially everything is duplicated. A partial match
        # means the cache still holds bytes that exist nowhere else.
        complete = matched_bytes >= total * 0.9
        found.append(Candidate(
            "cached", path, total,
            f"{human_bytes(matched_bytes)} of {human_bytes(total)} also real in the repo"
            f" ({len(matched)} files, e.g. {example})"
            + ("" if complete else "; PARTIAL, review before deleting"),
            deletable=complete,
        ))
    return found


def find_unreferenced(hub: Path = HF_HUB_DIR, repo: Path = REPO) -> list[Candidate]:
    """Cache entries whose repo id appears in no tracked source file. Report only."""
    found: list[Candidate] = []
    for repo_id, path in cache_repos(hub):
        if _mentioned(repo_id, repo):
            continue
        found.append(Candidate(
            "unreferenced", path, directory_size(path),
            f"'{repo_id}' appears in no tracked source file",
            deletable=False,
        ))
    return found


def _short(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise. Display only."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _resolved_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _mentioned(needle: str, repo: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "grep", "-lF", needle, "--", *SOURCE_PATHS],
            cwd=repo, capture_output=True, text=True, check=False,
        )
    except OSError:
        return True  # No git: assume referenced rather than propose a deletion.
    return bool(result.stdout.strip())


def collect(rules: tuple[str, ...], hub: Path = HF_HUB_DIR) -> list[Candidate]:
    found: list[Candidate] = []
    if "ckpt" in rules:
        found += find_superseded_checkpoints()
    if "cached" in rules:
        found += find_cached_duplicates(hub)
    if "unreferenced" in rules:
        found += find_unreferenced(hub)
    return sorted(found, key=lambda c: -c.bytes_)


def report(found: list[Candidate], apply: bool) -> int:
    if not found:
        print("nothing to reclaim")
        return 0
    reclaimable = sum(c.bytes_ for c in found if c.deletable)
    review = sum(c.bytes_ for c in found if not c.deletable)

    for rule in RULES:
        rows = [c for c in found if c.rule == rule]
        if not rows:
            continue
        print(f"\n[{rule}]")
        for c in rows:
            mark = " " if c.deletable else "?"
            try:
                shown = c.path.relative_to(Path.home())
                shown = Path("~") / shown
            except ValueError:
                shown = c.path
            print(f" {mark} {human_bytes(c.bytes_):>9}  {shown}")
            print(f"              {c.reason}")

    print(f"\nreclaimable: {human_bytes(reclaimable)}")
    if review:
        print(f"needs review (not deleted): {human_bytes(review)}")
    print("note: everything under the Hugging Face cache is re-downloadable, so the cost "
          "of deleting one is a re-fetch, not a loss. Files under the repo are not.")

    if not apply:
        print("\ndry run. re-run with --apply to delete the reclaimable entries.")
        return 0

    freed = 0
    for c in found:
        if not c.deletable:
            continue
        try:
            if c.path.is_dir():
                shutil.rmtree(c.path)
            else:
                c.path.unlink()
            freed += c.bytes_
            print(f"removed {c.path}")
        except OSError as exc:
            print(f"could not remove {c.path}: {exc}")
    print(f"\nfreed {human_bytes(freed)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="delete the reclaimable entries; without it this only reports")
    parser.add_argument("--rule", action="append", choices=RULES, dest="rules",
                        help="run one rule only; repeatable (default: all)")
    parser.add_argument("--hf-cache", type=Path, default=HF_HUB_DIR,
                        help="Hugging Face hub cache to audit")
    args = parser.parse_args()
    rules = tuple(args.rules) if args.rules else RULES
    return report(collect(rules, args.hf_cache), args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
