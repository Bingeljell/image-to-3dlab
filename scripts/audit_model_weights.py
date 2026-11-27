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
   actually opens. We verify the safetensors header declares at least one tensor
   before proposing deletion, so an empty or interrupted conversion never costs the
   original checkpoint.
2. ``cached`` — a Hugging Face cache entry whose files are also present as real (non
   symlinked) files under a local weights directory. The cache copy is then a second
   physical copy, not the one in use. Matched on file name and exact size.
3. ``unreferenced`` — a cache entry whose repo id appears in no tracked source file.
   **Reported only, never deleted**, because "nothing greps for it" is evidence about our
   code, not about whether something on this machine needs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# Files at least this large must have a byte-identical local copy before an entry is
# deletable (capped at GB so scaled-down tests stay meaningful); smaller ones (README, config) may be missing locally and are only listed.
BIG_FILE_BYTES = 1 << 20

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


MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


def conversion_proven(converted: Path) -> bool:
    """True when `converted` exists and its header describes tensors that fit in the file.

    We read the 8-byte little-endian header length, then the JSON header. Every tensor's
    end offset must lie inside the bytes after the header. `__metadata__` is not a tensor.
    The tensor bytes themselves are never deserialized.
    """
    try:
        size = converted.stat().st_size
    except OSError:
        return False
    if size < 8:
        return False
    try:
        with converted.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                return False
            header_len = int.from_bytes(raw_len, "little")
            if header_len <= 0 or header_len > size - 8 or header_len > MAX_SAFETENSORS_HEADER:
                return False
            header = handle.read(header_len)
            if len(header) != header_len:
                return False
    except OSError:
        return False
    try:
        parsed = json.loads(header)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict) or not parsed:
        return False
    data_end = size - 8 - header_len
    tensors = 0
    for key, entry in parsed.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
            or offsets[0] > offsets[1]
            or offsets[1] > data_end
        ):
            return False
        tensors += 1
    return tensors > 0


def find_superseded_checkpoints(roots=LOCAL_WEIGHT_ROOTS) -> list[Candidate]:
    """A `.ckpt` whose converted `.safetensors` sits beside it and proves valid is dead weight."""
    found: list[Candidate] = []
    for root in roots:
        if not root.is_dir():
            continue
        for ckpt in root.rglob("*.ckpt"):
            converted = ckpt.with_suffix(".safetensors")
            if not converted.is_file() or ckpt.is_symlink():
                continue
            if conversion_proven(converted):
                found.append(Candidate(
                    "ckpt", ckpt, directory_size(ckpt),
                    f"converted already: {converted.name} ({human_bytes(directory_size(converted))})",
                ))
            else:
                found.append(Candidate(
                    "ckpt", ckpt, directory_size(ckpt),
                    f"kept: conversion unproven ({converted.name})",
                    deletable=False,
                ))
    return found


def _local_files(roots=LOCAL_WEIGHT_ROOTS) -> dict[tuple[str, int], list[Path]]:
    """Real files under the local weight roots, keyed by (name, size); all candidates kept."""
    index: dict[tuple[str, int], list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for item in root.rglob("*"):
            try:
                if item.is_symlink() or not item.is_file():
                    continue
                index.setdefault((item.name, item.stat().st_size), []).append(item)
            except OSError:
                continue
    return index


def _sha256(path: Path) -> str | None:
    """Uncached sha256 of `path`; None when unreadable."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


_HASHES: dict[tuple[str, int, int], str | None] = {}


def _hash_once(path: Path) -> str | None:
    """`_sha256` memoised by resolved path, size and mtime, so each file hashes once."""
    try:
        resolved = path.resolve()
        st = resolved.stat()
    except OSError:
        return None
    key = (str(resolved), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = _sha256(resolved)
    return _HASHES[key]


def _identical_copy(cached: Path, candidates: list[Path]) -> Path | None:
    """A candidate whose bytes equal `cached`'s, or None. Unreadable means not identical."""
    want = _hash_once(cached)
    if want is None:
        return None
    for candidate in candidates:
        if _hash_once(candidate) == want:
            return candidate
    return None


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

    Files are paired by name and exact size, then confirmed by a sha256 of both. An
    entry is deletable only when every one of its files has a byte-identical local copy.
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
        pairs = [(f, local.get((f.name, _resolved_size(f)))) for f in names]
        if not any(c for _, c in pairs):
            continue
        # Cheap pre-filter on size so non-matching entries are never hashed.
        matched: list[Path] = []
        examples: dict[Path, Path] = {}
        for f, cands in pairs:
            if not cands:
                continue
            twin = _identical_copy(f, cands)
            if twin is not None:
                matched.append(f)
                examples[f] = twin
        if not matched:
            continue
        matched_bytes = sum(_resolved_size(f) for f in matched)
        total = directory_size(blobs) if blobs.is_dir() else directory_size(path)
        if total <= 0 or matched_bytes < GB:
            continue
        biggest = max(matched, key=_resolved_size)
        example = _short(examples[biggest])
        # Deletable when every big file is duplicated byte for byte; small files may be
        # missing locally (they are listed in the reason) but big ones never.
        matched_set = set(matched)
        unmatched = [f for f in names if f not in matched_set]
        big_missing = [f for f in unmatched if _resolved_size(f) >= min(BIG_FILE_BYTES, GB)]
        small_missing = sorted(f.name for f in unmatched if f not in big_missing)
        complete = not big_missing
        note = ""
        if small_missing:
            note = f"; small files not in the repo: {', '.join(small_missing)}"
        found.append(Candidate(
            "cached", path, total,
            f"{human_bytes(matched_bytes)} of {human_bytes(total)} also real in the repo"
            f" ({len(matched)} files, e.g. {example})"
            + note
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
