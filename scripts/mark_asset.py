#!/usr/bin/env python3
"""Record a human verdict on a generated asset, so winners stay findable.

    python scripts/mark_asset.py output/hero/fox.glb --verdict good \\
        --note "flowers survive, eyes amber" --tags flowers,eyes --culled

    python scripts/mark_asset.py --list --verdict good        # every winner
    python scripts/mark_asset.py --for assets_to_test/fox.png # every verdict per subject

**Why this exists.** On 2026-08-13 a render named `foxR_sweep.png` showed a fox that looked
right, and nothing in the repo could say which GLB produced it. Twenty minutes of
archaeology across forty files did not settle it. The provenance sidecar records source
art, settings and output for every run — it has never recorded whether the result was any
good, and that is the one judgement only a human can supply.

Three design choices, each earned:

* **Keyed by content hash, not path.** Half the interesting assets are derived files with
  no sidecar at all (`fox_repaired.glb`, `flicker_painted3.glb`), and they get renamed,
  copied and moved between folders.
* **Append-only.** A verdict is an observation made at a moment. Later disagreement is a
  new entry, not an edit — the same reason this repo keeps superseded docs.
* **The measurements are snapshotted alongside the verdict.** "This one was good" is a
  sticky note. "This one was good, and it had 45 non-manifold edges, consistent winding
  and a metallicRoughness map" is a data point you can learn from. This is the seed of the
  fine-tuning dataset described in docs/training-trellis.md, which is worth starting now
  and impossible to reconstruct later.

`--culled` records *how it was judged*. A verdict from a double-sided preview is much
weaker evidence than one from a backface-culled view, because a double-sided render cannot
distinguish a solid mesh from a hollow one — the failure that hid inside-out meshes here
for weeks. Unset means unknown, which is honest rather than convenient.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for glb_forensics

REGISTER = Path(__file__).resolve().parents[1] / "output" / "verdicts.jsonl"
VERDICTS = ("good", "bad", "mixed")


def load_register(path: Path) -> list[dict]:
    """Every verdict ever recorded, oldest first. Missing file means none yet."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def append_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def find_sidecar(asset: Path) -> Path | None:
    """The `.provenance.json` beside an asset, if this was a pipeline run."""
    candidate = asset.with_suffix(".provenance.json")
    return candidate if candidate.exists() else None


def measure(asset: Path) -> dict:
    """Forensic snapshot at judgement time — what was true when a human approved it.

    Failures are recorded rather than raised: a verdict on an asset we cannot parse is
    still worth keeping, and losing the human judgement to a tooling error would be the
    worst possible outcome here.
    """
    try:
        import glb_forensics

        report = glb_forensics.inspect(asset)
        material = report["materials"][0] if report["materials"] else {}
        geometry = next(iter(report["geometry"].values()), {})
        return {
            "faces": geometry.get("faces"),
            "vertices": geometry.get("vertices"),
            "boundary_edges": geometry.get("boundary_edges"),
            "nonmanifold_edges": geometry.get("nonmanifold_edges"),
            "winding_consistent": geometry.get("winding_consistent"),
            "volume": geometry.get("volume"),
            "face_area_ratio": geometry.get("face_area_ratio"),
            "textures": material.get("textures"),
            "doubleSided": material.get("doubleSided"),
            "metallicFactor": material.get("metallicFactor"),
        }
    except Exception as exc:  # noqa: BLE001 - a verdict must survive a broken measurement
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_entry(
    asset: Path,
    verdict: str,
    note: str,
    *,
    tags: list[str] | None = None,
    culled: bool | None = None,
    source: str | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Assemble one register entry. Pure apart from hashing and measuring the file."""
    from image_to_3dlab.provenance import sha256_file

    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if not note.strip():
        raise ValueError("a note is required - 'good' without a reason is not a data point")

    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    sidecar = find_sidecar(asset)
    return {
        "recorded_at": stamp,
        "asset": str(asset),
        "asset_sha256": sha256_file(asset),
        "verdict": verdict,
        "note": note.strip(),
        "tags": sorted(tags or []),
        "judged_backface_culled": culled,
        "source_image": source,
        "provenance": str(sidecar) if sidecar else None,
        "measured": measure(asset),
    }


def select(entries: list[dict], verdict: str | None = None, source: str | None = None) -> list[dict]:
    """Filter the register. Both filters are exact matches."""
    chosen = entries
    if verdict:
        chosen = [e for e in chosen if e.get("verdict") == verdict]
    if source:
        chosen = [e for e in chosen if e.get("source_image") == source]
    return chosen


def _format(entry: dict) -> str:
    measured = entry.get("measured", {})
    culled = entry.get("judged_backface_culled")
    flag = {True: "culled", False: "DOUBLE-SIDED", None: "culling unknown"}[culled]
    head = (f"{entry['recorded_at'][:16]}  {entry['verdict'].upper():5s}  "
            f"{Path(entry['asset']).name}")
    bits = []
    if measured.get("faces"):
        bits.append(f"{measured['faces']:,} faces")
    if measured.get("nonmanifold_edges") is not None:
        bits.append(f"{measured['nonmanifold_edges']:,} non-manifold")
    if measured.get("winding_consistent") is not None:
        bits.append(f"winding={measured['winding_consistent']}")
    if measured.get("textures"):
        bits.append("+".join(measured["textures"]))
    return (f"{head}\n      [{flag}] {entry['note']}\n"
            f"      {' | '.join(bits)}\n      sha {entry['asset_sha256'][:12]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("asset", nargs="?", type=Path)
    parser.add_argument("--verdict", choices=VERDICTS)
    parser.add_argument("--note", default="", help="why. required when recording")
    parser.add_argument("--tags", default="", help="comma separated, e.g. flowers,eyes")
    parser.add_argument("--source", help="the source image this came from")
    parser.add_argument("--culled", dest="culled", action="store_true",
                        help="judged with backface culling ON (the honest test)")
    parser.add_argument("--double-sided", dest="culled", action="store_false",
                        help="judged double-sided; weaker evidence, recorded as such")
    parser.set_defaults(culled=None)
    parser.add_argument("--list", action="store_true", help="show recorded verdicts")
    parser.add_argument("--for", dest="for_source", help="filter --list by source image")
    parser.add_argument("--register", type=Path, default=REGISTER)
    args = parser.parse_args()

    if args.list or (args.asset is None and args.for_source):
        entries = select(load_register(args.register), args.verdict, args.for_source)
        if not entries:
            print("no verdicts recorded yet")
            return 0
        for entry in entries:
            print(_format(entry))
            print()
        print(f"{len(entries)} verdict(s) in {args.register}")
        return 0

    if args.asset is None or args.verdict is None:
        parser.error("recording a verdict needs an asset and --verdict (or use --list)")
    if not args.asset.exists():
        parser.error(f"{args.asset} does not exist")

    entry = build_entry(
        args.asset,
        args.verdict,
        args.note,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        culled=args.culled,
        source=args.source,
    )
    append_entry(args.register, entry)
    print(_format(entry))
    if entry["judged_backface_culled"] is None:
        print("\n  note: culling not stated. Pass --culled or --double-sided next time;\n"
              "  a double-sided verdict cannot tell a solid mesh from a hollow one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
