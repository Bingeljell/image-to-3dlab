#!/usr/bin/env python3
"""The layout maths of the shape-guide kit: a detailed vehicle from a few numbers (#259).

`blender_shape_guide.py` builds the vehicle in Blender; everything here is plain maths, tested
without Blender (`tests/test_shape_guide.py`). The kit started from a hand-built 12 m bus whose
every window was placed by hand; here the same layout comes from rules, so a new vehicle needs
only its numbers.

    python scripts/shape_guide.py out/ --preset midibus_9m --name midibus --tex 4096
    python scripts/shape_guide.py out/ --spec delivery_van.json        # a preset plus your own numbers

A spec file names a preset and sets numbers on top of it, for example
`{"preset": "van_long", "name": "delivery", "set": {"rear_axle": -2.4}}`. Writes
`<name>_parts.glb` (body, wheels and doors as parts on their pivots), `<name>_joined.glb` (the same
as one mesh: what the multi-view rig renders and Pixel Match paints) and the atlas.
Needs Blender 4.2 or newer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# Each family is one Blender builder; a preset is a set of numbers for one of them.
FAMILIES = {"bus": "blender_shape_guide.py", "van": "blender_shape_guide_van.py"}
PRESETS = {
    "bus_12m": "bus", "bus_12m_rule": "bus", "midibus_9m": "bus", "metro_car": "bus",
    "ambulance": "van", "van_long": "van", "box_van": "van",
}

# A pane shorter than this is not worth a window: the stretch stays closed panelling.
MIN_PANE = 0.40


def spread_windows(start: float, end: float, pitch: float, pillar: float,
                   keep_out: list[tuple[float, float]] | tuple = ()) -> list[tuple[float, float]]:
    """Side windows along a vehicle's length, as (rear, front) pairs from the front backwards.

    The band runs from `start` (rear) to `end` (front). Each stretch of it that no door
    (`keep_out`, (rear, front) pairs) covers, less a `pillar` on each side of a door, is split
    into equal panes about `pitch` long including one pillar, with a `pillar` between panes.
    A stretch too short for a pane of `MIN_PANE` stays closed.
    """
    if pitch <= pillar:
        raise ValueError(f"the pitch ({pitch} m) must be longer than the pillar ({pillar} m)")
    stretches = [(start, end)]
    for d0, d1 in sorted(keep_out):
        cut = []
        for s0, s1 in stretches:
            if d1 + pillar <= s0 or d0 - pillar >= s1:
                cut.append((s0, s1))
                continue
            if d0 - pillar > s0:
                cut.append((s0, d0 - pillar))
            if d1 + pillar < s1:
                cut.append((d1 + pillar, s1))
        stretches = cut
    panes = []
    for s0, s1 in sorted(stretches, reverse=True):
        length = s1 - s0
        count = max(1, round((length + pillar) / pitch))
        pane = (length - (count - 1) * pillar) / count
        if pane < MIN_PANE:
            continue
        for k in range(count):
            front = s1 - k * (pane + pillar)
            panes.append((front - pane, front))
    return panes


def seat_rows(start: float, end: float, pitch: float, depth: float,
              keep_out: list[tuple[float, float]] | tuple = (),
              housings: list[tuple[float, float]] | tuple = (),
              margin: float = 0.05) -> list[tuple[float, bool]]:
    """Seats along one wall of a bus, as (rear edge, on a wheel housing) from the front backwards.

    One seat sits centred on top of each wheel housing (as the hand-built bus has them); the
    others stand on the floor at `pitch`, starting at the front end of every stretch that no door
    (`keep_out`) or housing covers, with `margin` to each, and all of them inside start..end.
    """
    rows = [((h0 + h1) / 2 - depth / 2, True) for h0, h1 in housings
            if start <= (h0 + h1) / 2 - depth / 2 and (h0 + h1) / 2 + depth / 2 <= end]
    for s0, s1 in free_stretches(start, end, list(keep_out) + list(housings), margin):
        y0 = s1 - depth
        while y0 >= s0 - 1e-9:
            rows.append((y0, False))
            y0 -= pitch
    return sorted(rows, key=lambda row: -row[0])


def free_stretches(start: float, end: float, keep_out: list[tuple[float, float]] | tuple,
                   margin: float) -> list[tuple[float, float]]:
    """The parts of start..end that no keep-out stretch covers, with `margin` to each."""
    stretches = [(start, end)]
    for d0, d1 in sorted(keep_out):
        cut = []
        for s0, s1 in stretches:
            if d1 + margin <= s0 or d0 - margin >= s1:
                cut.append((s0, s1))
                continue
            if d0 - margin > s0:
                cut.append((s0, d0 - margin))
            if d1 + margin < s1:
                cut.append((d1 + margin, s1))
        stretches = cut
    return stretches


def bench_runs(start: float, end: float, keep_out: list[tuple[float, float]] | tuple = (),
               margin: float = 0.25, shortest: float = 0.9) -> list[tuple[float, float]]:
    """Long benches along one wall of a metro car, as (rear end, front end) from the front backwards:
    every stretch between the cabs (start..end) that no door covers, `margin` clear of each door,
    and at least `shortest` long (room for two people)."""
    runs = [(s0, s1) for s0, s1 in free_stretches(start, end, keep_out, margin) if s1 - s0 >= shortest - 1e-9]
    return sorted(runs, key=lambda run: -run[1])


def door_poles(doors: list[tuple[float, float]], x: float, front_limit: float) -> list[tuple[float, float]]:
    """Handrail poles at both edges of every door, left out where an edge lies ahead of the
    driver's partition (`front_limit`): there the dashboard stands."""
    return [(x, y) for y0, y1 in doors for y in (y0, y1) if y < front_limit]


def family_of(preset: str) -> str:
    """The family (and so the Blender builder) a preset belongs to."""
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; the presets are {', '.join(PRESETS)}")
    return PRESETS[preset]


def read_spec(path: Path) -> dict:
    """A spec file: a known preset, an optional name, and numbers set on top of the preset."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("preset") not in PRESETS:
        raise SystemExit(f"{path}: name a preset ({', '.join(PRESETS)})")
    return spec


def overrides(values: dict) -> dict:
    """A spec's numbers as the builders read them: JSON keeps only string keys, so the sides of a
    vehicle (-1 left, 1 right) come back as numbers."""
    def side(key):
        return int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key
    return {key: ({side(k): v for k, v in value.items()} if isinstance(value, dict) else value)
            for key, value in values.items()}


def blender_command(blender: Path, preset: str, out_dir: Path, name: str, tex: int,
                    spec: Path | None = None) -> list[str]:
    """The headless Blender run of the preset's family builder."""
    command = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
               "--python", str(SCRIPTS / FAMILIES[family_of(preset)]), "--",
               "--out", str(out_dir), "--preset", preset, "--name", name, "--tex", str(tex)]
    if spec is not None:
        command += ["--spec", str(spec)]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="folder for the GLBs and the atlas")
    parser.add_argument("--preset", choices=list(PRESETS), help="a vehicle's numbers")
    parser.add_argument("--spec", type=Path, help="a JSON file: a preset plus your own numbers")
    parser.add_argument("--name", help="file and object name (default: the preset's)")
    parser.add_argument("--tex", type=int, default=4096, help="atlas size in pixels (default 4096)")
    parser.add_argument("--blender", type=Path, help="Blender to use (default: I2L_BLENDER, then the usual places)")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs Blender; its parts are tested
    args = build_parser().parse_args(argv)
    if args.spec is None and args.preset is None:
        raise SystemExit("give --preset or --spec")
    spec = read_spec(args.spec) if args.spec is not None else {}
    preset = spec.get("preset", args.preset)
    name = args.name or spec.get("name") or preset
    sys.path.insert(0, str(SCRIPTS.parent))
    from image_to_3dlab.blender import find_blender

    blender = args.blender or find_blender()
    if blender is None:
        raise SystemExit("needs Blender 4.2 or newer: set I2L_BLENDER or pass --blender")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.spec.resolve() if args.spec is not None else None
    done = subprocess.run(blender_command(blender, preset, args.out_dir.resolve(), name, args.tex, spec_path), check=False)
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
