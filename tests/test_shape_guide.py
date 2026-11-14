"""Tests for `scripts/shape_guide.py`: the layout maths of the shape-guide kit, without Blender.

The kit builds a detailed vehicle from a few numbers (#259). Where a hand-built model placed every window by hand, the
kit spreads them by a rule; the bus it started from is the yardstick.
"""

import sys
from itertools import pairwise
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import shape_guide as sg


def test_a_side_without_doors_comes_out_like_the_hand_built_bus():
    # the bus's left side, placed by hand: seven panes between 0.10 m pillars from 5.12 back to -4.48 (the first
    # 1.20 m, the others 1.30 m); the rule spreads them evenly and lands within 10 cm of every hand-placed edge
    panes = sg.spread_windows(-4.48, 5.12, pitch=1.40, pillar=0.10)
    by_hand = [(3.92, 5.12), (2.52, 3.82), (1.12, 2.42), (-0.28, 1.02), (-1.68, -0.38), (-3.08, -1.78), (-4.48, -3.18)]
    assert len(panes) == len(by_hand)
    for (a, b), (c, d) in zip(panes, by_hand):
        assert abs(a - c) < 0.10 and abs(b - d) < 0.10


def test_windows_keep_a_pillar_clear_of_every_door():
    doors = [(3.92, 5.22), (-0.10, 1.25), (-4.62, -3.32)]
    panes = sg.spread_windows(-5.40, 5.12, pitch=1.40, pillar=0.20, keep_out=doors)
    for y0, y1 in panes:
        for d0, d1 in doors:
            assert y1 <= d0 - 0.2 + 1e-9 or y0 >= d1 + 0.2 - 1e-9


def test_the_panes_of_one_stretch_are_equal_and_run_front_to_back():
    panes = sg.spread_windows(-3.0, 3.0, pitch=1.40, pillar=0.20)
    lengths = {round(y1 - y0, 6) for y0, y1 in panes}
    assert len(lengths) == 1
    assert [y0 for y0, _ in panes] == sorted((y0 for y0, _ in panes), reverse=True)
    assert panes[0][1] == pytest.approx(3.0) and panes[-1][0] == pytest.approx(-3.0)


def test_a_stretch_too_short_for_a_window_stays_closed():
    # between two doors 0.6 m apart there is no room for a pane and two pillars
    panes = sg.spread_windows(-5.0, 5.0, pitch=1.40, pillar=0.20, keep_out=[(-2.0, -0.3), (0.3, 2.0)])
    assert all(not (-0.3 < (y0 + y1) / 2 < 0.3) for y0, y1 in panes)


def test_the_pitch_must_leave_room_for_glass():
    with pytest.raises(ValueError, match="pitch"):
        sg.spread_windows(-3.0, 3.0, pitch=0.20, pillar=0.20)


# --- the interior: seats and poles by rule ----------------------------------------------------

HOUSINGS = [(2.62, 3.88), (-3.28, -2.02)]       # the 12 m bus's wheel housings (axle +- 0.63)


def test_one_seat_sits_centred_on_each_wheel_housing():
    rows = sg.seat_rows(-4.73, 4.25, pitch=0.80, depth=0.44, housings=HOUSINGS)
    high = [y0 for y0, on_housing in rows if on_housing]
    # the hand-built bus has them at 3.0 and -2.9 (rear edges); centred on the housings they come out within 5 cm
    assert [round(y, 2) for y in sorted(high, reverse=True)] == [3.03, -2.87]


def test_floor_seats_keep_clear_of_doors_and_housings_and_stay_in_the_room():
    doors = [(3.92, 5.22), (-0.10, 1.25), (-4.62, -3.32)]
    rows = sg.seat_rows(-4.73, 4.25, pitch=0.80, depth=0.44, keep_out=doors, housings=HOUSINGS)
    for y0, on_housing in rows:
        y1 = y0 + 0.44
        assert -4.73 - 1e-9 <= y0 and y1 <= 4.25 + 1e-9
        if not on_housing:
            for a, b in doors + HOUSINGS:
                assert y1 <= a or y0 >= b


def test_seats_run_front_to_back_at_the_pitch_on_a_wall_without_doors():
    rows = sg.seat_rows(-2.0, 2.0, pitch=0.80, depth=0.44)
    starts = [y0 for y0, _ in rows]
    assert starts == sorted(starts, reverse=True)
    assert all(abs((a - b) - 0.80) < 1e-9 for a, b in pairwise(starts))


def test_metro_benches_run_along_the_wall_between_the_doors_and_the_cabs():
    # an 18 m metro car: four doors a side, cabs behind 7.3 and ahead of -7.3; benches keep 0.25 m from each
    doors = [(5.3, 6.7), (1.3, 2.7), (-2.7, -1.3), (-6.7, -5.3)]
    runs = sg.bench_runs(-7.3, 7.3, keep_out=doors, margin=0.25)
    assert [round(y, 6) for run in runs for y in run] == [2.95, 5.05, -1.05, 1.05, -5.05, -2.95]


def test_a_gap_too_short_for_a_bench_stays_empty():
    # 0.3 m between the cab and the first door leaves no room for a seat
    runs = sg.bench_runs(-3.0, 3.0, keep_out=[(1.9, 2.7)], margin=0.25, shortest=0.9)
    assert [round(y, 6) for run in runs for y in run] == [-3.0, 1.65]


def test_handrail_poles_stand_at_both_edges_of_every_door_behind_the_driver():
    poles = sg.door_poles([(3.92, 5.22), (-0.10, 1.25)], x=0.50, front_limit=4.30)
    assert sorted(poles) == sorted([(0.50, 3.92), (0.50, -0.10), (0.50, 1.25)])


# --- the command line: presets, families, a spec file -----------------------------------------

def test_every_preset_has_a_family_with_a_blender_builder():
    for preset in sg.PRESETS:
        family = sg.family_of(preset)
        assert (HERE.parent / "scripts" / sg.FAMILIES[family]).is_file()


def test_an_unknown_preset_names_the_known_ones():
    with pytest.raises(SystemExit, match="midibus_9m"):
        sg.family_of("tram")


def test_a_spec_file_sets_numbers_on_top_of_its_preset(tmp_path):
    spec = tmp_path / "van.json"
    spec.write_text('{"preset": "van_long", "name": "delivery", "set": {"rear_axle": -2.4}}', encoding="utf-8")
    loaded = sg.read_spec(spec)
    assert loaded == {"preset": "van_long", "name": "delivery", "set": {"rear_axle": -2.4}}


def test_a_spec_file_without_a_known_preset_is_refused(tmp_path):
    spec = tmp_path / "bad.json"
    spec.write_text('{"set": {"rear_axle": -2.4}}', encoding="utf-8")
    with pytest.raises(SystemExit, match="preset"):
        sg.read_spec(spec)


def test_overrides_turn_side_keys_back_into_numbers():
    # JSON has only string keys; the builders look sides up as -1 and 1
    over = sg.overrides({"band_rear": {"-1": -3.1, "1": -4.0}, "doors": [[2.8, 3.7]], "livery": "plain"})
    assert over == {"band_rear": {-1: -3.1, 1: -4.0}, "doors": [[2.8, 3.7]], "livery": "plain"}


def test_the_blender_command_runs_the_family_builder_headless():
    command = sg.blender_command(Path("blender"), "box_van", Path("out"), "truck", 4096, spec=Path("s.json"))
    assert command[:2] == ["blender", "--background"]
    assert any(part.endswith("blender_shape_guide_van.py") for part in command)
    tail = command[command.index("--") + 1:]
    assert tail[tail.index("--preset") + 1] == "box_van" and tail[tail.index("--name") + 1] == "truck"
    assert tail[tail.index("--tex") + 1] == "4096" and tail[tail.index("--spec") + 1] == "s.json"
