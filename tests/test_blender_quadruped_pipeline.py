"""Pure contract tests for the reusable staged quadruped workflow."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "quad_pipeline",
    Path(__file__).resolve().parents[1] / "scripts/blender_quadruped_pipeline.py",
)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_settings_reject_typos_and_invalid_numbers():
    for values in [
        {"strdie": 0.3},
        {"duty": 1},
        {"frames": 2},
        {"lift": -1},
        {"bob": float("nan")},
        {"fps": 0},
        {"gait": "gallop"},
        {"frames": 16.5},
        {"transition_cycles": 0},
    ]:
        with pytest.raises(ValueError):
            pipeline.settings(values)


def test_diagonal_cycle_and_common_stance_speed():
    cfg = pipeline.settings({})
    for u in [0, 0.1, 0.3, 0.8]:
        assert pipeline.foot(u, "FL", cfg) == pytest.approx(
            pipeline.foot(u + 1, "FL", cfg)
        )
        assert pipeline.foot(u, "FL", cfg) == pytest.approx(pipeline.foot(u, "BR", cfg))
    speeds = []
    for limb, phase in pipeline.phases("trot").items():
        a = pipeline.foot(phase + 0.1, limb, cfg)[0]
        b = pipeline.foot(phase + 0.2, limb, cfg)[0]
        speeds.append((b - a) / 0.1)
    assert speeds == pytest.approx([-cfg["stride"] / cfg["duty"]] * 4)


def test_walk_and_foot_clearance():
    cfg = pipeline.settings({"gait": "walk"})
    assert cfg["duty"] > 0.5
    assert len(set(pipeline.phases("walk").values())) == 4
    for limb in pipeline.phases("walk"):
        for i in range(100):
            _, lift, curl = pipeline.foot(i / 100, limb, cfg)
            assert 0 <= lift <= cfg["lift"] + 1e-10
            assert curl >= 0


def test_transition_schedule():
    cfg = pipeline.settings({})
    total = pipeline.transition_length(cfg)
    assert pipeline.envelope(1, cfg)[:2] == (0, 0)
    assert pipeline.envelope(total, cfg)[:2] == (0, 0)
    for f in range(1, total + 1):
        p, g, _ = pipeline.envelope(f, cfg)
        assert 0 <= g <= p <= 1


def test_same_file_follows_a_link(tmp_path):
    source = tmp_path / "input.blend"
    source.write_text("keep")
    alias = tmp_path / "alias.blend"
    alias.symlink_to(source)
    assert pipeline.same_file(alias, source)
    assert not pipeline.same_file(tmp_path / "other.blend", source)


def test_refuse_open_blend_stops_on_the_opened_file(tmp_path):
    opened = tmp_path / "in.blend"
    opened.write_text("keep")
    with pytest.raises(SystemExit, match="opened"):
        pipeline.refuse_open_blend(opened, opened)
    pipeline.refuse_open_blend(tmp_path / "out.blend", opened)
    pipeline.refuse_open_blend(tmp_path / "out.blend", "")


def test_no_overwrite_or_input_alias(tmp_path):
    source = tmp_path / "input.blend"
    source.write_text("keep")
    with pytest.raises(ValueError):
        pipeline.new_path(source, source)
    alias = tmp_path / "alias.blend"
    alias.symlink_to(source)
    with pytest.raises(ValueError):
        pipeline.new_path(alias, source)
    assert pipeline.new_path(tmp_path / "new.blend", source).name == "new.blend"


def test_alignment_allows_tail_reversal_not_translation():
    a = ((0, 0, 0), (0, 1, 0))
    assert pipeline.endpoint_error(a, a[::-1]) == 0
    assert pipeline.endpoint_error(a, ((1, 0, 0), (1, 1, 0))) == 1
