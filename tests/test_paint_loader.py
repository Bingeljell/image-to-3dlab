"""Paint load refuses a short checkpoint, and DDIM steps back by whole division."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PAINT = REPO / "hunyuan_mlx" / "paint"


def _stub_mlx(monkeypatch) -> None:
    core = types.ModuleType("mlx.core")
    core.array = np.ndarray
    core.eval = lambda *_args, **_kwargs: None
    utils = types.ModuleType("mlx.utils")
    utils.tree_unflatten = lambda pairs: dict(pairs)
    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx.utils = utils
    mlx.__path__ = []
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)


def _paint(monkeypatch):
    _stub_mlx(monkeypatch)
    if str(PAINT) not in sys.path:
        sys.path.insert(0, str(PAINT))
    from hy3dpaint_mlx import convert, scheduler

    return scheduler, convert


class _TwoWeights:
    """A module with two parameters. update records that it ran."""

    def __init__(self) -> None:
        self.params = {"layer": {"weight": np.zeros(1), "bias": np.zeros(1)}}
        self.updated = None

    def update(self, tree) -> None:
        self.updated = tree

    def parameters(self):
        return self.params


def _probe(table):
    seen: list[int] = []

    class _Probe:
        def __getitem__(self, index):
            seen.append(int(index))
            return float(table[index])

    return seen, _Probe()


def test_ddim_previous_step_uses_whole_number_division(monkeypatch):
    scheduler, _convert = _paint(monkeypatch)
    sched = scheduler.DDIMScheduler()
    sample = np.zeros(2, dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    for steps, width in ((15, 66), (20, 50)):
        sched.set_timesteps(steps)
        table = sched.acp
        seen, probe = _probe(table)
        sched.acp = probe
        current = int(sched.timesteps[0])
        sched.step(velocity, current, sample)
        sched.acp = table
        assert seen[1] == current - width


def test_a_short_checkpoint_raises_after_the_update(monkeypatch):
    _scheduler, convert = _paint(monkeypatch)
    module = _TwoWeights()
    with pytest.raises(KeyError, match="bias"):
        convert.load_torch_weights(module, {"layer.weight": np.zeros(1)})
    assert module.updated is not None


def test_a_complete_checkpoint_loads(monkeypatch):
    _scheduler, convert = _paint(monkeypatch)
    module = _TwoWeights()
    convert.load_torch_weights(
        module, {"layer.weight": np.zeros(1), "layer.bias": np.zeros(1)}
    )
    assert module.updated is not None


def test_pbr_dino_goes_through_the_strict_loader():
    text = (PAINT / "scripts" / "run_paint_pbr.py").read_text()
    assert "dino.update" not in text
    assert "load_torch_weights(dino" in text


def test_paint_readme_does_not_offer_the_small_script_as_setup():
    readme = (PAINT / "README.md").read_text()
    block = readme.split("```bash", 1)[1].split("```", 1)[0]
    assert "run_paint.py" not in block
    assert "run_paint_pbr.py" in block
    assert "is not a setup command" in readme
