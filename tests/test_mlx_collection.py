"""Keep numerical tests out of portable test collection."""

from pathlib import Path

import pytest
from conftest import pytest_collection_modifyitems, pytest_ignore_collect


class Config:
    def __init__(self, enabled):
        self.enabled = enabled

    def getoption(self, option):
        assert option == "--run-mlx-unit"
        return self.enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_only_explicit_opt_in_collects_mlx_tests(enabled):
    root = Path(__file__).resolve().parent
    assert pytest_ignore_collect(root / "mlx", Config(enabled)) is not enabled
    assert pytest_ignore_collect(root / "mlx/test_example.py", Config(enabled)) is not enabled
    assert not pytest_ignore_collect(root / "test_example.py", Config(enabled))


@pytest.mark.parametrize("enabled", [False, True])
def test_numerical_selection_requires_separate_metal_opt_in(enabled):
    from types import SimpleNamespace

    deselected = []
    config = SimpleNamespace(
        getoption=lambda flag: enabled if flag == "--run-mlx-metal" else False,
        hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)),
    )
    cpu = SimpleNamespace(keywords={"mlx_unit"})
    metal = SimpleNamespace(keywords={"mlx_unit", "mlx_metal"})
    cases = [cpu, metal]
    pytest_collection_modifyitems(config, cases)
    assert cases == ([cpu, metal] if enabled else [cpu])
    assert deselected == ([] if enabled else [metal])
