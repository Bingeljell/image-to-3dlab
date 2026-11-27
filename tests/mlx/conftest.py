"""Use only CPU execution for the small MLX numerical units unless a test selects Metal."""

import pytest


@pytest.fixture(autouse=True)
def mlx_cpu_device():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)
