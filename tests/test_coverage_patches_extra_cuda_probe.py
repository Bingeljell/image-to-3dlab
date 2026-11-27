"""Exercise the CUDA probe in the selected Python process without using a GPU."""

import sys

import pytest

from image_to_3dlab import cuda_arch


@pytest.mark.parametrize(("scenario", "expected"), [
    ("device", {"archs": ["sm_86"], "capability": [8, 6], "device": "Fixture card"}),
    ("cpu", {"archs": ["sm_86"], "capability": None, "device": None}),
    ("broken", {"archs": ["sm_86"], "error": "driver unreadable"}),
])
def test_cuda_probe_facts_are_returned_by_the_real_selected_interpreter(monkeypatch, tmp_path, scenario, expected):
    torch = tmp_path / "torch.py"
    torch.write_text('''import os

class Card:
    @staticmethod
    def get_arch_list():
        return ("sm_86",)

    @staticmethod
    def is_available():
        if os.environ["CUDA_PROBE_FIXTURE"] == "broken":
            raise RuntimeError("driver unreadable")
        return os.environ["CUDA_PROBE_FIXTURE"] == "device"

    @staticmethod
    def get_device_capability(index):
        assert index == 0
        return (8, 6)

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "Fixture card"

cuda = Card()
''')
    monkeypatch.setenv("CUDA_PROBE_FIXTURE", scenario)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert cuda_arch.probe(sys.executable) == expected
    assert str(cuda_arch.CUDA_PROBE_PATH) == cuda_arch.__file__



def test_cuda_probe_fact_collection_preserves_missing_api_error():
    from types import SimpleNamespace

    got = cuda_arch.cuda_probe_facts(SimpleNamespace())
    assert "cuda" in got["error"]
