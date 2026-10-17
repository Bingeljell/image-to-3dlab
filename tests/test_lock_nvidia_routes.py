"""scripts/lock_nvidia_routes.py: the generator behind scripts/locks/*.txt."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import lock_nvidia_routes as gen


def test_pytorch_and_its_cuda_wheels_stay_out_of_a_lock():
    for line in ("torch==2.8.0", "torchvision==0.23.0", "triton==3.4.0",
                 "nvidia-cudnn-cu12==9.10.2.21", "cuda-toolkit==13.0.3.0"):
        assert gen.keep_line(line) is False, line
    for line in ("numpy==1.24.4", "torchmetrics==1.6.0", "torchdiffeq==0.2.5"):
        assert gen.keep_line(line) is True, line


def test_hunyuan_inputs_keep_the_pins_and_drop_the_demo():
    upstream = "gradio==5.33.0\nbpy==4.0\nopen3d==0.18.0\nnumpy==1.24.4\n"
    inputs = gen.hunyuan_inputs(upstream)
    assert "bpy==4.2.0" in inputs and "open3d==0.18.0" in inputs
    assert "setuptools<81" in inputs and not any("gradio" in i for i in inputs)

