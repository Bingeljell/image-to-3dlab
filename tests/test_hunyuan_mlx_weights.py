"""The two dgrauet shape commands refuse a hub id when the snapshot is absent."""

from pathlib import Path

import hunyuan_mlx_generate as mlx
import hunyuan_shape_octree_test as octree
import pytest
import runpod_trellis2_cuda_probe as probe


def test_mlx_shape_refuses_without_a_local_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="about 13 GB") as raised:
        mlx.local_shape_repo()
    message = str(raised.value)
    assert "Hunyuan3D-MLX" in message and "dgrauet shape" in message


def test_run_shape_refuses_before_importing_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="about 13 GB"):
        mlx.run_shape(Path("in.png"), 512, 1, 0.0)


def test_octree_script_refuses_the_same_way(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="dgrauet shape"):
        octree.shape_repo()


def test_the_cuda_probe_refuses_without_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="NVIDIA probe"):
        probe.model_dir()
