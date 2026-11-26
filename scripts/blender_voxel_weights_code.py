"""Build the payload sent to Blender for voxel proxy weights.

Run (internal): imported by blender_voxel_weights.py.
"""

from __future__ import annotations

from pathlib import Path

from blender_worker_source import build_worker_code


def build_code(voxel_size: float, weight_threshold: float, keep_proxy: bool) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_rig_weights_worker.py"),
        {
            "operation": "voxel", "voxel_size": voxel_size,
            "weight_threshold": weight_threshold, "keep_proxy": bool(keep_proxy),
        },
    )
