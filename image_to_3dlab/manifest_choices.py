"""Allowed values for settings shared by the CLI flags, run manifests and backends."""

from __future__ import annotations

MANIFEST_CHOICES = {
    "remesh": ("none", "triangle", "quad"),
    "pipeline_type": ("512", "1024", "1024_cascade"),
    "texture_size": (512, 1024, 2048),
    "material_mode": ("matte", "pbr"),
}
