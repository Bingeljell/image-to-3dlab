"""Tests for the GLB forensic dump.

The load-bearing case is `material_summary`: the difference between our output and a
reference asset turned out to be a *deleted metallicRoughness texture*, which is invisible
in a screenshot. If this reports channels wrongly we would compare the wrong things again.

`boundary_edges` is tested against shapes whose answer is known by hand (a box is closed,
a single triangle has three), because the repo previously trusted a hole metric that was
silently counting flipped faces.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "glb_forensics.py"


def _load():
    spec = importlib.util.spec_from_file_location("glb_forensics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gf = _load()


# --- material_summary: the channel that went missing ------------------------------

def test_reports_metallic_roughness_texture_when_present():
    gltf = {
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": False,
            }
        ]
    }
    (mat,) = gf.material_summary(gltf)
    assert mat["textures"] == ["baseColor", "metallicRoughness"]
    assert mat["doubleSided"] is False
    assert mat["metallicFactor"] == 1.0


def test_reports_missing_metallic_roughness_texture():
    """Exactly what `--material-mode matte` leaves behind: flat factors, no map."""
    gltf = {
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ]
    }
    (mat,) = gf.material_summary(gltf)
    assert mat["textures"] == ["baseColor"]
    assert "metallicRoughness" not in mat["textures"]
    assert mat["doubleSided"] is True
    assert mat["metallicFactor"] == 0.0


def test_reports_extra_channels_and_alpha_mode_default():
    gltf = {
        "materials": [
            {
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 2},
                "occlusionTexture": {"index": 3},
            }
        ]
    }
    (mat,) = gf.material_summary(gltf)
    assert mat["textures"] == ["baseColor", "normal", "occlusion"]
    assert mat["alphaMode"] == "OPAQUE"  # glTF default when absent


def test_no_materials_is_empty_not_an_error():
    assert gf.material_summary({}) == []


# --- geometry_summary: counts we can verify by hand -------------------------------

def test_closed_box_has_no_boundary_edges():
    summary = gf.geometry_summary(trimesh.creation.box())
    assert summary["boundary_edges"] == 0
    assert summary["watertight"] is True
    assert summary["winding_consistent"] is True
    assert summary["volume"] > 0


def test_single_triangle_has_three_boundary_edges():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    assert gf.geometry_summary(mesh)["boundary_edges"] == 3


def test_boundary_count_ignores_uv_seam_splits():
    """A vertex duplicated at the same position is a seam, not a hole.

    This is `mesh-topology-measurement-trap`: without merging by position first, an atlas
    seam inflates the hole count and we chase a defect that is not there.
    """
    box = trimesh.creation.box()
    split = box.copy()
    split.unmerge_vertices()  # every face gets its own vertices, positions unchanged
    assert len(split.vertices) > len(box.vertices)
    assert gf.geometry_summary(split)["boundary_edges"] == 0


def test_inside_out_box_reports_negative_volume():
    flipped = trimesh.creation.box()
    flipped.faces = np.fliplr(flipped.faces)
    assert gf.geometry_summary(flipped)["volume"] < 0


def test_uniform_mesh_has_lower_edge_cv_than_irregular_one():
    """Edge-length CV is how we tell a remeshed asset from a decimated one."""
    uniform = trimesh.creation.icosphere(subdivisions=3)
    irregular = uniform.copy()
    rng = np.random.default_rng(0)
    irregular.vertices += rng.normal(scale=0.06, size=irregular.vertices.shape)
    assert (
        gf.geometry_summary(uniform)["edge_length_cv"]
        < gf.geometry_summary(irregular)["edge_length_cv"]
    )


# --- container handling ------------------------------------------------------------

def test_read_gltf_json_rejects_non_glb(tmp_path):
    bogus = tmp_path / "not.glb"
    bogus.write_bytes(b"this is not a GLB file at all, not even close")
    with pytest.raises(ValueError, match="not a binary glTF"):
        gf.read_gltf_json(bogus)


def test_inspect_roundtrips_a_real_exported_glb(tmp_path):
    """Test the real artifact: export a GLB, read it back through the public entry point."""
    path = tmp_path / "box.glb"
    path.write_bytes(trimesh.creation.box().export(file_type="glb"))

    report = gf.inspect(path)
    assert report["file"] == str(path)
    (geom,) = report["geometry"].values()
    assert geom["faces"] == 12
    assert geom["boundary_edges"] == 0
    assert geom["watertight"] is True
