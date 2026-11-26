"""Material and texture edge cases with tiny real GLBs and images."""

import runpy
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import (
    colour_match_albedo,
    restore_pbr_material,
    surface_detail,
)
from scripts import (
    compress_glb_textures as cg,
)
from scripts import (
    living_organic_material as organic,
)


def test_restore_cli_keeps_source_bytes_then_replays_idempotently(tmp_path, monkeypatch, capsys):
    source, out = tmp_path / "source.glb", tmp_path / "restored.glb"
    doc = {"asset": {"version": "2.0"}, "textures": [{"source": 0}, {"source": 1}],
           "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}
    data = cg.build_glb(doc, b"mesh")
    source.write_bytes(data)
    cli = restore_pbr_material.__file__
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", [cli, str(source), "--output", str(out), "--roughness-scale", "0.75"])
    runpy.run_path(cli, run_name="__main__")
    assert "attached metallicRoughnessTexture=1" in capsys.readouterr().out
    result, binary = cg.parse_glb(out.read_bytes())
    assert binary == b"mesh" and source.read_bytes() == data
    material = result["materials"][0]
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == 0.75
    assert material["doubleSided"] is False
    monkeypatch.setattr(sys, "argv", [cli, str(out), "--mr-texture", "1", "--roughness-scale", "0.75"])
    restore_pbr_material.main()
    assert "already PBR" in capsys.readouterr().out


def test_compression_handles_slots_without_a_source_and_external_images():
    doc = {"asset": {"version": "2.0"}, "textures": [{}, {"extensions": {"EXT_texture_webp": {"source": 0}}}],
           "images": [{"uri": "base.png"}], "bufferViews": [{"byteOffset": 0, "byteLength": 4}],
           "materials": [{"normalTexture": {"index": 0}}, {"normalTexture": {"index": 1}}]}
    assert cg.normal_texture_images(doc) == {0}
    raw = cg.build_glb(doc, b"geom")
    assert cg.compress(raw) == (raw, [])
    assert cg.has_visible_alpha(Image.new("RGB", (2, 2))) is False
    assert cg.has_visible_alpha(Image.new("LA", (2, 2), (40, 0))) is True


def test_prune_preserves_external_uri_and_retained_sampler():
    doc = {
        "images": [{"uri": "discard.png"}, {"uri": "colour.png"}, {"bufferView": 0}],
        "textures": [{"source": 0}, {"source": 1, "sampler": 0}, {"extensions": {"EXT_texture_webp": {"source": 2}}}],
        "samplers": [{"wrapS": 10497}, {"wrapS": 33071}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}, "normalTexture": {"index": 2}}],
        "bufferViews": [{"byteOffset": 0, "byteLength": 4}],
    }
    assert cg.prune_unused(doc) == 1
    assert doc["images"] == [{"uri": "colour.png"}, {"bufferView": 0}]
    assert doc["textures"] == [
        {"source": 0, "sampler": 0}, {"extensions": {"EXT_texture_webp": {"source": 1}}},
    ]
    assert doc["samplers"] == [{"wrapS": 10497}]


def make_material_asset(path, *, atlas=True):
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.vertices[:, :2] + 0.5,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (2, 2), (50, 100, 150)) if atlas else None,
        ),
    )
    mesh.export(path)
    return mesh


@pytest.mark.parametrize("module", [colour_match_albedo, surface_detail])
def test_colour_and_detail_report_missing_atlas(tmp_path, monkeypatch, module):
    source, target = tmp_path / "no-atlas.glb", tmp_path / "output.glb"
    make_material_asset(source, atlas=False)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (2, 2), (250, 0, 0)).save(reference)
    args = [str(source), str(target)] if module is surface_detail else [str(source), str(reference), str(target)]
    monkeypatch.setattr(sys, "argv", [module.__file__, *args])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(SystemExit, match="no baseColorTexture"):
        runpy.run_path(module.__file__, run_name="__main__")
    assert not target.exists()


def test_surface_detail_zero_radius_returns_original_height_values():
    height = np.array([[0.0, 1.0], [0.5, 0.25]])
    np.testing.assert_array_equal(surface_detail._blur(height, 0), height)


def test_organic_recipe_rejects_missing_uv_atlas_and_out_of_range_chroma(tmp_path):
    source = tmp_path / "shape.glb"
    mesh = make_material_asset(source, atlas=False)
    kwargs = dict(body_colour=np.array([0.1, 0.5, 0.3]), moss_colour=np.array([0.3, 0.8, 0.4]),
                  eye_colour=np.array([0.3, 0.2, 0.1]), body_strength=0, moss_strength=0,
                  eye_strength=0, rough_low=0.6, rough_high=0.9, eye_roughness=0.25,
                  moss_blur=0, lightness_gamma=1, eye_lightness=1, moss_up_threshold=0.1, moss_up_softness=0.4)
    eye = Image.new("L", (2, 2))
    with pytest.raises(ValueError, match="no baseColorTexture"):
        organic.apply_recipe(mesh, eye, **kwargs)
    mesh.visual.uv = None
    with pytest.raises(ValueError, match="no UV coordinates"):
        organic.moss_mask(mesh, (2, 2))
    mesh = make_material_asset(source)
    with pytest.raises(ValueError, match=r"moss chroma strength must be in \[0, 1\]"):
        organic.apply_recipe(mesh, eye, moss_chroma_strength=1.1, **kwargs)


def test_organic_colour_can_use_single_strength_for_every_texel():
    rgb = np.array([[[0.1, 0.6, 0.4], [0.2, 0.3, 0.7]]])
    np.testing.assert_allclose(organic.colourise_preserving_lightness(rgb, np.array([1, 0, 0]), 0),
                               rgb, atol=1e-5)
