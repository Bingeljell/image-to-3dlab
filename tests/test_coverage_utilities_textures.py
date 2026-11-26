"""Read and write tiny local meshes to exercise the texture utility commands."""

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import attach_normal_map, feature_mask


def _textured_mesh(vertices, faces, uv, material):
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )


@pytest.mark.parametrize("front_only", [False, True])
def test_feature_mask_cli_can_exclude_back_facing_triangles(monkeypatch, tmp_path, capsys, front_only):
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.1], [0, 1, 0.1], [1, 0, 0.1]]
    faces = [[0, 1, 2], [3, 4, 5]]
    uv = [[0, 1], [0.25, 1], [0, 0.75], [0.75, 0.25], [1, 0.25], [0.75, 0]]
    mesh = _textured_mesh(vertices, faces, uv, trimesh.visual.material.PBRMaterial())
    source, out = tmp_path / "asset.glb", tmp_path / "nested" / "mask.png"
    mesh.export(source)
    opts = ["--front-only", "--facing", "0", "0", "1"] if front_only else []
    monkeypatch.setattr("sys.argv", ["feature_mask.py", str(source), str(out), "--centre", "0", "0", "0",
                                      "--radius", "2", "--size", "9", "--dilate", "0", *opts])
    assert feature_mask.main() == 0
    assert f"{1 if front_only else 2} of 2 faces" in capsys.readouterr().out
    with Image.open(out) as mask:
        assert mask.getpixel((0, 0)) == 255
        assert mask.getpixel((6, 6)) == (0 if front_only else 255)
        assert mask.getpixel((4, 4)) == 0


def test_mask_growth_has_no_wraparound_or_diagonal_growth():
    pixels = np.zeros((5, 5), np.uint8)
    pixels[0, 0] = 255
    grown = feature_mask.dilate_mask(pixels, 2)
    assert np.argwhere(grown).tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]
    assert feature_mask.dilate_mask(pixels, 0) is pixels
    with pytest.raises(ValueError, match="radius must be positive"):
        feature_mask.select_faces(np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros(3), 0)


def test_attach_normal_cli_preserves_source_and_embeds_softened_map(monkeypatch, tmp_path, capsys):
    source = tmp_path / "asset.glb"
    normal_file = tmp_path / "normal.png"
    output = tmp_path / "out" / "normal.glb"
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    mesh = _textured_mesh(positions, [[0, 1, 2]], [[0, 0], [1, 0], [0, 1]],
                          trimesh.visual.material.PBRMaterial(name="paint"))
    mesh.export(source)
    Image.new("RGB", (2, 2), (0, 255, 1)).save(normal_file)
    monkeypatch.setattr("sys.argv", ["attach_normal_map.py", str(source), str(normal_file), str(output),
                                      "--strength", "0.5"])
    assert attach_normal_map.main() == 0
    assert "1 survived the round-trip" in capsys.readouterr().out
    loaded = trimesh.load(output, force="mesh", process=False)
    np.testing.assert_array_equal(loaded.vertices, positions)
    assert loaded.visual.material.normalTexture.getpixel((0, 0))[:3] == (64, 191, 128)
    assert trimesh.load(source, force="mesh").visual.material.normalTexture is None


def test_normal_strength_above_one_clips_and_zero_flattens():
    normal = Image.new("RGB", (1, 1), (40, 200, 50))
    assert attach_normal_map.soften(normal, 0).getpixel((0, 0)) == (128, 128, 255)
    assert attach_normal_map.soften(normal, 2).getpixel((0, 0)) == (0, 255, 0)
    with pytest.raises(ValueError, match="strength must not be negative"):
        attach_normal_map.soften(normal, -0.1)


def test_feature_mask_dilation_expands_raster_without_wrapping():
    uv = np.array([[0, 1], [0.5, 1], [0, 0.5]])
    faces = np.array([[0, 1, 2]])
    bare = feature_mask.rasterise_uv(uv, faces, np.array([True]), 5, dilate=0)
    expanded = feature_mask.rasterise_uv(uv, faces, np.array([True]), 5, dilate=1)
    assert bare.getpixel((0, 3)) == 0
    assert expanded.getpixel((0, 3)) == 255
    assert expanded.getpixel((4, 0)) == expanded.getpixel((0, 4)) == 0
    np.testing.assert_array_equal(feature_mask._shifted(np.asarray(bare), 0, 0), np.asarray(bare))


def test_attach_normal_passes_through_original_at_full_strength():
    image = Image.new("RGB", (2, 2), (12, 123, 250))
    assert attach_normal_map.soften(image, strength=1.0) is image


def test_attach_normal_rejects_a_scene_with_no_geometry(monkeypatch, tmp_path):
    monkeypatch.setattr(attach_normal_map.trimesh, "load", lambda *_a, **_k: trimesh.Scene())
    out = tmp_path / "result.glb"
    monkeypatch.setattr("sys.argv", ["attach_normal_map.py", "empty.glb", "missing.png", str(out)])
    with pytest.raises(SystemExit, match="no geometry in empty.glb"):
        attach_normal_map.main()
    assert not out.exists()


def test_attach_normal_reports_if_writer_drops_texture(monkeypatch, tmp_path):
    source, normal, out = (tmp_path / name for name in ("source.glb", "normal.png", "result.glb"))
    mesh = _textured_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]],
                          [[0, 0], [1, 0], [0, 1]], trimesh.visual.material.PBRMaterial())
    source.write_bytes(mesh.export(file_type="glb"))
    Image.new("RGB", (2, 2)).save(normal)
    scene = trimesh.Scene(mesh)
    unmaterialed = mesh.copy()
    unmaterialed.visual.material = None
    scene.add_geometry(unmaterialed, geom_name="unmaterialed")

    def writer_drops_texture(path):
        path.write_bytes(source.read_bytes())

    scene.export = writer_drops_texture
    load = trimesh.load
    monkeypatch.setattr(attach_normal_map.trimesh, "load",
                        lambda path, **kwargs: scene if path == source else load(path, **kwargs))
    monkeypatch.setattr("sys.argv", ["attach_normal_map.py", str(source), str(normal), str(out)])
    with pytest.raises(SystemExit, match="normalTexture did not survive export"):
        attach_normal_map.main()
    assert out.exists()
