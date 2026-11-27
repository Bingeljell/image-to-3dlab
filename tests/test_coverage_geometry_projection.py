"""Exercise image projection through tiny mesh and image files."""

import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import project_labels, project_markings


@pytest.fixture
def box_file(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box().export(path)
    return path


@pytest.fixture
def textured_triangle(tmp_path):
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0], [1, -1, 0], [-1, 1, 0]], faces=[[0, 1, 2]], process=False
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0, 0], [1, 0], [0, 1]],
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (5, 5), (200, 200, 200))
        ),
    )
    path = tmp_path / "triangle.glb"
    mesh.export(path)
    return path


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_subject_crop_and_missing_subject(mode):
    background = (255, 255, 255) if mode == "RGB" else (0, 0, 0, 0)
    image = Image.new(mode, (9, 9), background)
    with pytest.raises(ValueError, match="could not find a subject"):
        project_labels.subject_bbox(image)
    image.paste("red", (1, 3, 8, 6))
    assert project_labels.subject_bbox(image) == (1, 3, 7, 5)
    assert project_labels.crop_box(image) == (1, 1, 7, 7)


@pytest.mark.parametrize("depth_flip", [False, True])
@pytest.mark.parametrize("buffer", [0, 8])
def test_projection_rejects_back_vertices_after_turning_camera(depth_flip, buffer):
    mesh = trimesh.creation.box()
    image = Image.new("RGB", (5, 5), "red")
    colours, visible, u, v, _ = project_labels.project(
        mesh, image, "z", True, True, depth_flip, buffer, .02, yaw=90
    )
    np.testing.assert_array_equal(colours, np.tile([255, 0, 0], (8, 1)))
    np.testing.assert_array_equal(visible, mesh.vertices[:, 0] < 0 if depth_flip else mesh.vertices[:, 0] > 0)
    np.testing.assert_allclose(u, mesh.vertices[:, 2] + .5, atol=1e-12)
    np.testing.assert_allclose(v, mesh.vertices[:, 1] + .5, atol=1e-12)


@pytest.mark.parametrize("fill,painted,expected_hidden", [(False, True, 4), (True, True, 0), (True, False, 8)])
def test_label_cli_reads_mask_snaps_palette_and_exports(
    box_file, tmp_path, monkeypatch, capsys, fill, painted, expected_hidden
):
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    output = tmp_path / "results" / "projected.glb"
    Image.new("RGB", (5, 5), "blue").save(source)
    Image.new("RGBA", (5, 5), (4, 240, 4, 255 if painted else 0)).save(mask)
    args = [
        "project_labels", str(box_file), str(mask), str(output), "--source", str(source),
        "--depth-buffer", "8", "--snap", "--auto-yaw", "--yaw-step", "90",
    ]
    if fill:
        args.append("--fill")
    monkeypatch.setattr(sys, "argv", args)
    assert project_labels.main() == 0
    colours = trimesh.load(output, force="mesh").visual.vertex_colors[:, :3]
    assert np.all(colours[np.any(colours != 20, axis=1)] == [0, 255, 0])
    assert np.all(colours == 20, axis=1).sum() == expected_hidden
    recorded = capsys.readouterr().out
    assert "AUTOYAW::" in recorded
    assert "FILL::" in recorded if fill and painted else "FILL::" not in recorded


def test_label_cli_rejects_different_mask_dimensions(box_file, tmp_path, monkeypatch):
    source, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (5, 5), "blue").save(source)
    Image.new("RGB", (3, 3), "green").save(mask)
    monkeypatch.setattr(sys, "argv", ["labels", str(box_file), str(mask), "unused.glb", "--source", str(source)])
    with pytest.raises(SystemExit, match="must be the same size"):
        project_labels.main()


def test_label_helpers_handle_empty_silhouette_and_equal_depth():
    empty = np.array([])
    assert project_labels.silhouette_iou(empty, empty, np.zeros((2, 2), dtype=bool)) == 0
    assert not project_labels.occluded_binned(np.zeros(2), np.zeros(2), np.ones(2), 2, .02).any()


@pytest.mark.parametrize(
    "flags,with_erased,expected",
    [
        (["--validate", "--mirror"], False, 80),
        ([], False, 80),
        ([], True, 100),
        (["--no-dark-cores"], True, 100),
    ],
)
def test_marking_cli_updates_only_projected_texels(
    textured_triangle, tmp_path, monkeypatch, flags, with_erased, expected
):
    source, erased = tmp_path / "marked.png", tmp_path / "erased.png"
    output, strength = tmp_path / "output.glb", tmp_path / "strength.png"
    Image.new("RGB", (5, 5), (80, 80, 80)).save(source)
    Image.new("RGB", (5, 5), (160, 160, 160)).save(erased)
    args = [
        "project_markings", str(textured_triangle), str(source), str(output),
        "--depth-buffer", "8", "--dilate", "0", "--dump-strength", str(strength),
        *flags,
    ]
    if with_erased:
        args.extend(["--erased", str(erased)])
    monkeypatch.setattr(sys, "argv", args)
    assert project_markings.main() == 0
    mesh = project_markings.load_mesh(output)
    pixels = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    np.testing.assert_array_equal(pixels[0, 0], [expected] * 3)
    np.testing.assert_array_equal(pixels[0, -1], [200] * 3)
    if "--validate" not in flags:
        assert Image.open(strength).getpixel((0, 0)) == 255


def test_markings_reject_mesh_without_texture_coordinates(box_file):
    with pytest.raises(ValueError, match="no UV coordinates"):
        project_markings.load_mesh(box_file)


def test_marking_pair_ignores_body_pixels_with_zero_headroom():
    original = np.array([[[200, 200, 200], [80, 80, 80]]], dtype=np.uint8)
    erased = np.array([[[200, 200, 200], [140, 140, 140]]], dtype=np.uint8)
    np.testing.assert_allclose(project_markings.strength_from_pair(original, erased), [[0, .5]])


@pytest.mark.parametrize("dump", [False, True])
def test_organic_recipe_preserves_geometry_and_records_outputs(
    textured_triangle, tmp_path, monkeypatch, dump
):
    import hashlib
    import json

    from scripts import living_organic_material

    output, eye = tmp_path / "finished.glb", tmp_path / "eye.png"
    mask = Image.new("L", (5, 5))
    mask.putpixel((0, 0), 255)
    mask.save(eye)
    args = [
        "living_organic_material", str(textured_triangle), str(output), "--eye-mask", str(eye),
        "--moss-up-threshold", "-1", "--moss-up-softness", ".5", "--moss-blur", "0",
        "--eye-lightness", ".3",
    ]
    recipe = output.with_suffix(".recipe.json")
    if dump:
        recipe = tmp_path / "metadata" / "recipe.json"
        args.extend(["--dump-dir", str(tmp_path / "maps"), "--recipe", str(recipe)])
    monkeypatch.setattr(sys, "argv", args)
    assert living_organic_material.main() == 0
    initial, finished = map(project_markings.load_mesh, (textured_triangle, output))
    np.testing.assert_array_equal(initial.faces, finished.faces)
    np.testing.assert_allclose(initial.vertices, finished.vertices)
    material = finished.visual.material
    assert material.metallicFactor == 0
    assert material.roughnessFactor == 1
    assert material.alphaMode == "OPAQUE"
    assert material.doubleSided is False
    roughness = np.asarray(material.metallicRoughnessTexture)
    assert roughness[0, 0, 1] == int(.18 * 255)
    assert roughness[0, 1, 1] > roughness[0, 0, 1]
    assert not roughness[..., 2].any()
    recorded = json.loads(recipe.read_text())
    for field, path in (("input", textured_triangle), ("output", output), ("eye_mask", eye)):
        assert recorded[field] == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert recorded["parameters"]["eye_lightness"] == .3
    if dump:
        moss = Image.open(tmp_path / "maps" / "moss_mask.png")
        assert moss.getpixel((0, 0)) == 0
        assert moss.getpixel((0, 1)) == 255
        assert {p.name for p in (tmp_path / "maps").iterdir()} == {
            "basecolor.png", "metallic_roughness.png", "moss_mask.png", "eye_mask.png",
        }


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--moss-strength", "-1", "--moss-strength must be"),
        ("--rough-low", "1", "require 0 <= rough-low"),
        ("--eye-roughness", "-1", "--eye-roughness must be"),
    ],
)
def test_organic_cli_validates_parameters_before_reading_inputs(monkeypatch, option, value, message):
    from scripts import living_organic_material

    monkeypatch.setattr(sys, "argv", ["recipe", "in.glb", "out.glb", "--eye-mask", "eye.png", option, value])
    with pytest.raises(SystemExit, match=message):
        living_organic_material.main()


@pytest.mark.parametrize("value", ["#fff", "#ffxx00"])
def test_organic_colour_rejects_malformed_hex(value):
    from scripts import living_organic_material

    with pytest.raises(ValueError, match="#RRGGBB"):
        living_organic_material.parse_colour(value)


def test_organic_chroma_detects_green_only():
    from scripts import living_organic_material

    rgb = np.array([[[0, 1, 0], [1, 0, 0], [.5, .5, .5]]])
    np.testing.assert_array_equal(living_organic_material.moss_chroma_evidence(rgb), [[1, 0, 0]])
    with pytest.raises(ValueError, match="high must exceed low"):
        living_organic_material.smoothstep(1, 1, np.zeros(1))


def test_organic_moss_mask_uses_face_orientation_and_blur(textured_triangle):
    from scripts import living_organic_material

    mesh = project_markings.load_mesh(textured_triangle)
    assert living_organic_material.moss_mask(mesh, (5, 5), up_axis=1).getbbox() is None
    blurred = living_organic_material.moss_mask(mesh, (5, 5), up_axis=2, blur=.5)
    assert 0 < blurred.getpixel((3, 1)) < 255


def test_alpha_defines_subject_even_for_white_pixels(box_file, tmp_path, monkeypatch):
    from runpy import run_path

    source, output = tmp_path / "source.png", tmp_path / "projected.glb"
    image = Image.new("RGBA", (5, 5), (255, 255, 255, 255))
    image.putpixel((0, 0), (255, 255, 255, 0))
    image.save(source)
    monkeypatch.setattr(sys, "argv", ["labels", str(box_file), str(source), str(output)])
    with pytest.raises(SystemExit) as result:
        run_path(project_labels.__file__, run_name="__main__")
    assert result.value.code == 0
    colours = trimesh.load(output, force="mesh").visual.vertex_colors[:, :3]
    assert np.all(colours == 255, axis=1).sum() == 3
    assert np.all(colours == 20, axis=1).sum() == 5


def test_uv_triangles_outside_atlas_do_not_paint_texels():
    texels = np.array([[10, 10], [20, 10], [10, 20]])
    values, painted = project_markings.rasterize(
        texels, [[0, 1, 2]], np.ones((3, 1)), 3, 3
    )
    assert not painted.any()
    assert not values.any()


@pytest.mark.parametrize(
    "options,error",
    [
        (["--lightness-gamma", "0"], "lightness gamma"),
        (["--moss-up-softness", "0"], "smoothstep high"),
    ],
)
def test_recipe_rejects_invalid_geometry_math_inputs(
    textured_triangle, tmp_path, monkeypatch, options, error
):
    from scripts import living_organic_material

    eye, output = tmp_path / "eye.png", tmp_path / "unused.glb"
    Image.new("L", (2, 2)).save(eye)
    monkeypatch.setattr(
        sys, "argv", ["recipe", str(textured_triangle), str(output), "--eye-mask", str(eye), *options]
    )
    with pytest.raises(ValueError, match=error):
        living_organic_material.main()
    assert not output.exists()


def test_one_albedo_is_graded_toward_reference_and_keeps_geometry(
    textured_triangle, tmp_path, monkeypatch
):
    from scripts import colour_match_albedo

    # Use a range of atlas values so the channel spreads can be matched.
    mesh = project_markings.load_mesh(textured_triangle)
    atlas = Image.new("RGBA", (3, 1), (130, 148, 112, 255))
    atlas.putpixel((1, 0), (135, 145, 119, 255))
    atlas.putpixel((2, 0), (0, 0, 0, 0))
    mesh.visual.material.baseColorTexture = atlas
    mesh.export(textured_triangle)
    source, output, dump = (tmp_path / name for name in ("source.png", "out.glb", "dump.png"))
    reference = Image.new("RGBA", (3, 1), (174, 131, 100, 255))
    reference.putpixel((1, 0), (172, 138, 92, 255))
    reference.putpixel((2, 0), (255, 255, 255, 0))
    reference.save(source)
    monkeypatch.setattr(
        sys, "argv", ["grade", str(textured_triangle), str(source), str(output), "--dump-atlas", str(dump)]
    )
    assert colour_match_albedo.main() == 0
    changed = project_markings.load_mesh(output)
    np.testing.assert_array_equal(changed.faces, mesh.faces)
    np.testing.assert_allclose(changed.vertices, mesh.vertices)
    record = np.asarray(Image.open(dump))
    np.testing.assert_array_equal(record[..., 3], [[255, 255, 0]])
    assert (record[0, :2, 0] > record[0, :2, 1]).all()
    np.testing.assert_array_equal(np.asarray(changed.visual.material.baseColorTexture), record)


@pytest.mark.parametrize("copy", [False, True])
def test_opaque_material_cli_updates_flags_and_preserves_texture(
    textured_triangle, tmp_path, monkeypatch, copy
):
    from scripts import fix_glb_opaque_material

    mesh = project_markings.load_mesh(textured_triangle)
    mesh.visual.material.alphaMode = "BLEND"
    mesh.visual.material.doubleSided = True
    mesh.export(textured_triangle)
    output = tmp_path / "result" / "output.glb" if copy else textured_triangle
    args = ["opaque", str(textured_triangle)]
    if copy:
        args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    assert fix_glb_opaque_material.main() == 0
    changed = project_markings.load_mesh(output).visual.material
    assert changed.alphaMode == "OPAQUE"
    assert changed.doubleSided is False
    assert changed.baseColorTexture.getpixel((0, 0)) == (200, 200, 200)
    if copy:
        assert project_markings.load_mesh(textured_triangle).visual.material.alphaMode == "BLEND"
    else:
        assert fix_glb_opaque_material.main() == 0


def test_opaque_material_cli_requires_a_material(box_file, monkeypatch):
    from scripts import fix_glb_opaque_material

    monkeypatch.setattr(sys, "argv", ["opaque", str(box_file)])
    with pytest.raises(ValueError, match="has no materials"):
        fix_glb_opaque_material.main()


def test_soften_command_preserves_protected_marking_and_alpha(tmp_path, monkeypatch):
    from scripts import soften_markings

    source, protection, output = (tmp_path / name for name in ("source.png", "mask.png", "softened.png"))
    image = Image.new("RGBA", (4, 4), (170, 170, 170, 255))
    image.putpixel((0, 0), (80, 80, 80, 255))
    image.putpixel((1, 0), (80, 80, 80, 255))
    image.putpixel((2, 0), (80, 80, 80, 0))
    image.save(source)
    mask = Image.new("L", image.size)
    mask.putpixel((0, 0), 255)
    mask.save(protection)
    monkeypatch.setattr(sys, "argv", [
        "soften", str(source), str(output), "--protect", str(protection),
        "--lighten", "1", "--feather", "0",
    ])
    assert soften_markings.main() == 0
    softened = Image.open(output)
    assert softened.getpixel((0, 0)) == (80, 80, 80, 255)
    assert softened.getpixel((1, 0)) == (170, 170, 170, 255)
    assert softened.getpixel((2, 0))[3] == 0
    assert softened.getpixel((3, 0)) == (170, 170, 170, 255)


def test_lightness_cli_lifts_colour_in_exported_texture(textured_triangle, tmp_path, monkeypatch):
    from scripts import lift_lightness

    output = tmp_path / "lifted.glb"
    monkeypatch.setattr(sys, "argv", [
        "lift_lightness", str(textured_triangle), str(output), "--gain", "1.3",
    ])
    assert lift_lightness.main() == 0
    mip = project_markings.load_mesh(output).visual.material.baseColorTexture
    assert mip.getpixel((0, 0))[:3] == (224, 224, 224)


@pytest.mark.parametrize("with_masks", [False, True])
def test_surface_maps_are_embedded_and_preserve_albedo(
    textured_triangle, tmp_path, monkeypatch, with_masks
):
    from scripts import surface_detail

    output, ao, gloss, dump = (tmp_path / name for name in ("detail.glb", "ao.png", "gloss.png", "maps"))
    args = ["surface_detail", str(textured_triangle), str(output)]
    if with_masks:
        Image.new("L", (1, 1), 64).save(ao)
        Image.new("L", (1, 1), 255).save(gloss)
        args.extend(["--ao", str(ao), "--gloss-mask", str(gloss), "--dump-maps", str(dump)])
    monkeypatch.setattr(sys, "argv", args)
    assert surface_detail.main() == 0
    material = project_markings.load_mesh(output).visual.material
    assert material.baseColorTexture.getpixel((0, 0)) == (200, 200, 200)
    assert material.normalTexture.getpixel((0, 0)) == (127, 127, 255)
    assert material.metallicFactor == 0
    pixel = material.metallicRoughnessTexture.getpixel((0, 0))
    if with_masks:
        assert pixel == pytest.approx((64, 30, 0), abs=1)
        assert material.occlusionTexture.getpixel((0, 0)) == pixel
        assert Image.open(dump / "normal.png").getpixel((0, 0)) == (127, 127, 255)
    else:
        assert material.occlusionTexture is None
        assert pixel[1] > 30
