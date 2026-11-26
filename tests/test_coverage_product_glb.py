"""Round-trip model files with mixed materials, texture sources, and transforms."""

import hashlib
import json

import numpy as np
import pytest
import trimesh

from image_to_3dlab import glb_parts, real_height
from image_to_3dlab.glb_turn import build_glb, parse_glb


def mixed_model():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(positions)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "images": [{"uri": "unused.png"}, {"uri": "fallback.png"}, {"uri": "preferred.webp"}],
        "samplers": [{"minFilter": 9729}],
        "textures": [
            {},
            {"source": 1, "sampler": 0, "extensions": {"EXT_texture_webp": {"source": 2}, "EXT_test": {}}},
            {"extensions": {"EXT_texture_webp": {"source": 2}}},
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
                       "normalTexture": {"index": 2},
                       "extras": {"tags": [{"name": "painted"}, {"index": "label"}]}}],
        "meshes": [{"primitives": [
            {"attributes": {"POSITION": 0, "NORMAL": 1}, "material": 0},
            {"attributes": {"POSITION": 0}},
        ]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}],
        "extensionsUsed": ["EXT_texture_webp"],
    }, positions


def test_parts_preserve_external_texture_sources_and_implicit_accessors(tmp_path):
    document, buf = mixed_model()
    source = tmp_path / "mixed.glb"
    source.write_bytes(build_glb(document, buf))
    painted, plain = glb_parts.split(source, tmp_path / "parts")
    doc, packed = parse_glb(painted.read_bytes())
    assert packed[:len(buf)] == buf
    assert doc["images"] == [{"uri": "fallback.png"}, {"uri": "preferred.webp"}]
    assert doc["samplers"] == [{"minFilter": 9729}]
    assert doc["textures"] == [
        {"source": 0, "sampler": 0, "extensions": {"EXT_texture_webp": {"source": 1}, "EXT_test": {}}},
        {"extensions": {"EXT_texture_webp": {"source": 1}}},
    ]
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert doc["materials"][0]["normalTexture"]["index"] == 1
    assert doc["materials"][0]["extras"] == document["materials"][0]["extras"]
    assert "bufferView" not in doc["accessors"][1]
    bare, _ = parse_glb(plain.read_bytes())
    assert bare["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0}
    assert "materials" not in bare

    joined = tmp_path / "joined.glb"
    glb_parts.join([painted, painted, plain], joined)
    doc, _ = parse_glb(joined.read_bytes())
    assert glb_parts.count(joined) == 3
    assert doc["textures"][2]["source"] == 2
    assert doc["textures"][2]["sampler"] == 1
    assert doc["textures"][2]["extensions"]["EXT_texture_webp"]["source"] == 3
    assert doc["materials"][1]["normalTexture"]["index"] == 3
    assert doc["meshes"][0]["primitives"][-1] == {"attributes": {"POSITION": 4}}
    assert doc["extensionsUsed"] == ["EXT_texture_webp"]


def test_split_rejects_unbaked_transform_without_creating_parts(tmp_path):
    document, buf = mixed_model()
    document["nodes"][0]["translation"] = [1, 0, 0]
    source = tmp_path / "translated.glb"
    source.write_bytes(build_glb(document, buf))
    folder = tmp_path / "parts"
    with pytest.raises(ValueError, match="node transforms are not supported"):
        glb_parts.split(source, folder)
    assert not folder.exists()


@pytest.mark.parametrize(("alter", "message"), [
    (lambda doc: doc.update(animations=[{}]), "before rigging"),
    (lambda doc: doc["scenes"].append({}), "requires one scene"),
    (lambda doc: doc["scenes"][0]["nodes"].append(0), "repeated node"),
    (lambda doc: doc["nodes"][0].update(scale=[-1, 1, 1]), "non-mirrored"),
    (lambda doc: doc["nodes"][0].update(scale=[0, 1, 1]), "non-zero"),
    (lambda doc: doc["accessors"][0].update(sparse={}), "plain float"),
    (lambda doc: doc["meshes"][0]["primitives"][0].update(targets=[{}]), "morph or compressed"),
    (lambda doc: doc["nodes"][0].pop("mesh"), "no measurable height"),
])
def test_height_rejects_models_it_cannot_bake(alter, message):
    document, buf = mixed_model()
    alter(document)
    with pytest.raises(ValueError, match=message):
        real_height.set_height(build_glb(document, buf), 2)


@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf")])
def test_height_rejects_invalid_number_before_parsing_mesh(height):
    with pytest.raises(ValueError, match="positive finite"):
        real_height.set_height(b"", height)


@pytest.mark.parametrize("with_sidecar", [True, False])
def test_height_keeps_normals_and_updates_existing_provenance(tmp_path, with_sidecar):
    mesh = trimesh.creation.box(extents=(2, 1, 2))
    binary = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
    document, buf = parse_glb(binary)
    scene = document["scenes"][0]
    scene["nodes"] = [len(document["nodes"])]
    matrix = np.diag([1, 2, 1, 1.])
    matrix[:3, 3] = [2, 3, 4]
    document["nodes"].append({"matrix": matrix.T.ravel().tolist(), "children": [0]})
    path = tmp_path / "mesh.glb"
    path.write_bytes(build_glb(document, buf))
    sidecar = path.with_suffix(".provenance.json")
    if with_sidecar:
        sidecar.write_text(json.dumps({"license": {"name": "MIT"}, "output": {"sha256": "old"}}))

    real_height.size_file(path, 4)

    result = trimesh.load(path, force="mesh", process=False)
    np.testing.assert_allclose(result.bounds, [[-2, 0, -2], [2, 4, 2]])
    np.testing.assert_allclose(np.linalg.norm(result.vertex_normals, axis=1), 1, atol=1e-6)
    out, _ = parse_glb(path.read_bytes())
    assert all("matrix" not in node for node in out["nodes"])
    assert not path.with_suffix(".sized.glb").exists()
    if with_sidecar:
        record = json.loads(sidecar.read_text())
        assert record["output"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["license"] == {"name": "MIT"}
    else:
        assert not sidecar.exists()


@pytest.mark.parametrize(("mode", "count"), [(4, 2), (5, 4), (6, 4), (1, 0)])
def test_glb_stats_counts_triangles_and_retains_external_image_type(mode, count):
    from test_roblox_export import _model

    from image_to_3dlab.glb_stats import glb_stats

    doc, data = parse_glb(_model())
    doc["meshes"][0]["primitives"][0]["mode"] = mode
    doc["images"] = [{"uri": "not-read.png", "mimeType": "image/png"}]
    result = glb_stats(build_glb(doc, data))
    assert result["triangles"] == count
    assert result["textures"] == [{"width": None, "height": None, "format": "png"}]


def test_glb_stats_refuses_a_truncated_header():
    from image_to_3dlab.glb_stats import glb_stats

    with pytest.raises(ValueError, match="not a GLB"):
        glb_stats(b"glTF")


def test_turn_refuses_invalid_glb_and_sparse_positions():
    import struct

    from image_to_3dlab.glb_turn import GLB_MAGIC, turn_glb

    with pytest.raises(ValueError, match="not a GLB file"):
        turn_glb(b"\0" * 12)
    with pytest.raises(ValueError, match="no JSON chunk"):
        turn_glb(struct.pack("<III", GLB_MAGIC, 2, 12))
    document, buf = mixed_model()
    document["accessors"][0]["sparse"] = {"count": 1}
    with pytest.raises(ValueError, match="not plain float data"):
        turn_glb(build_glb(document, buf))


def test_variant_export_requires_material_and_skips_unpainted_parts():
    from test_glb_skins import _glb

    from image_to_3dlab import glb_skins

    skin = _glb(b"paint", b"shine")
    doc, buf = parse_glb(skin)
    doc["meshes"][0]["primitives"].append({"attributes": {}})
    dressed, _ = parse_glb(glb_skins.with_variants(build_glb(doc, buf), [("Rust", skin)]))
    assert glb_skins.VARIANTS in dressed["meshes"][0]["primitives"][0]["extensions"]
    assert dressed["meshes"][0]["primitives"][-1] == {"attributes": {}}
    doc.pop("materials")
    with pytest.raises(ValueError, match="no material"):
        glb_skins.with_variants(build_glb(doc, buf), [("Rust", skin)])
