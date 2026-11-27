"""Check GLB pruning and command output where geometry shares a buffer with images."""

import io
import runpy
import struct
import sys

import pytest
from PIL import Image

from scripts import compress_glb_textures as compress


def test_prune_preserves_sparse_accessors_and_extension_images():
    document = {
        "materials": [
            {"normalTexture": {"index": 0}, "extensions": {
                "example": {"layers": [{"layerTexture": {"index": 1}}]},
            }},
        ],
        "textures": [
            {"source": 0, "sampler": 0},
            {"extensions": {"EXT_texture_webp": {"source": 1}}, "sampler": 1},
        ],
        "samplers": [{"wrapS": 10497}, {"wrapS": 33071}],
        "images": [{"bufferView": 0}, {"bufferView": 2}, {"uri": "unused.png"}],
        "bufferViews": [{"byteOffset": i * 4, "byteLength": 4} for i in range(4)],
        "accessors": [{"sparse": {
            "count": 1, "indices": {"bufferView": 1}, "values": {"bufferView": 3},
        }}],
    }
    assert compress.normal_texture_images(document) == {0}
    compress.strip_surface_maps(document, roughness=0.8)
    assert compress.prune_unused(document) == 2
    texture = document["textures"][0]
    assert texture["extensions"]["EXT_texture_webp"]["source"] == texture["sampler"] == 0
    assert document["samplers"] == [{"wrapS": 33071}]
    assert document["images"] == [{"bufferView": 1}]
    assert [view["byteOffset"] for view in document["bufferViews"]] == [4, 8, 12]
    sparse = document["accessors"][0]["sparse"]
    assert sparse["indices"]["bufferView"] == 0 and sparse["values"]["bufferView"] == 2
    assert document["materials"][0]["extensions"]["example"]["layers"][0]["layerTexture"]["index"] == 0


def test_view_pruning_leaves_a_view_shared_by_a_sparse_accessor():
    document = {
        "bufferViews": [{"byteOffset": 0, "byteLength": 4}],
        "accessors": [{"sparse": {"values": {"bufferView": 0}}}],
    }
    assert compress.drop_views(document, {0}) == set()
    assert document["bufferViews"][0]["byteOffset"] == 0


def test_glb_requires_json_and_ignores_unknown_chunks():
    junk = struct.pack("<II4s", 4, 0x12345678, b"misc")
    header = b"glTF" + struct.pack("<II", 2, 12 + len(junk))
    with pytest.raises(ValueError, match="no JSON chunk"):
        compress.parse_glb(header + junk)
    minimal = {"asset": {"version": "2.0"}}
    valid = compress.build_glb(minimal, b"")
    payload = valid[12:] + junk
    combined = b"glTF" + struct.pack("<II", 2, len(payload) + 12) + payload
    assert compress.parse_glb(combined) == (minimal, b"")


@pytest.mark.parametrize("optimised", [True, False])
def test_compress_cli_reports_kept_or_resized_pixels(tmp_path, monkeypatch, capsys, optimised):
    size = (4, 4) if optimised else (32, 16)
    buffer = io.BytesIO()
    Image.new("RGBA", size, (40, 80, 120, 32)).save(
        buffer, "PNG", optimize=optimised, compress_level=9 if optimised else 0,
    )
    texture = buffer.getvalue()
    document = {
        "asset": {"version": "2.0"},
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "images": [{"bufferView": 0, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(texture)}],
        "buffers": [{"byteLength": len(texture)}],
    }
    source, out = tmp_path / "original.glb", tmp_path / "new" / "converted.glb"
    source.write_bytes(compress.build_glb(document, texture))
    args = ["compress_glb_textures.py", str(source), str(out), "--max-size", "8",
            "--map-size", "0", "--normal-size", "0", "--format", "jpeg"]
    if not optimised:
        args.extend(["--drop-surface-maps", "--roughness", "0.2"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as status:
        runpy.run_path(compress.__file__, run_name="__main__")
    assert status.value.code == 0
    printed = capsys.readouterr().out
    assert "original.glb:" in printed
    if optimised:
        assert "re-encoding was larger" in printed
        assert out.read_bytes() == source.read_bytes()
    else:
        assert "32x16" in printed and "surface maps dropped" in printed
        result, binary = compress.parse_glb(out.read_bytes())
        assert result["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.2
        view = result["bufferViews"][0]
        with Image.open(io.BytesIO(binary[:view["byteLength"]])) as decoded:
            assert decoded.size == (8, 4)
            assert decoded.getpixel((0, 0)) == pytest.approx((40, 80, 120, 32), abs=1)
