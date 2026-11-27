"""Keep a model's cameras and pictures inside its GLB, so the file alone carries them.

A Pixal3D build saves the cameras it saw the pictures from beside the model
(`<name>.svviews/`: transforms.json + one picture per view). Pixel Match needs them, but a
folder beside a file is easy to lose: share or upload the GLB and they stay behind.

`embed` stores them in the GLB: the pictures as lossless WebP (every pixel exactly as saved,
about a third smaller than the PNG) in the file's binary chunk, and the cameras in the
top-level `extras` under `KEY`. glTF readers ignore both, so the model opens as before.
`extract` writes them back out as a `.svviews` folder that `photo_paint.load_views` reads.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

KEY = "assetfurnace_views"
VERSION = 1


def _split(data: bytes) -> tuple[dict, bytes]:
    if data[:4] != b"glTF":
        raise ValueError("not a GLB file")
    json_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + json_len])
    offset = 20 + json_len
    binary = b""
    if offset < len(data):
        bin_len = struct.unpack_from("<I", data, offset)[0]
        binary = data[offset + 8:offset + 8 + bin_len]
    return doc, binary


def _join(doc: dict, binary: bytes) -> bytes:
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    out = struct.pack("<III", 0x46546C67, 2, length)
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    if binary:
        out += struct.pack("<II", len(binary), 0x004E4942) + binary
    return out


def webp(picture: Path) -> bytes:
    """A picture as lossless WebP. `exact` keeps the colour under see-through pixels too, so
    every pixel comes back as saved."""
    from PIL import Image

    with Image.open(picture) as image:
        out = io.BytesIO()
        image.convert("RGBA").save(out, "WEBP", lossless=True, exact=True, method=6)
    return out.getvalue()


def has_views(data: bytes) -> bool:
    return KEY in _split(data)[0].get("extras", {})


def embed(data: bytes, views_dir: Path) -> bytes:
    """`data` (a GLB) with the cameras and pictures of `views_dir` stored inside it.

    Frames whose picture is missing are left out. Embedding again replaces the cameras in the
    JSON; the old pictures' bytes stay in the binary chunk, so embed once, after the GLB is final.
    """
    views_dir = Path(views_dir)
    meta = json.loads((views_dir / "transforms.json").read_text())
    doc, binary = _split(data)
    if len(doc.get("buffers", [])) > 1 or any("uri" in b for b in doc.get("buffers", [])):
        raise ValueError("only a single-buffer GLB can carry its views")
    if not doc.get("buffers"):
        doc["buffers"] = [{"byteLength": 0}]
    views = doc.setdefault("bufferViews", [])
    frames = []
    for frame in meta.get("frames", []):
        name = frame.get("file_path", "")
        picture = views_dir / name
        if Path(name).name != name or not picture.is_file():
            continue
        binary += b"\0" * (-len(binary) % 4)
        encoded = webp(picture)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(encoded)})
        binary += encoded
        frames.append(dict(frame, bufferView=len(views) - 1, mimeType="image/webp"))
    if not frames:
        raise ValueError(f"no pictures found for the cameras in {views_dir}")
    doc["buffers"][0]["byteLength"] = len(binary) + (-len(binary) % 4)
    doc.setdefault("extras", {})[KEY] = dict(meta, frames=frames, version=VERSION)
    return _join(doc, binary)


def extract(data: bytes, out_dir: Path) -> bool:
    """Write the views stored in `data` (a GLB) to `out_dir` as a .svviews folder.

    The pictures come out as WebP, named `view_000.webp` and so on. False if it has none.
    """
    doc, binary = _split(data)
    meta = doc.get("extras", {}).get(KEY)
    if not isinstance(meta, dict) or not isinstance(meta.get("frames"), list):
        return False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in meta["frames"]:
        view = doc["bufferViews"][frame["bufferView"]]
        start = view.get("byteOffset", 0)
        name = Path(Path(frame["file_path"]).name).with_suffix(".webp").name
        (out_dir / name).write_bytes(binary[start:start + view["byteLength"]])
        frames.append({k: v for k, v in frame.items() if k not in ("bufferView", "mimeType")} | {"file_path": name})
    rest = {k: v for k, v in meta.items() if k not in ("frames", "version")}
    (out_dir / "transforms.json").write_text(json.dumps(dict(rest, frames=frames), indent=2) + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Store a model's cameras in its GLB, or take them out.")
    parser.add_argument("action", choices=("embed", "extract"))
    parser.add_argument("glb", type=Path)
    parser.add_argument("views", type=Path, nargs="?", help="the .svviews folder (default: beside the GLB)")
    args = parser.parse_args(argv)
    views = args.views or args.glb.with_suffix(".svviews")
    if args.action == "embed":
        before = args.glb.stat().st_size
        args.glb.write_bytes(embed(args.glb.read_bytes(), views))
        print(f"cameras stored in {args.glb.name}: +{(args.glb.stat().st_size - before) / 1048576:.1f} MB")
    elif not extract(args.glb.read_bytes(), views):
        print(f"{args.glb.name} carries no cameras")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
