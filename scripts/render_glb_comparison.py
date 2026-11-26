#!/usr/bin/env python3
"""Render several GLBs from one fixed camera and lay them out as a comparison image.

For documentation: a claim like "these two settings produce the same asset" is far easier
to check against a picture than against a paragraph. This renders each GLB headlessly, so
it never touches a running Blender session, and composes the results side by side.

    python scripts/render_glb_comparison.py out.jpg \
        a.glb:"sdpa" b.glb:"MLX fp32" c.glb:"MLX fp16" [--azimuth 215] [--elevation 14]

Every panel uses the same camera, the same lights, and -- importantly -- the *same* crop
box, computed across all of them. Cropping each panel to its own subject would rescale them
independently and make identical assets look different, which is the exact failure the
image is meant to rule out.

Output is JPEG sized for a slim repository; see docs/images/README.md.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from . import blender_render_glb_comparison_worker as _worker
else:
    import blender_render_glb_comparison_worker as _worker

bounding_sphere_radius = _worker.bounding_sphere_radius
camera_distance = _worker.camera_distance

BLENDER_DEFAULT = Path("/Applications/Blender.app/Contents/MacOS/Blender")
PANEL_W, PANEL_H = 1200, 820
BLENDER_WORKER = Path(__file__).resolve().with_name("blender_render_glb_comparison_worker.py")


def shared_crop_box(boxes, size, pad: int = 18):
    """One crop box covering every panel's subject, clamped to the frame.

    Shared on purpose. Per-panel crops would scale each subject differently and
    manufacture visual differences between assets that are actually identical.
    """
    if not boxes:
        raise ValueError("need at least one subject box")
    width, height = size
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return (
        max(0, left - pad), max(0, top - pad),
        min(width, right + pad), min(height, bottom + pad),
    )


def parse_asset(spec: str) -> tuple[Path, str]:
    """``path.glb:Label`` -> (path, label). The label defaults to the file stem."""
    if ":" in spec:
        head, _, label = spec.rpartition(":")
        if head and not head.endswith(".glb") and Path(spec).exists():
            return Path(spec), Path(spec).stem
        return Path(head), label
    return Path(spec), Path(spec).stem


def blender_render_code() -> str:
    """Return the tracked worker source for callers that need the script text."""
    return BLENDER_WORKER.read_text(encoding="utf-8")


def compose(panels, output: Path, width: int, quality: int) -> Path:
    from PIL import Image, ImageChops, ImageDraw

    loaded = [(label, Image.open(path).convert("RGB")) for label, path in panels]

    def subject_box(im):
        bg = Image.new("RGB", im.size, im.getpixel((2, 2)))
        mask = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
        return mask.getbbox() or (0, 0, *im.size)

    box = shared_crop_box([subject_box(im) for _, im in loaded], loaded[0][1].size)
    crops = [(label, im.crop(box)) for label, im in loaded]

    w, h = crops[0][1].size
    bar, gap = 44, 6
    sheet = Image.new("RGB", (w * len(crops) + gap * (len(crops) - 1), h + bar), (18, 18, 21))
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(crops):
        x = i * (w + gap)
        sheet.paste(im, (x, bar))
        draw.text((x + 14, 14), label, fill=(233, 233, 237))

    sheet = sheet.resize((width, int(sheet.height * width / sheet.width)), Image.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=quality, optimize=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="JPEG to write")
    parser.add_argument("assets", nargs="+", help="each as path.glb or path.glb:Label")
    parser.add_argument("--azimuth", type=float, default=215.0)
    parser.add_argument("--elevation", type=float, default=14.0)
    parser.add_argument("--width", type=int, default=1800, help="final image width")
    parser.add_argument("--quality", type=int, default=76)
    parser.add_argument("--clay", action="store_true",
                        help="strip materials and render neutral clay, so the comparison "
                             "is about shape rather than paint")
    parser.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    args = parser.parse_args(argv)

    blender = args.blender if args.blender.exists() else Path(shutil.which("blender") or "")
    if not blender or not blender.exists():
        raise SystemExit(f"Blender not found at {args.blender}; pass --blender")

    parsed = [parse_asset(spec) for spec in args.assets]
    missing = [str(p) for p, _ in parsed if not p.is_file()]
    if missing:
        raise SystemExit(f"missing GLB(s): {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panels = []
        for index, (path, label) in enumerate(parsed):
            png = tmp / f"panel_{index}.png"
            # check=False on purpose: Blender can exit non-zero having still written a
            # usable frame, and can exit zero having written nothing. The file is the
            # only honest success signal, so that is what is tested below.
            result = subprocess.run(
                [str(blender), "--background", "--python", str(BLENDER_WORKER), "--",
                 str(path), str(png), str(args.azimuth), str(args.elevation),
                 str(PANEL_W), str(PANEL_H), "1" if args.clay else "0"],
                capture_output=True, text=True, check=False,
            )
            if not png.is_file():
                sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
                raise SystemExit(f"render failed for {path}")
            panels.append((label, png))
            print(f"rendered {label}", flush=True)
        out = compose(panels, args.output, args.width, args.quality)

    print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
