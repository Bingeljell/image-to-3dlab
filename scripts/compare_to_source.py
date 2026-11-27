#!/usr/bin/env python3
"""Put a generated asset next to the image it was generated from.

Every metric in this repo measures the mesh against itself -- tear fraction, thickness,
ribbon width, duplicate counts. None of them can see the thing we actually care about:
how close the result looks to the source image. Two days went into optimising a number
that was blind to a dead texture and to markings baked in as grooves, because nothing
compared the output to the input.

This is that comparison. It renders the asset from a camera aimed to match the source's
viewing angle, crops both images to the subject, scales them to a common height, and
lays them side by side -- so the eye judges, which is the only judge that has been
reliable here.

Three panels matter:

* **source** -- the input image, cropped to the subject.
* **textured** -- the asset as it ships, same angle.
* **grey** -- the asset with every texture stripped. Anything still visible here is
  *shape*, not paint. This is what proves markings became geometry: a stripe that
  survives with the texture removed was carved into the mesh.

A fourth panel, ``silhouette``, overlays the two outlines so proportion errors show up
directly rather than having to be eyeballed across a gap.

Camera angles use ``azimuth``/``elevation`` in degrees, measured from the asset's front:

* azimuth 0 places the camera in front of the asset (-X, matching ``front_xneg``)
* positive azimuth swings the camera toward -Y, so +27 is the familiar 3/4 front view
* elevation 0 is level with the asset's middle, +90 is directly overhead

Nothing here is asset-specific. The angle that matches a given source is found once, by
eye, with ``--sweep``, and then recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

if __package__:
    from . import blender_compare_source_worker as render_worker
else:
    import blender_compare_source_worker as render_worker

ENVIRONMENTS = render_worker.ENVIRONMENTS
orbit_position = render_worker.orbit_position

# Pillow is imported lazily by the compositing helpers so the geometry functions -- and
# their tests -- run without it.

# Colours for the silhouette overlay, chosen to stay distinguishable for the most common
# forms of colour blindness: magenta reads where only the source covers, green where only
# the render covers, near-white where they agree.
SOURCE_ONLY = (214, 39, 168)
RENDER_ONLY = (46, 168, 82)
BOTH = (238, 238, 238)
NEITHER = (24, 24, 27)

def subject_bbox(image, threshold: int = 8):
    """Bounding box of the subject, or None when the image is empty.

    Both the source cut-outs and our renders carry alpha (the renders because we set
    ``film_transparent``), so the subject is simply the non-transparent region. Falling
    back to a luminance threshold would pick up the world background and defeat the
    crop, so an image without alpha is treated as fully opaque and returns its full
    extent -- callers that care should pass cut-outs.
    """
    if "A" not in image.getbands():
        return (0, 0, image.width, image.height)
    alpha = image.getchannel("A")
    return alpha.point(lambda value: 255 if value > threshold else 0).getbbox()


def crop_to_subject(image, pad_frac: float = 0.02):
    """Crop to the subject with a small margin, so panels are framed comparably.

    Without this the source (subject filling the frame) and the render (subject at
    whatever size the camera happened to give) are compared at different scales, and
    every silhouette difference is drowned by that scale difference.
    """
    box = subject_bbox(image)
    if box is None:
        return image
    left, upper, right, lower = box
    pad = round(max(right - left, lower - upper) * pad_frac)
    return image.crop(
        (
            max(left - pad, 0),
            max(upper - pad, 0),
            min(right + pad, image.width),
            min(lower + pad, image.height),
        )
    )


def scale_to_height(image, height: int):
    """Resize preserving aspect ratio. Height is the common axis across panels."""
    if image.height == height:
        return image
    from PIL import Image

    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.LANCZOS)


def silhouette_overlay(source, render, height: int = 900):
    """Overlay the two outlines so proportion errors are visible directly.

    Magenta is source-only, green is render-only, white is agreement.

    **This compares proportion, not size.** Both subjects are cropped to their own
    outline and scaled to a common height before being overlaid, because the camera
    distance is arbitrary -- an asset rendered slightly closer is not "wrong". What
    survives that normalisation is the shape: a body that came out too narrow leaves
    magenta down both flanks, ears too stubby leave magenta at the tips, a snout that
    ballooned leaves green around it.

    The corollary is that a uniformly scaled copy of the source registers as a perfect
    match, which is correct here and would not be if we were checking real-world size.
    """
    import numpy as np
    from PIL import Image

    source_mask = _subject_mask(crop_to_subject(source), height)
    render_mask = _subject_mask(crop_to_subject(render), height)

    width = max(source_mask.width, render_mask.width)
    in_source = np.array(_center_on(source_mask, width, height)) > 127
    in_render = np.array(_center_on(render_mask, width, height)) > 127

    overlay = np.empty((height, width, 3), dtype=np.uint8)
    overlay[...] = NEITHER
    overlay[in_render] = RENDER_ONLY
    overlay[in_source] = SOURCE_ONLY
    overlay[in_source & in_render] = BOTH
    return Image.fromarray(overlay)


def _subject_mask(image, height: int):
    """Binary alpha mask, scaled to the common height."""
    from PIL import Image

    if "A" in image.getbands():
        mask = image.getchannel("A")
    else:
        mask = Image.new("L", image.size, 255)
    return scale_to_height(mask, height).point(lambda value: 255 if value > 127 else 0)


def _center_on(mask, width: int, height: int):
    """Pad a mask into a common canvas, centred horizontally."""
    from PIL import Image

    canvas = Image.new("L", (width, height), 0)
    canvas.paste(mask, ((width - mask.width) // 2, 0))
    return canvas


def contact_sheet(
    panels: list[tuple[object, str]],
    height: int = 900,
    gap: int = 16,
    label_height: int = 34,
    background: tuple[int, int, int] = (24, 24, 27),
):
    """Lay panels out in a row, each cropped to its subject and labelled.

    Panels are (image, label) pairs. They are scaled to a common height so that a
    silhouette held against its source is directly comparable rather than merely
    adjacent.
    """
    from PIL import Image, ImageDraw

    prepared = [(scale_to_height(crop_to_subject(image), height), label) for image, label in panels]
    total_width = sum(image.width for image, _ in prepared) + gap * (len(prepared) + 1)
    sheet = Image.new("RGB", (total_width, height + label_height + gap * 2), background)
    draw = ImageDraw.Draw(sheet)
    font = _label_font(label_height)

    x = gap
    for image, label in prepared:
        if image.mode == "RGBA":
            sheet.paste(image, (x, label_height + gap), image)
        else:
            sheet.paste(image, (x, label_height + gap))
        draw.text((x, gap // 2), label, fill=(235, 235, 235), font=font)
        x += image.width + gap
    return sheet


def _label_font(label_height: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=int(label_height * 0.62))
    except TypeError:  # Pillow < 10.1 takes no size
        return ImageFont.load_default()


def headless_argv(request: dict) -> list[str]:
    """Run the tracked worker in a separate Blender process."""
    blender = os.environ.get("BLENDER", "blender")
    worker = Path(__file__).resolve().with_name("blender_compare_source_worker.py")
    return [
        blender, "--background", "--python-exit-code", "1", "--python", str(worker), "--",
        json.dumps(request),
    ]


def render_headless(
    asset: Path,
    output_dir: Path,
    label: str,
    angles: list[float],
    elevation: float,
    env: str,
    grey: bool,
) -> str:
    """Render one asset and return the Blender process output."""
    request = {
        "asset_path": str(asset),
        "output_dir": str(output_dir),
        "label": label,
        "angles": angles,
        "elevation_deg": elevation,
        "env": env,
        "grey_mode": grey,
    }
    done = subprocess.run(headless_argv(request), capture_output=True, text=True, check=False)
    output = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        raise SystemExit(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the input image the asset was generated from")
    parser.add_argument("asset", type=Path, help="the generated .glb")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="cmp")
    parser.add_argument(
        "--azimuth",
        type=float,
        default=27.0,
        help="degrees around the asset; 0 is front, +27 is the usual 3/4 front view",
    )
    parser.add_argument("--elevation", type=float, default=8.0)
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="comma-separated azimuths to render as a contact sheet, for finding the "
        "angle that matches the source. Skips the textured/grey/silhouette panels.",
    )
    parser.add_argument("--env", choices=tuple(ENVIRONMENTS), default="studio")
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    from PIL import Image

    source_path = args.source.expanduser().resolve()
    asset = args.asset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(source_path).convert("RGBA")

    if args.sweep:
        angles = [float(value) for value in args.sweep.split(",")]
        print(render_headless(
            asset, output_dir, args.label, angles, args.elevation, args.env, grey=False,
        ))
        panels = [(source, "source")]
        for azimuth in angles:
            path = output_dir / f"{args.label}_tex_az{round(azimuth)}.png"
            panels.append((Image.open(path).convert("RGBA"), f"az {azimuth:g}"))
        sheet = contact_sheet(panels, height=args.height)
        out = output_dir / f"{args.label}_sweep.png"
        sheet.save(out)
        print(f"wrote {out}")
        return 0

    for grey in (False, True):
        print(render_headless(
            asset, output_dir, args.label, [args.azimuth], args.elevation, args.env, grey=grey,
        ))

    tag = round(args.azimuth)
    textured = Image.open(output_dir / f"{args.label}_tex_az{tag}.png").convert("RGBA")
    grey_render = Image.open(output_dir / f"{args.label}_grey_az{tag}.png").convert("RGBA")

    sheet = contact_sheet(
        [
            (source, "source"),
            (textured, "generated (textured)"),
            (grey_render, "generated (grey - shape only)"),
            (silhouette_overlay(source, textured, args.height), "silhouette: magenta=source green=ours"),
        ],
        height=args.height,
    )
    out = output_dir / f"{args.label}_vs_source.png"
    sheet.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
