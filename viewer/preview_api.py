"""A still picture of an asset from four sides, for anyone who can't spin it in the viewer.

An agent driving the lab through MCP has no 3D viewer: it can only look at pictures. So
this renders a model from the front, three-quarter, side and back into one labelled sheet,
headless in Blender (it never touches a running Blender session), and keeps it in
`output/.previews` so asking twice costs nothing. A picture-stage asset is its own preview.

The maths and the sheet layout are plain functions so tests can import them; only the
thin `bpy` calls live in the string handed to Blender.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PREVIEWS_DIR = ".previews"
VIEWS = (("Front", 0.0), ("Three-quarter", 45.0), ("Side", 90.0), ("Back", 180.0))
MOMENTS = (0.1, 0.37, 0.63, 0.9)   # where in a move its four panels fall (start and end are often the same pose)
MOVE_AZIMUTH = 35.0
PANEL_W, PANEL_H = 512, 640
LABEL_PX = 40          # big enough to read in a chat window (labels at default size are unreadable)
SHEET_WIDTH = 1600
RENDER_TIMEOUT = 300
# The stages a preview can show, and the asset field holding each one's file.
STAGE_FIELDS = {"picture": "picture", "model": "model", "finished": "finished", "rigged": "rigged", "animated": "rigged"}


def front_azimuth(facing: list[float] | None) -> float:
    """Degrees to turn the camera so "Front" faces the model's front.

    `facing` is the asset's forward direction in glTF axes (+Z is forward). Blender imports
    glTF +Z as -Y, and the camera at azimuth a sits at (sin a, -cos a), so a = atan2(x, z).
    """
    if not facing or len(facing) < 3 or (facing[0] == 0 and facing[2] == 0):
        return 0.0
    return math.degrees(math.atan2(facing[0], facing[2]))


def stage_file(asset: dict[str, Any], stage: str = "latest") -> tuple[str, str] | None:
    """(stage, relative file) to preview, or None when the asset has nothing for that stage.

    "latest" is the furthest step the asset has reached. An animated asset previews its
    rigged model: a still can't show a move, so its rest pose is the honest picture.
    """
    if stage == "latest":
        for name in ("rigged", "finished", "model", "picture"):
            if asset.get(name):
                return ("animated" if name == "rigged" and asset.get("clips") else name), asset[name]
        return None
    field = STAGE_FIELDS.get(stage)
    if field is None:
        raise ValueError(f"unknown stage {stage!r}; pick one of latest, {', '.join(STAGE_FIELDS)}")
    return (stage, asset[field]) if asset.get(field) else None


def cache_key(source: Path, azimuth: float, look: str) -> str:
    """Changes whenever the file, the camera or the look changes, and not otherwise."""
    stat = source.stat()
    return hashlib.sha1(f"{source}|{stat.st_size}|{stat.st_mtime}|{azimuth:.1f}|{look}|{VIEWS}".encode()).hexdigest()[:20]


def label_font(size: int = LABEL_PX):
    from PIL import ImageFont

    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def compose_sheet(panels: list[tuple[str, Path]], output: Path, title: str = "") -> Path:
    """Lay renders side by side under big labels, all cropped to one shared box.

    One box for every panel, never one each: separate crops would scale each view
    differently and make the same model look like it changes size as it turns.
    """
    from PIL import Image, ImageChops, ImageDraw

    images = [(label, Image.open(path).convert("RGB")) for label, path in panels]
    boxes = []
    for _, image in images:
        background = Image.new("RGB", image.size, image.getpixel((2, 2)))
        mask = ImageChops.difference(image, background).convert("L").point(lambda v: 255 if v > 12 else 0)
        boxes.append(mask.getbbox() or (0, 0, *image.size))
    pad = 16
    w0, h0 = images[0][1].size
    box = (max(0, min(b[0] for b in boxes) - pad), max(0, min(b[1] for b in boxes) - pad),
           min(w0, max(b[2] for b in boxes) + pad), min(h0, max(b[3] for b in boxes) + pad))
    crops = [(label, image.crop(box)) for label, image in images]

    w, h = crops[0][1].size
    font = label_font()
    bar, gap = LABEL_PX + 28, 6
    top = bar if title else 0
    sheet = Image.new("RGB", (w * len(crops) + gap * (len(crops) - 1), h + bar + top), (18, 18, 21))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((16, 12), title, fill=(255, 120, 40), font=font)
    for i, (label, image) in enumerate(crops):
        x = i * (w + gap)
        sheet.paste(image, (x, bar + top))
        draw.text((x + 16, top + 12), label, fill=(236, 236, 240), font=font)
    if sheet.width > SHEET_WIDTH:
        sheet = sheet.resize((SHEET_WIDTH, round(sheet.height * SHEET_WIDTH / sheet.width)), Image.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part.png")
    sheet.save(partial, "PNG", optimize=True)
    partial.replace(output)  # never leave half a sheet behind for the next request
    return output


def picture_preview(source: Path, target: Path, back: Path | None = None) -> Path:
    """A picture-stage asset: the picture itself (front and back side by side when there are two)."""
    from PIL import Image

    pictures = [Image.open(p).convert("RGB") for p in ([source, back] if back else [source])]
    height = min(1024, max(p.height for p in pictures))
    scaled = [p.resize((round(p.width * height / p.height), height), Image.LANCZOS) for p in pictures]
    sheet = Image.new("RGB", (sum(p.width for p in scaled) + 6 * (len(scaled) - 1), height), (18, 18, 21))
    x = 0
    for p in scaled:
        sheet.paste(p, (x, 0))
        x += p.width + 6
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "PNG", optimize=True)
    return target


def shots_for(move: bool) -> list[tuple[str, float, float | None]]:
    """(label, camera turn, moment) per panel. A model turns round; a move holds the camera
    at three-quarter and samples four moments, since a still of one pose can't show a move."""
    if move:
        return [(f"m{i}", MOVE_AZIMUTH, fraction) for i, fraction in enumerate(MOMENTS)]
    return [(name, az, None) for name, az in VIEWS]


def shots_arg(shots: list[tuple[str, float, float | None]]) -> str:
    return ";".join(f"{name}={az}={'' if frac is None else frac}" for name, az, frac in shots)


def moment_label(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds:.1f} s"


def blender_code() -> str:
    """The script handed to Blender: import once, render every shot. Kept thin on purpose."""
    return """
import sys, math, json, bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_dir, front, w, h, clay = argv[0], argv[1], float(argv[2]), int(argv[3]), int(argv[4]), argv[5] == "1"
shots = []
for item in argv[6].split(";"):
    name, az, frac = item.split("=")
    shots.append((name, float(az), float(frac) if frac else None))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
# only what shows: a file can carry hidden helper meshes that would wreck the framing
for o in scene.objects:
    if o.type == "MESH" and not o.visible_get():
        o.hide_render = True
meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
if not meshes:
    raise SystemExit("no mesh in " + src)
if clay:
    neutral = bpy.data.materials.new("clay")
    neutral.use_nodes = True
    neutral.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.62, 0.62, 0.60, 1.0)
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(neutral)

# a move's length is its actions' keys (an empty scene's own range is Blender's default 250 frames)
ranges = [a.frame_range for a in bpy.data.actions]
first, last = (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else (scene.frame_start, scene.frame_end)
fps = scene.render.fps / (scene.render.fps_base or 1)
frames = {name: (first + frac * (last - first) if frac is not None else None) for name, _, frac in shots}

def bounds_now(lo, hi):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        m = ev.matrix_world
        for v in mesh.vertices:
            p = m @ v.co
            for i in range(3):
                lo[i] = min(lo[i], p[i]); hi[i] = max(hi[i], p[i])
        ev.to_mesh_clear()

# one frame for every shot: the box round the model at every moment shown
lo, hi = [float("inf")] * 3, [float("-inf")] * 3
for f in sorted({f for f in frames.values() if f is not None}) or [None]:
    if f is not None:
        scene.frame_set(int(f), subframe=f - int(f))
    bounds_now(lo, hi)
centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
radius = math.sqrt(sum(((hi[i] - lo[i]) / 2) ** 2 for i in range(3))) or 1.0

world = bpy.data.worlds.new("W"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.055, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.resolution_percentage = 100
available = {i.identifier for i in type(scene.render).bl_rna.properties["engine"].enum_items}
scene.render.engine = next(e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES") if e in available)
scene.render.image_settings.file_format = "PNG"

cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
# the lens angle spans the frame's longer side; fit the subject to the shorter one
narrow = math.atan(math.tan(cam_data.angle / 2) * min(w, h) / max(w, h))
dist = radius / math.sin(narrow) * 1.05

rig = bpy.data.objects.new("Lights", None)
scene.collection.objects.link(rig)
rig.location = centre
for name, offset, energy in (("key", (1.6, -2.2, 1.8), 6.0), ("fill", (-2.4, -1.6, 0.6), 2.4), ("rim", (0.4, 2.8, 1.8), 3.0)):
    light = bpy.data.lights.new(name, type="AREA")
    light.energy = energy * radius ** 2 * 40
    light.size = radius * 2
    obj = bpy.data.objects.new(name, light)
    obj.parent = rig
    obj.location = tuple(offset[i] * radius for i in range(3))
    obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(obj)

times = {}
for name, az, frac in shots:
    if frames[name] is not None:
        f = frames[name]
        scene.frame_set(int(f), subframe=f - int(f))
        times[name] = round((f - first) / fps, 2)
    a, e = math.radians(front + az), math.radians(8)
    cam.location = (centre[0] + dist * math.cos(e) * math.sin(a), centre[1] - dist * math.cos(e) * math.cos(a),
                    centre[2] + dist * math.sin(e))
    cam.rotation_euler = (Vector(centre) - cam.location).to_track_quat("-Z", "Y").to_euler()
    rig.rotation_euler = (0, 0, a)  # lights turn with the camera, so the back is lit like the front
    scene.render.filepath = out_dir + "/" + name + ".png"
    bpy.ops.render.render(write_still=True)
with open(out_dir + "/times.json", "w") as handle:
    json.dump(times, handle)
"""


def render_shots(blender: Path, source: Path, out_dir: Path, azimuth: float, look: str,
                 shots: list[tuple[str, float, float | None]]) -> list[tuple[str, Path]]:
    """Render every shot in one Blender run; returns (label, PNG) per panel."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as script:
        script.write(blender_code())
    try:
        result = subprocess.run(
            [str(blender), "--background", "--factory-startup", "--python", script.name, "--",
             str(source), str(out_dir), str(azimuth), str(PANEL_W), str(PANEL_H), "1" if look == "clay" else "0",
             shots_arg(shots)],
            capture_output=True, text=True, timeout=RENDER_TIMEOUT, check=False)
    finally:
        Path(script.name).unlink(missing_ok=True)
    files = [(name, out_dir / f"{name}.png") for name, _, _ in shots]
    if result.returncode != 0 or not all(p.is_file() for _, p in files):
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-6:])
        raise RuntimeError(f"Blender could not render the preview: {tail}")
    try:
        times = json.loads((out_dir / "times.json").read_text())
    except (OSError, ValueError):
        times = {}
    return [(moment_label(times.get(name)) if frac is not None else name, path)
            for (name, _, frac), (_, path) in zip(shots, files)]


def find_move(asset: dict[str, Any], move: str) -> str:
    """The GLB of one of the asset's moves, by name."""
    clip = next((c for c in asset.get("clips", []) if c.get("name") == move), None)
    if clip is None:
        have = ", ".join(c.get("name", "") for c in asset.get("clips", [])) or "none"
        raise ValueError(f"{asset.get('name', 'this asset')} has no move {move!r} (it has: {have})")
    return clip["file"]


def preview(output: Path, asset: dict[str, Any], stage: str = "latest", look: str = "paint",
            blender: Path | None = None, move: str | None = None) -> tuple[Path, str]:
    """(PNG path, what is shown) for one asset, made once and kept in output/.previews.

    With `move`, four moments of that move instead of four sides of the model."""
    if look not in ("paint", "clay"):
        raise ValueError("look is paint or clay")
    if move:
        shown, relative = f"move: {move}", find_move(asset, move)
    else:
        picked = stage_file(asset, stage)
        if picked is None:
            raise ValueError(f"{asset.get('name', 'this asset')} has no {stage} yet")
        shown, relative = picked
    source = (output / relative).resolve()
    if output.resolve() not in source.parents or not source.is_file():
        raise ValueError(f"not a file in output/: {relative}")
    azimuth = front_azimuth(asset.get("facing"))
    target = output / PREVIEWS_DIR / f"{cache_key(source, azimuth, look + (':move' if move else ''))}.png"
    if target.is_file():
        return target, shown
    if shown == "picture":
        back = (output / asset["back"]).resolve() if asset.get("back") else None
        return picture_preview(source, target, back if back and back.is_file() else None), shown
    if blender is None:
        raise RuntimeError("3D previews need Blender. Install it from the studio's Setup page.")
    with tempfile.TemporaryDirectory() as tmp:
        panels = render_shots(blender, source, Path(tmp), azimuth, look, shots_for(bool(move)))
        title = f"{asset.get('name', '')} · {shown}" if asset.get("name") else ""
        return compose_sheet(panels, target, title=title), shown
