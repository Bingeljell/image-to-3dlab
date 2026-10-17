#!/usr/bin/env python3
"""Render a full frame sequence of a named Action at the locked dungeon-stage
camera angle (docs/design/bake-spec.md section 10 in the worklings repo).

Reusable across characters: every `.blend` built to the section-6 rig contract
already carries `CAM_TARGET` (the turntable empty) and an ortho camera parented
to it named `QuickCam2` (or `Camera`), so this script only ever swaps which
Action drives `rig` and which azimuth `CAM_TARGET` is turned to -- it never
re-authors the camera rig itself.

Talks to a running Blender over the execute_code RPC (see blender_pick_pixel.py
for the wire format) rather than running headless, so it renders whatever
`.blend` is currently open in that Blender instance. Switching files is the
caller's job (bpy.ops.wm.open_mainfile from another snippet, or just opening it
by hand) -- this script never opens or saves a file itself, so it can't clobber
unsaved work.
"""

from __future__ import annotations

import argparse
import json
import socket


def build_code(
    action_name: str,
    az: float,
    el: float,
    ortho_scale: float | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
    resolution: int,
    output_dir: str,
    label: str,
) -> str:
    return f'''
import bpy, os, math, json

arm = bpy.data.objects.get("rig")
if arm is None:
    raise RuntimeError("no object named 'rig' in the scene")
if arm.animation_data is None:
    raise RuntimeError("'rig' has no animation_data")

action = bpy.data.actions.get({action_name!r})
if action is None:
    raise RuntimeError("no action named {action_name!r} in this file")
arm.animation_data.action = action

target = bpy.data.objects.get("CAM_TARGET")
if target is None:
    raise RuntimeError("no CAM_TARGET empty in this file")
target.rotation_euler = (math.radians({el}), 0.0, math.radians({az}))

cam = bpy.data.objects.get("QuickCam2") or bpy.data.objects.get("Camera")
if cam is None or cam.data.type != "ORTHO":
    raise RuntimeError("no ortho camera parented to CAM_TARGET found (expected 'QuickCam2')")
{"cam.data.ortho_scale = " + repr(ortho_scale) if ortho_scale is not None else "# ortho_scale left as authored in the file"}
bpy.context.scene.camera = cam

scene = bpy.context.scene
scene.render.resolution_x = {resolution}
scene.render.resolution_y = {resolution}
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

out_dir = {str(output_dir)!r}
os.makedirs(out_dir, exist_ok=True)

native_start, native_end = int(action.frame_range[0]), int(action.frame_range[1])
start = {frame_start!r} if {frame_start!r} is not None else native_start
end = {frame_end!r} if {frame_end!r} is not None else native_end
step = {frame_step}

rendered = []
for frame in range(start, end + 1, step):
    scene.frame_set(frame)
    path = os.path.join(out_dir, f"{label}_{{action.name}}_az{az:g}_f{{frame:03d}}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    rendered.append(os.path.basename(path))

print("STAGE_BAKE " + json.dumps({{
    "label": {label!r}, "action": action.name, "az": {az}, "el": {el},
    "native_range": [native_start, native_end], "step": step,
    "frame_count": len(rendered), "frames": rendered,
}}))
'''


def send(code: str, host: str, port: int) -> str:
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(120)
        connection.sendall(json.dumps(request).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
    return b"".join(chunks).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", help="Action name already authored on the 'rig' armature")
    parser.add_argument("--label", required=True, help="Character label used in output filenames, e.g. tempest-ram")
    parser.add_argument("--az", type=float, required=True, help="CAM_TARGET Z rotation in degrees (35=stageFaceBL, 245=stageFaceTR)")
    parser.add_argument("--el", type=float, default=-28.0, help="CAM_TARGET X rotation in degrees (locked dungeon-stage value)")
    parser.add_argument("--ortho-scale", type=float, default=None, help="Override the camera's ortho scale; omit to use whatever is authored in the file")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1, help="1 = every native frame, 2 = half native fps, etc.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square render size; fast-preview default per the bake-spec pickup note")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()

    code = build_code(
        args.action, args.az, args.el, args.ortho_scale,
        args.frame_start, args.frame_end, args.frame_step,
        args.resolution, args.output_dir, args.label,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
