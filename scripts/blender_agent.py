#!/usr/bin/env python3
"""Let a local LLM stage a scene in the running Blender, by calling a small set of tools.

Run (live): python scripts/blender_agent.py --asset model.glb --task "stage and light this so the face reads" --out-dir renders

Needs a local OpenAI-compatible model server as well (--api, default http://127.0.0.1:1234/v1).

Why a tool schema instead of letting the model write bpy directly: the published
measurement (VIGA / BlenderBench, arXiv 2601.11109) is that small open models writing raw
bpy hallucinate operators, mis-spell parameters and emit non-executable loops, and that the
fix is not a bigger model but a high-level skill library split into *observation* (look,
change nothing) and *modification* (change something, persistently). The same 8B model that
scores 0.28 writing raw bpy scores 1.31 driving a library like this one. So the model never
sees bpy here. It sees eight verbs, and every line of Python that reaches Blender was
written by us.

The loop is interleaved: the model calls a tool, we run it in the live Blender, and when it
calls ``render_view`` the resulting PNG goes back into the conversation *as an image*. That
feedback is the other half of the measured gain, and it is why this needs a multimodal
model. ``gemma-4-12b-it-qat`` and ``gemma-4-31b-it-qat`` both qualify and are local.

    # with Blender running and its socket listening, and LM Studio serving a model:
    python scripts/blender_agent.py --asset output/foo/result.glb \
        --task "stage and light this so the face reads clearly" --model gemma-4-12b-it-qat

Every render, the full message transcript and every tool call are written to the run
directory, because the only honest way to judge this is to look at what it produced.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULT_SENTINEL = "AGENT_RESULT "
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
DEFAULT_API = "http://127.0.0.1:1234/v1"


# --------------------------------------------------------------------------------------
# The skill library.
#
# Each tool owns a JSON schema (what the model is allowed to ask for) and a builder (the
# bpy it turns into). Builders are pure functions from validated arguments to source text,
# which is what makes them testable without Blender in the room: `tests/test_blender_agent`
# compiles every one of them. Rule 2 in AGENTS.md, learned the expensive way.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    kind: str  # "observe" or "modify"
    description: str
    parameters: dict[str, Any]
    builder: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema, which is what LM Studio's API expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_NUM = {"type": "number"}
_STR = {"type": "string"}


def _preamble() -> str:
    return "import bpy, json, math, mathutils\n"


def _emit(expr: str) -> str:
    """Every snippet reports through one sentinel line, so the reply can be found in
    whatever else Blender decided to print to stdout that second."""
    return f'print({RESULT_SENTINEL!r} + json.dumps({expr}))\n'


def build_describe_scene() -> str:
    return _preamble() + '''
bpy.context.view_layer.update()
scene = bpy.context.scene
objects = []
for obj in bpy.data.objects:
    if obj.type not in {"MESH", "ARMATURE", "LIGHT", "CAMERA", "EMPTY"}:
        continue
    entry = {"name": obj.name, "type": obj.type,
             "location": [round(v, 4) for v in obj.matrix_world.translation]}
    if obj.type == "MESH":
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        lo = [round(min(c[i] for c in corners), 4) for i in range(3)]
        hi = [round(max(c[i] for c in corners), 4) for i in range(3)]
        entry["bounds_min"] = lo
        entry["bounds_max"] = hi
        entry["dimensions"] = [round(hi[i] - lo[i], 4) for i in range(3)]
        entry["polygons"] = len(obj.data.polygons)
    if obj.type == "LIGHT":
        entry["light_type"] = obj.data.type
        entry["energy"] = round(obj.data.energy, 3)
    if obj.type == "CAMERA":
        entry["lens_mm"] = round(obj.data.lens, 2)
    objects.append(entry)
world = scene.world
payload = {
    "objects": objects,
    "active_camera": scene.camera.name if scene.camera else None,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "film_transparent": bool(scene.render.film_transparent),
    "engine": scene.render.engine,
    "world_strength": (
        round(world.node_tree.nodes["Background"].inputs[1].default_value, 3)
        if world and world.use_nodes and "Background" in world.node_tree.nodes
        else None
    ),
}
''' + _emit("payload")


def build_render_view(output_path: str, resolution: int = 640, samples: int = 32) -> str:
    return _preamble() + f'''
scene = bpy.context.scene
scene.render.resolution_x = {int(resolution)}
scene.render.resolution_y = {int(resolution)}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
if scene.render.engine == "CYCLES":
    scene.cycles.samples = {int(samples)}
if scene.camera is None:
    payload = {{"error": "no active camera; call set_camera first"}}
else:
    scene.render.filepath = {output_path!r}
    bpy.ops.render.render(write_still=True)
    payload = {{"rendered": {output_path!r},
                "resolution": [scene.render.resolution_x, scene.render.resolution_y]}}
''' + _emit("payload")


def build_import_asset(path: str, clear_scene: bool = True) -> str:
    return _preamble() + f'''
clear = {bool(clear_scene)!r}
if clear:
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
before = set(bpy.data.objects.keys())
bpy.ops.import_scene.gltf(filepath={path!r})
imported = [n for n in bpy.data.objects.keys() if n not in before]
meshes = [n for n in imported if bpy.data.objects[n].type == "MESH"]
payload = {{"imported": imported, "meshes": meshes}}
''' + _emit("payload")


def build_set_camera(
    azimuth: float,
    elevation: float,
    distance_factor: float = 2.2,
    lens_mm: float = 50.0,
    target: str | None = None,
) -> str:
    """Orbit placement, because absolute coordinates are the thing models get wrong.

    `azimuth` 0 is the subject's front (minimum Y in Blender space, matching the glTF
    importer's Y-up to Z-up conversion this repo assumes everywhere). `distance_factor`
    multiplies the target's bounding-sphere radius, so the framing survives a change of
    asset scale.
    """
    return _preamble() + f'''
import math
target_name = {target!r}
candidates = [o for o in bpy.data.objects if o.type == "MESH"]
if target_name:
    candidates = [o for o in candidates if o.name == target_name]
else:
    # Ground planes are not the subject. Framing at 3x the radius of everything in the
    # scene put the camera 21 units back from a 2-unit sphere, because a 10-unit floor
    # dominated the bounds, and the model's reasonable "3x the subject" became a wide
    # shot of a floor. Anything effectively flat is treated as scenery unless it is named
    # explicitly, and it is only dropped when something else remains to look at.
    def _flat(obj):
        dims = sorted(obj.dimensions)
        return dims[2] > 1e-6 and dims[0] / dims[2] < 0.05
    subjects = [o for o in candidates if not _flat(o)]
    if subjects:
        candidates = subjects
if not candidates:
    payload = {{"error": "no mesh to aim at" + (" named " + repr(target_name) if target_name else "")}}
else:
    corners = []
    for obj in candidates:
        corners += [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    lo = mathutils.Vector([min(c[i] for c in corners) for i in range(3)])
    hi = mathutils.Vector([max(c[i] for c in corners) for i in range(3)])
    centre = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    az = math.radians({float(azimuth)})
    el = math.radians({float(elevation)})
    dist = radius * {float(distance_factor)}
    offset = mathutils.Vector((
        math.sin(az) * math.cos(el),
        -math.cos(az) * math.cos(el),
        math.sin(el),
    )) * dist
    cam = bpy.data.objects.get("AGENT_CAM")
    if cam is None:
        cam = bpy.data.objects.new("AGENT_CAM", bpy.data.cameras.new("AGENT_CAM"))
        bpy.context.scene.collection.objects.link(cam)
    cam.data.lens = {float(lens_mm)}
    cam.location = centre + offset
    direction = centre - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    payload = {{"camera": cam.name,
                "location": [round(v, 4) for v in cam.location],
                "aimed_at": [round(v, 4) for v in centre],
                "distance": round(dist, 4), "lens_mm": cam.data.lens}}
''' + _emit("payload")


def build_set_light(
    role: str,
    azimuth: float,
    elevation: float,
    energy: float,
    size: float = 1.0,
    color: list[float] | None = None,
) -> str:
    """Lights orbit the subject on the same convention as the camera, and are named by
    role so the model can move one without disturbing the others."""
    rgb = list(color or [1.0, 1.0, 1.0])
    return _preamble() + f'''
import math
role = {role!r}
name = "AGENT_LIGHT_" + role.upper()
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    payload = {{"error": "no mesh in the scene to light"}}
else:
    corners = []
    for obj in meshes:
        corners += [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    lo = mathutils.Vector([min(c[i] for c in corners) for i in range(3)])
    hi = mathutils.Vector([max(c[i] for c in corners) for i in range(3)])
    centre = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    az = math.radians({float(azimuth)})
    el = math.radians({float(elevation)})
    offset = mathutils.Vector((
        math.sin(az) * math.cos(el),
        -math.cos(az) * math.cos(el),
        math.sin(el),
    )) * radius * 3.0
    light = bpy.data.objects.get(name)
    if light is None:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
        bpy.context.scene.collection.objects.link(light)
    light.data.type = "AREA"
    light.data.size = {float(size)} * radius
    light.data.energy = {float(energy)}
    light.data.color = tuple({rgb!r})[:3]
    light.location = centre + offset
    direction = centre - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    payload = {{"light": name, "energy": light.data.energy,
                "location": [round(v, 4) for v in light.location]}}
''' + _emit("payload")


def build_set_world(strength: float, color: list[float] | None = None) -> str:
    rgb = list(color or [0.05, 0.05, 0.06])
    return _preamble() + f'''
scene = bpy.context.scene
world = scene.world
if world is None:
    world = bpy.data.worlds.new("AGENT_WORLD")
    scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg is None:
    payload = {{"error": "world has no Background node"}}
else:
    rgb = tuple({rgb!r})[:3]
    bg.inputs[0].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bg.inputs[1].default_value = {float(strength)}
    payload = {{"world_strength": {float(strength)}, "world_color": list(rgb)}}
''' + _emit("payload")


def build_move_object(name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    return _preamble() + f'''
obj = bpy.data.objects.get({name!r})
if obj is None:
    payload = {{"error": "no object named " + {name!r}}}
else:
    obj.location = (obj.location.x + {float(dx)},
                    obj.location.y + {float(dy)},
                    obj.location.z + {float(dz)})
    payload = {{"moved": obj.name, "location": [round(v, 4) for v in obj.location]}}
''' + _emit("payload")


def build_set_render_settings(
    engine: str = "eevee", film_transparent: bool = False, exposure: float = 0.0
) -> str:
    """Engine names are resolved against the running Blender, not hardcoded.

    Blender renamed the realtime engine between versions: 4.2+ calls it
    BLENDER_EEVEE_NEXT, 5.2 calls it BLENDER_EEVEE. Hardcoding either one breaks on the
    other, which is how this was found: the generated code crashed with
    `enum "BLENDER_EEVEE_NEXT" not found` against Blender 5.2. So the model names an
    engine loosely and the snippet matches it against whatever this build actually offers.
    """
    return _preamble() + f'''
scene = bpy.context.scene
wanted = {engine!r}.strip().upper()
available = [item.identifier for item in
             bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
match = None
for identifier in available:
    if identifier == wanted or wanted in identifier or identifier.replace("BLENDER_", "") == wanted:
        match = identifier
        break
if match is None:
    payload = {{"error": "no engine matching " + wanted + "; this Blender has " + ", ".join(available)}}
else:
    scene.render.engine = match
    scene.render.film_transparent = {bool(film_transparent)!r}
    scene.view_settings.exposure = {float(exposure)}
    payload = {{"engine": scene.render.engine,
                "film_transparent": bool(scene.render.film_transparent),
                "exposure": round(scene.view_settings.exposure, 3),
                "available": available}}
''' + _emit("payload")

def build_clear_scene(keep_lights: bool = False) -> str:
    """Start from nothing.

    Added after a live run where the model, asked to clear the scene and given no verb for
    it, improvised `import_asset("empty.glb")` and then fell back to deleting objects by
    guessed names. A missing verb does not stop a model; it makes one up.
    """
    return _preamble() + f'''
keep_lights = {bool(keep_lights)!r}
removable = {{"MESH"}}
if not keep_lights:
    removable = removable | {{"LIGHT"}}
removed = []
for obj in list(bpy.data.objects):
    if obj.type in removable:
        removed.append(obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
payload = {{"removed": removed, "count": len(removed)}}
''' + _emit("payload")


def build_add_primitive(
    kind: str,
    name: str,
    location: list[float] | None = None,
    size: float = 1.0,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> str:
    """Primitives are how anything gets built from nothing. The name is the model's
    handle on the result, so it is applied explicitly rather than left as Blender's
    "Cube.003".

    `scale` is here rather than only on set_object_transform because almost nothing worth
    building is made of cubes: a torso is a flattened box, a limb is a stretched one. With
    creation and shaping split across two calls, and a forced render between rounds, a
    fifteen-part figure cannot fit in any sane round budget.
    """
    loc = list(location or [0.0, 0.0, 0.0])[:3] + [0.0, 0.0, 0.0]
    rot = list(rotation or [0.0, 0.0, 0.0])[:3] + [0.0, 0.0, 0.0]
    scl = list(scale or [1.0, 1.0, 1.0])[:3] + [1.0, 1.0, 1.0]
    return _preamble() + f'''
import math
adders = {{
    "cube": lambda: bpy.ops.mesh.primitive_cube_add(size={float(size)}),
    "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius={float(size)} / 2.0),
    "cylinder": lambda: bpy.ops.mesh.primitive_cylinder_add(radius={float(size)} / 2.0, depth={float(size)}),
    "cone": lambda: bpy.ops.mesh.primitive_cone_add(radius1={float(size)} / 2.0, depth={float(size)}),
    "torus": lambda: bpy.ops.mesh.primitive_torus_add(major_radius={float(size)} / 2.0, minor_radius={float(size)} / 6.0),
    "plane": lambda: bpy.ops.mesh.primitive_plane_add(size={float(size)}),
}}
kind = {kind!r}
if kind not in adders:
    payload = {{"error": "unknown primitive " + kind + "; have " + ", ".join(sorted(adders))}}
elif {name!r} in bpy.data.objects:
    payload = {{"error": "an object named " + {name!r} + " already exists; pick another name"}}
else:
    adders[kind]()
    obj = bpy.context.active_object
    obj.name = {name!r}
    obj.location = tuple({loc[:3]!r})
    obj.rotation_euler = tuple(math.radians(a) for a in {rot[:3]!r})
    obj.scale = tuple({scl[:3]!r})
    # Dimensions derive from the object's matrix and do not refresh until the dependency
    # graph does. Without this the payload reported the UNSCALED size, so the model was
    # told its 0.02-wide arm was a 0.2 cube and had no way to notice the figure it was
    # building had limbs like threads. Reporting a stale number is worse than reporting
    # none: it is a confident lie the model then reasons from.
    bpy.context.view_layer.update()
    payload = {{"created": obj.name, "kind": kind,
                "location": [round(v, 4) for v in obj.location],
                "dimensions": [round(v, 4) for v in obj.dimensions]}}
''' + _emit("payload")


def build_set_object_transform(
    name: str,
    location: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> str:
    """Absolute placement, as against move_object's relative nudge. Both exist because
    models reach for whichever matches how they are thinking, and forcing one means they
    fake the other with arithmetic they get wrong."""
    # Built before the f-string: `list(x)` inside one is evaluated whether or not the
    # conditional around it is taken, so a None here raised TypeError at build time.
    loc = list(location)[:3] if location is not None else None
    rot = list(rotation)[:3] if rotation is not None else None
    scl = list(scale)[:3] if scale is not None else None
    return _preamble() + f'''
import math
obj = bpy.data.objects.get({name!r})
location = {loc!r}
rotation = {rot!r}
scale = {scl!r}
if obj is None:
    payload = {{"error": "no object named " + {name!r}}}
else:
    if location is not None:
        obj.location = tuple(location[:3])
    if rotation is not None:
        obj.rotation_euler = tuple(math.radians(a) for a in rotation[:3])
    if scale is not None:
        obj.scale = tuple(scale[:3])
    bpy.context.view_layer.update()
    payload = {{"name": obj.name,
                "location": [round(v, 4) for v in obj.location],
                "rotation_deg": [round(math.degrees(a), 2) for a in obj.rotation_euler],
                "scale": [round(v, 4) for v in obj.scale],
                "dimensions": [round(v, 4) for v in obj.dimensions]}}
''' + _emit("payload")


def build_set_material(
    color: list[float],
    name: str | None = None,
    names: list[str] | None = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> str:
    """Colour one object or many, with one shared material per colour.

    `names` exists because of a round budget, not elegance: colouring a Rubik's cube one
    cubelet per round costs 26 rounds and the model never reaches the part where it looks
    at the result. A whole face should be one call.
    """
    targets = list(names) if names else ([name] if name else [])
    if not targets:
        # An error the model can read and act on, rather than an exception. Neither field
        # is required by the schema (JSON Schema cannot say "one of these two" in a way
        # small models reliably honour), so this path is reachable by an honest caller.
        return _preamble() + _emit(
            '{"error": "set_material needs name (one object) or names (several)"}'
        )
    rgb = (list(color)[:3] + [0.0, 0.0, 0.0])[:3]
    return _preamble() + f'''
targets = {targets!r}
rgb = tuple({rgb!r})
key = "AGENT_MAT_%.3f_%.3f_%.3f_%.2f_%.2f" % (rgb[0], rgb[1], rgb[2], {float(roughness)}, {float(metallic)})
mat = bpy.data.materials.get(key)
if mat is None:
    mat = bpy.data.materials.new(key)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = {float(roughness)}
        bsdf.inputs["Metallic"].default_value = {float(metallic)}
coloured = []
missing = []
for target in targets:
    obj = bpy.data.objects.get(target)
    if obj is None or obj.type != "MESH":
        missing.append(target)
        continue
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    coloured.append(obj.name)
payload = {{"coloured": coloured, "material": key, "color": list(rgb)}}
if missing:
    payload["not_found"] = missing
''' + _emit("payload")


def build_duplicate_grid(
    name: str, counts: list[int], spacing: list[float], prefix: str | None = None
) -> str:
    """A 3x3x3 grid is 27 objects. Without this the model spends its whole round budget
    on primitive_add calls and never reaches the part where it looks at the result.

    Capped at 200 copies: a model that means 3 and types 30 should get an error, not a
    hung Blender.
    """
    counts = [max(1, int(c)) for c in (list(counts)[:3] + [1, 1, 1])[:3]]
    space = (list(spacing)[:3] + [0.0, 0.0, 0.0])[:3]
    return _preamble() + f'''
source = bpy.data.objects.get({name!r})
counts = {counts!r}
spacing = {space!r}
prefix = {prefix!r} or ({name!r} + "_")
total = counts[0] * counts[1] * counts[2]
if source is None:
    payload = {{"error": "no object named " + {name!r}}}
elif total > 200:
    payload = {{"error": "that is " + str(total) + " copies; the limit is 200"}}
else:
    created = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                if i == j == k == 0:
                    copy = source
                else:
                    copy = source.copy()
                    # source.data.copy(), not source.data. Linked duplicates share one
                    # mesh datablock, and material slots live on the mesh, so colouring
                    # one cubelet coloured all 27. The first live Rubik's cube came out
                    # uniformly blue for exactly this reason: red, then green, then blue,
                    # last write winning across the whole grid.
                    copy.data = source.data.copy()
                    bpy.context.scene.collection.objects.link(copy)
                copy.name = prefix + "%d_%d_%d" % (i, j, k)
                copy.location = (source.location.x + i * spacing[0],
                                 source.location.y + j * spacing[1],
                                 source.location.z + k * spacing[2])
                created.append(copy.name)
    payload = {{"created": created, "count": len(created)}}
''' + _emit("payload")


def build_mirror_object(name: str, new_name: str, axis: str = "x") -> str:
    """Mirror one object across a world axis, as a new object.

    Added because symmetry is where the model measurably fails, and the fix belongs in the
    library rather than in the prompt. Asked for a symmetrical character it placed
    `eye_l` at y=-0.30 (the face) and `eye_r` at y=+0.32 (the back of the head): one sign
    flipped, on a figure it had been told twice to keep symmetrical. Doing the reflection
    as arithmetic means getting it right once per part; doing it as a verb means getting
    it right once, here.

    Scale is negated rather than the mesh rebuilt, so a mirrored part stays linked in
    shape to its original.
    """
    return _preamble() + f'''
source = bpy.data.objects.get({name!r})
axis = {axis!r}.strip().lower()
index = {{"x": 0, "y": 1, "z": 2}}.get(axis)
if source is None:
    payload = {{"error": "no object named " + {name!r}}}
elif index is None:
    payload = {{"error": "axis must be x, y or z, not " + axis}}
elif {new_name!r} in bpy.data.objects:
    payload = {{"error": "an object named " + {new_name!r} + " already exists"}}
else:
    copy = source.copy()
    copy.data = source.data.copy()
    bpy.context.scene.collection.objects.link(copy)
    copy.name = {new_name!r}
    location = list(source.location)
    location[index] = -location[index]
    copy.location = tuple(location)
    scale = list(source.scale)
    scale[index] = -scale[index]
    copy.scale = tuple(scale)
    bpy.context.view_layer.update()
    payload = {{"created": copy.name, "mirrored": source.name, "axis": axis,
                "location": [round(v, 4) for v in copy.location]}}
''' + _emit("payload")


def build_delete_object(name: str) -> str:
    return _preamble() + f'''
obj = bpy.data.objects.get({name!r})
if obj is None:
    payload = {{"error": "no object named " + {name!r}}}
else:
    bpy.data.objects.remove(obj, do_unlink=True)
    payload = {{"deleted": {name!r}}}
''' + _emit("payload")


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "describe_scene",
            "observe",
            "List every object in the live Blender scene with its type, world position, "
            "bounding box, dimensions and polygon count, plus the active camera, render "
            "resolution and world lighting strength. Changes nothing. Call this first.",
            _obj({}, []),
            build_describe_scene,
        ),
        Tool(
            "render_view",
            "observe",
            "Render the active camera to a PNG and show it to you as an image. This is "
            "how you see your own work; call it after any change you want to judge.",
            _obj(
                {
                    "resolution": {**_NUM, "description": "Square pixel size, 256-1024."},
                    "samples": {**_NUM, "description": "Cycles samples. 32 is enough to judge."},
                },
                [],
            ),
            build_render_view,
        ),
        Tool(
            "import_asset",
            "modify",
            "Import a .glb asset into the scene, clearing any previous mesh first.",
            _obj({"path": {**_STR, "description": "Filesystem path to the .glb."},
                  "clear_scene": {"type": "boolean"}}, ["path"]),
            build_import_asset,
        ),
        Tool(
            "set_camera",
            "modify",
            "Place the camera on an orbit around the subject. azimuth 0 is straight in "
            "front of the subject and increases to the subject's left; elevation 0 is "
            "level with the middle of the subject and 90 is directly overhead. "
            "distance_factor multiplies the subject's radius, so 2.2 is a close portrait "
            "and 4 is a wide shot. Flat ground planes are ignored when working out what "
            "the subject is, unless you name one as the target.",
            _obj(
                {
                    "azimuth": {**_NUM, "description": "Degrees, 0 is front."},
                    "elevation": {**_NUM, "description": "Degrees, negative looks up from below."},
                    "distance_factor": _NUM,
                    "lens_mm": {**_NUM, "description": "Focal length; 85 flatters a face, 28 distorts."},
                    "target": {**_STR, "description": "Optional mesh name to frame."},
                },
                ["azimuth", "elevation"],
            ),
            build_set_camera,
        ),
        Tool(
            "set_light",
            "modify",
            "Create or move one named light on an orbit around the subject. Use role "
            "'key' for the main light, 'fill' to open up the shadows and 'rim' behind the "
            "subject to separate it from the background. Same angle convention as the "
            "camera.",
            _obj(
                {
                    "role": {"type": "string", "enum": ["key", "fill", "rim"]},
                    "azimuth": _NUM,
                    "elevation": _NUM,
                    "energy": {**_NUM, "description": "Watts. Start near 200 and judge by eye."},
                    "size": {**_NUM, "description": "Area size as a fraction of subject radius; bigger is softer."},
                    "color": {"type": "array", "items": _NUM, "description": "RGB 0-1."},
                },
                ["role", "azimuth", "elevation", "energy"],
            ),
            build_set_light,
        ),
        Tool(
            "set_world",
            "modify",
            "Set the world background colour and strength, which is the ambient light "
            "everything sits in.",
            _obj({"strength": _NUM, "color": {"type": "array", "items": _NUM}}, ["strength"]),
            build_set_world,
        ),
        Tool(
            "move_object",
            "modify",
            "Nudge one named object by a relative offset in Blender world space "
            "(x right, y back, z up).",
            _obj({"name": _STR, "dx": _NUM, "dy": _NUM, "dz": _NUM}, ["name"]),
            build_move_object,
        ),
        Tool(
            "clear_scene",
            "modify",
            "Delete every mesh and light in the scene so you can start from nothing. "
            "Cameras are always kept. Set keep_lights true to keep your lighting.",
            _obj({"keep_lights": {"type": "boolean"}}, []),
            build_clear_scene,
        ),
        Tool(
            "add_primitive",
            "modify",
            "Create one primitive mesh: cube, sphere, cylinder, cone, torus or plane. "
            "You choose its name, and that name is how you refer to it afterwards, so "
            "make it descriptive. size is the full width in Blender units.",
            _obj(
                {
                    "kind": {"type": "string",
                             "enum": ["cube", "sphere", "cylinder", "cone", "torus", "plane"]},
                    "name": _STR,
                    "location": {"type": "array", "items": _NUM, "description": "[x, y, z]."},
                    "size": {**_NUM, "description": "Overall width before scaling."},
                    "rotation": {"type": "array", "items": _NUM, "description": "[x, y, z] degrees."},
                    "scale": {"type": "array", "items": _NUM,
                              "description": "[x, y, z] stretch. Use this to make a flat "
                                             "torso or a long limb from a cube in one call."},
                },
                ["kind", "name"],
            ),
            build_add_primitive,
        ),
        Tool(
            "set_object_transform",
            "modify",
            "Set one object's absolute location, rotation (degrees) and scale. Any field "
            "you leave out is left alone.",
            _obj(
                {
                    "name": _STR,
                    "location": {"type": "array", "items": _NUM},
                    "rotation": {"type": "array", "items": _NUM},
                    "scale": {"type": "array", "items": _NUM},
                },
                ["name"],
            ),
            build_set_object_transform,
        ),
        Tool(
            "set_material",
            "modify",
            "Give one object, or many at once, a solid colour. roughness runs 0 glossy "
            "to 1 matte, metallic 0 for plastic or 1 for metal. Use names to colour a "
            "whole set in one call, which is much cheaper than one call each.",
            _obj(
                {
                    "name": {**_STR, "description": "A single object to colour."},
                    "names": {"type": "array", "items": _STR,
                              "description": "Several objects to colour the same way."},
                    "color": {"type": "array", "items": _NUM, "description": "RGB, each 0-1."},
                    "roughness": _NUM,
                    "metallic": _NUM,
                },
                ["color"],
            ),
            build_set_material,
        ),
        Tool(
            "duplicate_grid",
            "modify",
            "Copy one object into a 3D grid, which is how you build anything repetitive "
            "without calling add_primitive dozens of times. counts is [nx, ny, nz] and "
            "spacing is the gap between copies on each axis. Copies are named "
            "<prefix>i_j_k so you can colour them individually afterwards.",
            _obj(
                {
                    "name": _STR,
                    "counts": {"type": "array", "items": {"type": "integer"}},
                    "spacing": {"type": "array", "items": _NUM},
                    "prefix": _STR,
                },
                ["name", "counts", "spacing"],
            ),
            build_duplicate_grid,
        ),
        Tool(
            "mirror_object",
            "modify",
            "Copy an object mirrored across a world axis. Use axis 'x' for left/right "
            "symmetry: build the left arm, eye or leg, get it right, then mirror it "
            "instead of working out the opposite position yourself.",
            _obj(
                {
                    "name": {**_STR, "description": "The object to mirror."},
                    "new_name": {**_STR, "description": "Name for the mirrored copy."},
                    "axis": {"type": "string", "enum": ["x", "y", "z"]},
                },
                ["name", "new_name"],
            ),
            build_mirror_object,
        ),
        Tool(
            "delete_object",
            "modify",
            "Remove one object by name. Use this to undo something you regret.",
            _obj({"name": _STR}, ["name"]),
            build_delete_object,
        ),
        Tool(
            "set_render_settings",
            "modify",
            "Choose the render engine, whether the background is transparent, and the "
            "exposure in stops. Use eevee while you are iterating.",
            _obj(
                {
                    "engine": {"type": "string", "enum": ["eevee", "cycles", "workbench"],
                               "description": "eevee is fast and good enough to judge by; "
                                              "cycles is slow and accurate."},
                    "film_transparent": {"type": "boolean"},
                    "exposure": _NUM,
                },
                [],
            ),
            build_set_render_settings,
        ),
    )
}

OBSERVE_TOOLS = tuple(n for n, t in TOOLS.items() if t.kind == "observe")
MODIFY_TOOLS = tuple(n for n, t in TOOLS.items() if t.kind == "modify")


def build_call(name: str, arguments: dict[str, Any],
               injected: dict[str, Any] | None = None) -> str:
    """Turn one model tool call into the bpy that runs it.

    Unknown tools and unknown arguments are refused rather than ignored: a model that
    invents `set_camera(roll=...)` should be told, not silently obeyed with the roll
    dropped, because a silently dropped argument looks to the model like the tool is
    broken.

    `injected` is for arguments the harness owns and the model must not choose, notably
    where a render is written. They are deliberately absent from the JSON schema, so the
    model cannot be talked into writing a PNG over something that matters.
    """
    tool = TOOLS.get(name)
    if tool is None:
        raise KeyError(f"no tool named {name!r}; have {sorted(TOOLS)}")
    allowed = set(tool.parameters.get("properties", {}))
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise TypeError(f"{name} got unknown argument(s) {unknown}; accepts {sorted(allowed)}")
    missing = sorted(set(tool.parameters.get("required", [])) - set(arguments))
    if missing:
        raise TypeError(f"{name} is missing required argument(s) {missing}")
    return tool.builder(**{**arguments, **(injected or {})})


def parse_response(raw: str) -> dict[str, Any]:
    """Pull our sentinel line out of whatever Blender sent back.

    Blender's own logging shares that stdout, and a failed snippet comes back as an error
    envelope with no sentinel at all, so both cases have to be readable.
    """
    start = raw.find("{")
    if start < 0:
        return {"error": "no JSON in Blender's reply", "raw": raw[:500]}
    try:
        envelope = json.loads(raw[start:])
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable reply: {exc}", "raw": raw[:500]}
    if envelope.get("status") == "error":
        return {"error": envelope.get("message", "Blender reported an error")}
    body = envelope.get("result", {})
    stdout = body.get("result", "") if isinstance(body, dict) else str(body)
    marker = stdout.rfind(RESULT_SENTINEL)
    if marker < 0:
        return {"error": "tool produced no result line", "stdout": stdout[-500:]}
    line = stdout[marker + len(RESULT_SENTINEL):].splitlines()[0]
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable result line: {exc}", "raw": line[:500]}


def send(code: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: int = 300) -> str:
    """Same protocol the other blender_* scripts use: one execute_code request per call."""
    request = {"type": "execute_code", "params": {"code": code}}
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(timeout)
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


# --------------------------------------------------------------------------------------
# The conversation.
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are building and staging 3D scenes in Blender through a fixed set of tools.

You cannot write Blender Python. You can only call the tools listed. Work like a
photographer: look before you touch, change one thing, render, and judge what you see.

The loop that works:
1. describe_scene, to learn what is actually there.
2. Make one change.
3. render_view, and actually look at the image you get back.
4. Say in one sentence what is wrong with it, then fix that one thing.

Rules:
- Never call two modifying tools in a row without rendering in between.
- If a render looks black, the lights are too weak or the world strength is 0.
- If a render looks blown out white, the energy is too high.
- Judge the image, not your intentions. If the last change made it worse, undo it by
  calling the same tool with different values.
- To build something repetitive, make one piece, get it right, then duplicate_grid it.
- For anything left/right symmetrical, build ONE side, check it in a render, then
  mirror_object it on the x axis. Do not work out the opposite position by hand.
- Name objects for what they are, because names are how you refer back to them.
- When the task is met, say DONE and stop calling tools.
"""


def image_message(path: Path, note: str) -> dict[str, Any]:
    """Feed a render back as an image, which is the whole point of the loop."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": note},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
    }


def needs_forced_render(called: list[str]) -> bool:
    """True when a round changed the scene but never looked at the result.

    Measured on the first live run (2026-09-22, gemma-4-31b): asked to build a red cube on
    a ground plane, the model emitted eight tool calls in a single round, seven of them
    modifications, and never called render_view at all. The system prompt asks for one
    change then a render; the model ignored it, and parallel tool calls make that easy to
    do by accident.

    The interleaved feedback is the whole mechanism the benchmark attributes its gain to,
    so it cannot be left to the model's goodwill. If a round modified the scene without
    rendering, the harness renders anyway and hands back the picture.
    """
    modified = any(TOOLS.get(name) is not None and TOOLS[name].kind == "modify"
                   for name in called)
    return modified and "render_view" not in called


@dataclass
class RunLog:
    directory: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    renders: list[str] = field(default_factory=list)

    def record(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self.calls.append({"tool": name, "arguments": arguments, "result": result})

    def write(self, messages: list[dict[str, Any]]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "calls.json").write_text(json.dumps(self.calls, indent=2))
        # Images are stripped from the saved transcript; they are already on disk as PNGs
        # and a base64 blob per turn makes the log unreadable.
        trimmed = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [c for c in content if c.get("type") != "image_url"] + [
                    {"type": "text", "text": "<render attached>"}
                ]
            trimmed.append({**message, "content": content})
        (self.directory / "transcript.json").write_text(json.dumps(trimmed, indent=2))


def chat(api: str, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
         timeout: int = 600) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        f"{api.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def opening_messages(task: str, asset: Path | None, reference: Path | None) -> list[dict[str, Any]]:
    """The first user turn, which is the only place the target enters the conversation.

    A reference image turns the run into inverse graphics: the model is no longer staging
    to taste, it is building until its render matches a picture. That is the task the
    benchmark actually measures, and it is a far sharper judge than a prose brief.
    """
    opening = task
    if asset is not None:
        opening = f"The asset to work with is at {asset}. Import it first, then: {task}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening},
    ]
    if reference is not None:
        messages.append(image_message(
            reference,
            "This is the reference to match. Study it, then build toward it. After every "
            "render, compare your render against this reference and name the single "
            "biggest difference.",
        ))
    return messages


def run(task: str, asset: Path | None, model: str, api: str, out_dir: Path,
        max_rounds: int, host: str, port: int, resolution: int,
        reference: Path | None = None, auto_render: bool = True) -> int:
    # Absolute, always. The generated code runs inside Blender, whose working directory
    # is not ours, so a relative --out-dir made every render fail with "Render error
    # (Read-only file system)" while every other tool succeeded. Found on the second live
    # run, where it cost the whole run: the model kept changing the scene and never once
    # got a picture back.
    out_dir = out_dir.expanduser().resolve()
    if asset is not None:
        asset = asset.expanduser().resolve()
    if reference is not None:
        reference = reference.expanduser().resolve()
    log = RunLog(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = [tool.schema() for tool in TOOLS.values()]

    messages = opening_messages(task, asset, reference)

    for round_index in range(max_rounds):
        try:
            reply = chat(api, model, messages, schemas)
        except urllib.error.URLError as exc:
            print(f"cannot reach the model API at {api}: {exc}", file=sys.stderr)
            print("start LM Studio's server ('lms server start') and load the model.",
                  file=sys.stderr)
            return 2
        choice = reply["choices"][0]["message"]
        messages.append(choice)
        calls = choice.get("tool_calls") or []
        text = (choice.get("content") or "").strip()
        if text:
            print(f"[{round_index}] {text}")
        if not calls:
            print("model stopped calling tools.")
            break

        called: list[str] = []
        for call in calls:
            name = call["function"]["name"]
            called.append(name)
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"your arguments were not valid JSON: {exc}"}
                arguments = {}
            else:
                injected: dict[str, Any] = {}
                if name == "render_view":
                    injected["output_path"] = str(out_dir / f"round{round_index:02d}.png")
                    arguments.setdefault("resolution", resolution)
                try:
                    code = build_call(name, arguments, injected)
                except (KeyError, TypeError) as exc:
                    result = {"error": str(exc)}
                else:
                    result = parse_response(send(code, host, port))
            print(f"    -> {name}({json.dumps(arguments)[:120]}) = {json.dumps(result)[:160]}")
            log.record(name, arguments, result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "content": json.dumps(result),
            })
            if name == "render_view" and "rendered" in result:
                path = Path(result["rendered"])
                if path.exists():
                    log.renders.append(str(path))
                    messages.append(image_message(path, "Here is that render. What is wrong with it?"))

        if auto_render and needs_forced_render(called):
            forced = out_dir / f"round{round_index:02d}-auto.png"
            result = parse_response(send(
                build_call("render_view", {"resolution": resolution},
                           {"output_path": str(forced)}), host, port))
            print(f"    -> render_view(forced) = {json.dumps(result)[:120]}")
            log.record("render_view", {"forced": True}, result)
            if "rendered" in result and forced.exists():
                log.renders.append(str(forced))
                messages.append(image_message(
                    forced,
                    "You changed the scene without looking at it, so here is the render. "
                    "Name the single biggest thing wrong with it, then fix only that.",
                ))

        log.write(messages)

    log.write(messages)
    print(f"\n{len(log.calls)} tool calls, {len(log.renders)} renders, logged in {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="What the model should achieve.")
    parser.add_argument("--asset", type=Path, help="A .glb to import before starting.")
    parser.add_argument("--reference", type=Path,
                        help="A target image to build toward. Turns the run into "
                             "inverse graphics: match this picture.")
    parser.add_argument("--model", default="gemma-4-31b-it-qat",
                        help="Model id as the local server reports it. gemma-4-12b-qat "
                             "does NOT work: LM Studio cannot render tool schemas into "
                             "its Jinja template and every request 400s.")
    parser.add_argument("--api", default=DEFAULT_API, help="OpenAI-compatible base URL.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where renders and the transcript are written.")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--no-auto-render", action="store_true",
                        help="Do not force a render after a round that changed the scene "
                             "without looking. Only useful for measuring how well the "
                             "model self-regulates, which on the evidence is badly.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--list-tools", action="store_true",
                        help="Print the skill library and exit, without touching Blender.")
    args = parser.parse_args(argv)

    if args.list_tools:
        for kind, names in (("observe", OBSERVE_TOOLS), ("modify", MODIFY_TOOLS)):
            print(f"{kind}:")
            for name in names:
                print(f"  {name}: {TOOLS[name].description.splitlines()[0]}")
        return 0

    if args.asset is not None and not args.asset.exists():
        parser.error(f"no such asset: {args.asset}")
    if args.reference is not None and not args.reference.exists():
        parser.error(f"no such reference image: {args.reference}")
    return run(args.task, args.asset, args.model, args.api, args.out_dir,
               args.max_rounds, args.host, args.port, args.resolution, args.reference,
               not args.no_auto_render)


if __name__ == "__main__":
    raise SystemExit(main())
