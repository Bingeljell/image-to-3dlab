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

if __package__:
    from .blender_agent_worker import RESULT_SENTINEL
    from .blender_worker_source import build_worker_code
else:
    from blender_agent_worker import RESULT_SENTINEL
    from blender_worker_source import build_worker_code

WORKER_PATH = Path(__file__).with_name("blender_agent_worker.py")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
DEFAULT_API = "http://127.0.0.1:1234/v1"


# --------------------------------------------------------------------------------------
# The skill library.
#
# Each tool pairs a JSON schema with a builder that packs the shipped worker and its
# arguments. The worker runs inside Blender without needing a checkout.
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


def _build_action(action: str, **parameters: Any) -> str:
    try:
        return build_worker_code(WORKER_PATH, {"action": action, "parameters": parameters})
    except ValueError as exc:
        raise TypeError(f"{action} requires JSON-compatible arguments: {exc}") from exc


def build_describe_scene() -> str:
    return _build_action("describe_scene")


def build_render_view(output_path: str, resolution: int = 640, samples: int = 32) -> str:
    return _build_action("render_view", output_path=output_path,
                         resolution=int(resolution), samples=int(samples))


def build_import_asset(path: str, clear_scene: bool = True) -> str:
    return _build_action("import_asset", path=path, clear_scene=bool(clear_scene))


def build_set_camera(
    azimuth: float,
    elevation: float,
    distance_factor: float = 2.2,
    lens_mm: float = 50.0,
    target: str | None = None,
) -> str:
    """Frame the subject from the requested orbit angles."""
    return _build_action(
        "set_camera", azimuth=float(azimuth), elevation=float(elevation),
        distance_factor=float(distance_factor), lens_mm=float(lens_mm), target=target,
    )


def build_set_light(
    role: str,
    azimuth: float,
    elevation: float,
    energy: float,
    size: float = 1.0,
    color: list[float] | None = None,
) -> str:
    """Place or reposition the light identified by its role."""
    return _build_action(
        "set_light", role=role, azimuth=float(azimuth), elevation=float(elevation),
        energy=float(energy), size=float(size), color=list(color or [1.0, 1.0, 1.0]),
    )


def build_set_world(strength: float, color: list[float] | None = None) -> str:
    return _build_action("set_world", strength=float(strength),
                         color=list(color or [0.05, 0.05, 0.06]))


def build_move_object(name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    return _build_action("move_object", name=name, dx=float(dx), dy=float(dy), dz=float(dz))


def build_set_render_settings(
    engine: str = "eevee", film_transparent: bool = False, exposure: float = 0.0
) -> str:
    return _build_action("set_render_settings", engine=engine,
                         film_transparent=bool(film_transparent), exposure=float(exposure))


def build_clear_scene(keep_lights: bool = False) -> str:
    return _build_action("clear_scene", keep_lights=bool(keep_lights))


def build_add_primitive(
    kind: str,
    name: str,
    location: list[float] | None = None,
    size: float = 1.0,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> str:
    """Create and shape a named primitive in a single action."""
    loc = (list(location or []) + [0.0, 0.0, 0.0])[:3]
    rot = (list(rotation or []) + [0.0, 0.0, 0.0])[:3]
    scl = (list(scale or []) + [1.0, 1.0, 1.0])[:3]
    return _build_action("add_primitive", kind=kind, name=name, location=loc,
                         size=float(size), rotation=rot, scale=scl)


def build_set_object_transform(
    name: str,
    location: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> str:
    """Set the transform channels supplied in the call."""
    loc = list(location)[:3] if location is not None else None
    rot = list(rotation)[:3] if rotation is not None else None
    scl = list(scale)[:3] if scale is not None else None
    return _build_action("set_object_transform", name=name, location=loc, rotation=rot, scale=scl)


def build_set_material(
    color: list[float],
    name: str | None = None,
    names: list[str] | None = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> str:
    targets = list(names) if names else ([name] if name else [])
    if not targets:
        return _build_action("set_material", targets=[], color=[], roughness=0.5, metallic=0.0)
    rgb = (list(color)[:3] + [0.0, 0.0, 0.0])[:3]
    return _build_action("set_material", targets=targets, color=rgb,
                         roughness=float(roughness), metallic=float(metallic))


def build_duplicate_grid(
    name: str, counts: list[int], spacing: list[float], prefix: str | None = None
) -> str:
    counts = [max(1, int(c)) for c in (list(counts)[:3] + [1, 1, 1])[:3]]
    space = (list(spacing)[:3] + [0.0, 0.0, 0.0])[:3]
    return _build_action("duplicate_grid", name=name, counts=counts, spacing=space, prefix=prefix)


def build_mirror_object(name: str, new_name: str, axis: str = "x") -> str:
    return _build_action("mirror_object", name=name, new_name=new_name, axis=axis)


def build_delete_object(name: str) -> str:
    return _build_action("delete_object", name=name)


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
