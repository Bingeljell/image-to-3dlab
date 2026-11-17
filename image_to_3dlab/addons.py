"""Pure add-on helpers: manifest parsing, command building, and provenance records.

Nothing here touches the network, `bpy`, or a subprocess. The CLI wrapper lives in
`scripts/addon.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INPUT_TYPES = {"text", "number", "choice", "toggle", "picture", "model"}
ADDON_TYPES = {"tool", "app"}
REQUIRED_MANIFEST_FIELDS = ("schema", "id", "name", "version", "author", "licence", "summary")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class Input:
    id: str
    type: str
    label: str
    optional: bool = False
    choices: tuple[str, ...] | None = None
    min: float | int | None = None
    max: float | int | None = None


@dataclass(frozen=True)
class Tool:
    id: str
    label: str
    hint: str
    inputs: tuple[Input, ...]
    run: tuple[str, ...]
    makes: str


@dataclass(frozen=True)
class Manifest:
    schema: int
    id: str
    name: str
    version: str
    author: dict[str, str]
    licence: str
    summary: str
    needs: dict[str, str]
    setup: tuple[str, ...]
    models: tuple[dict[str, Any], ...]
    tools: tuple[Tool, ...]
    type: str = "tool"


@dataclass(frozen=True)
class Listing:
    id: str
    name: str
    repo: str
    commit: str
    author: dict[str, str]
    summary: str
    type: str


def validate_addon_id(addon_id: str) -> None:
    """Refuse ids that could escape the addons/ folder or look like paths."""
    if not isinstance(addon_id, str) or not ID_RE.fullmatch(addon_id):
        raise ValueError(f"invalid add-on id: {addon_id!r}")


def _require_field(data: dict[str, Any], field: str, where: str) -> Any:
    if field not in data:
        raise ValueError(f"{where} is missing required field '{field}'")
    return data[field]


def _read_input(tool_index: int, index: int, data: dict[str, Any]) -> Input:
    where = f"tools[{tool_index}].inputs[{index}]"
    _require_field(data, "id", where)
    _require_field(data, "type", where)
    _require_field(data, "label", where)
    input_type = data["type"]
    if input_type not in INPUT_TYPES:
        raise ValueError(f"{where} has unknown type {input_type!r}")
    if input_type == "choice" and not data.get("choices"):
        raise ValueError(f"{where} of type 'choice' needs a non-empty 'choices' list")
    return Input(
        id=str(data["id"]),
        type=input_type,
        label=str(data["label"]),
        optional=bool(data.get("optional", False)),
        choices=tuple(str(c) for c in data["choices"]) if input_type == "choice" else None,
        min=data.get("min"),
        max=data.get("max"),
    )


def _read_tool(index: int, data: dict[str, Any]) -> Tool:
    where = f"tools[{index}]"
    _require_field(data, "id", where)
    _require_field(data, "label", where)
    _require_field(data, "hint", where)
    _require_field(data, "run", where)
    _require_field(data, "makes", where)
    inputs = data.get("inputs", [])
    if not isinstance(inputs, list):
        raise ValueError(f"{where}.inputs must be a list")
    return Tool(
        id=str(data["id"]),
        label=str(data["label"]),
        hint=str(data["hint"]),
        inputs=tuple(_read_input(index, i, item) for i, item in enumerate(inputs)),
        run=tuple(str(arg) for arg in data["run"]),
        makes=str(data["makes"]),
    )


def tool_from_dict(data: dict[str, Any]) -> Tool:
    """Build a Tool from a plain dict (tests use this to avoid nested dataclass construction)."""
    return _read_tool(0, data)


def read_manifest(path: Path) -> Manifest:
    """Read and validate an add-on's `addon.json`."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read addon.json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("addon.json must be an object")
    schema = _require_field(data, "schema", "addon.json")
    if schema != 1:
        raise ValueError(f"addon.json schema must be 1; got {schema!r}")
    for field in REQUIRED_MANIFEST_FIELDS:
        _require_field(data, field, "addon.json")
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("addon.json 'tools' must be a list")
    addon_type = data.get("type", "tool")
    if addon_type not in ADDON_TYPES:
        raise ValueError(f"addon.json 'type' must be one of {ADDON_TYPES}; got {addon_type!r}")
    validate_addon_id(data["id"])
    return Manifest(
        schema=int(schema),
        id=str(data["id"]),
        name=str(data["name"]),
        version=str(data["version"]),
        author=_author(data.get("author", {})),
        licence=str(data["licence"]),
        summary=str(data["summary"]),
        needs=dict(data.get("needs", {})),
        setup=tuple(str(arg) for arg in data.get("setup", [])),
        models=tuple(dict(item) for item in data.get("models", [])),
        tools=tuple(_read_tool(i, item) for i, item in enumerate(tools)),
        type=addon_type,
    )


def _author(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError("addon.json 'author' must be an object with at least 'name'")
    if "name" not in data:
        raise ValueError("addon.json 'author' is missing 'name'")
    return {"name": str(data["name"]), "url": str(data.get("url", ""))}


def read_registry(data: list[dict[str, Any]]) -> list[Listing]:
    """Read the registry of reviewed add-ons."""
    if not isinstance(data, list):
        raise ValueError("registry must be a list")
    listings: list[Listing] = []
    for index, item in enumerate(data):
        where = f"registry[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        for field in ("id", "name", "repo", "commit"):
            _require_field(item, field, where)
        if not item["commit"]:
            raise ValueError(f"{where} 'commit' must be a pinned commit hash")
        validate_addon_id(item["id"])
        author = _author(item.get("author", {}))
        summary = str(item.get("summary", ""))
        addon_type = item.get("type", "tool")
        if addon_type not in ADDON_TYPES:
            raise ValueError(f"{where} 'type' must be one of {ADDON_TYPES}; got {addon_type!r}")
        listings.append(
            Listing(
                id=str(item["id"]),
                name=str(item["name"]),
                repo=str(item["repo"]),
                commit=str(item["commit"]),
                author=author,
                summary=summary,
                type=addon_type,
            )
        )
    return listings


def install_plan(listing: Listing, manifest: Manifest, *, reviewed: bool = True) -> dict[str, Any]:
    """What the yes/no prompt shows before installing an add-on."""
    models = [
        {"name": str(model.get("name", "")), "licence": str(model.get("licence", "")),
         "size": str(model.get("size", ""))}
        for model in manifest.models
    ]
    if reviewed:
        warning = (
            f"This add-on runs code on this machine. "
            f"It was reviewed by AssetFurnace at {listing.commit}."
        )
    else:
        warning = (
            "This add-on runs code on this machine. "
            "It is not reviewed by AssetFurnace; install it only if you trust the source."
        )
    return {
        "author": manifest.author["name"],
        "licence": manifest.licence,
        "models": models,
        "setup": list(manifest.setup),
        "warning": warning,
    }


def _validate_input(value: Any, spec: Input) -> None:
    if spec.type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"input '{spec.id}' must be a number")
        if spec.min is not None and value < spec.min:
            raise ValueError(f"input '{spec.id}' must be at least {spec.min}")
        if spec.max is not None and value > spec.max:
            raise ValueError(f"input '{spec.id}' must be at most {spec.max}")
    elif spec.type == "choice":
        if value not in (spec.choices or ()):
            raise ValueError(f"input '{spec.id}' must be one of {spec.choices}; got {value!r}")
    elif spec.type == "toggle":
        if not isinstance(value, bool):
            raise ValueError(f"input '{spec.id}' must be true or false")
    elif spec.type in ("picture", "model", "text"):
        if not isinstance(value, str):
            raise ValueError(f"input '{spec.id}' must be text")


def _coerce_input(value: str, spec: Input, cwd: Path | None) -> Any:
    """Turn one raw --input string into the type the manifest declares."""
    if spec.type == "text":
        return value
    if spec.type == "number":
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"input '{spec.id}' must be a number") from exc
    if spec.type == "choice":
        return value
    if spec.type == "toggle":
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"input '{spec.id}' must be true or false")
    if spec.type in ("picture", "model"):
        path = Path(value)
        if cwd is not None:
            path = cwd / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"input '{spec.id}' file does not exist: {path}")
        return str(path)
    raise ValueError(f"input '{spec.id}' has unknown type {spec.type!r}")


def parse_inputs(
    raw: dict[str, str],
    tool: Tool,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Parse raw --input key=value strings using the tool's declared input types."""
    resolved_cwd = Path(cwd).resolve() if cwd is not None else None
    inputs: dict[str, Any] = {}
    for spec in tool.inputs:
        if spec.id not in raw:
            continue
        inputs[spec.id] = _coerce_input(raw[spec.id], spec, resolved_cwd)
    return inputs


def build_command(tool: Tool, inputs: dict[str, Any], *, out_dir: str | Path,
                  blender: str | Path) -> list[str]:
    """Return an argv list for the tool, with placeholders filled and inputs checked."""
    provided = set(inputs)
    unknown = provided - {spec.id for spec in tool.inputs}
    if unknown:
        raise ValueError(f"unknown input: {min(unknown)}")
    for spec in tool.inputs:
        if spec.id in provided:
            _validate_input(inputs[spec.id], spec)
        elif not spec.optional:
            raise ValueError(f"missing required input: {spec.id}")

    def fill(arg: str) -> str:
        if arg == "{out}":
            return str(out_dir)
        if arg == "{blender}":
            return str(blender)
        match = re.fullmatch(r"\{([A-Za-z0-9_-]+)\}", arg)
        if match:
            key = match.group(1)
            if key not in inputs:
                raise ValueError(f"missing required input: {key}")
            return str(inputs[key])
        return arg

    return [fill(arg) for arg in tool.run]


def output_dir(addon_id: str, run_stem: str) -> Path:
    """Directory an add-on run should put its result in."""
    return Path("output") / f"addon-{addon_id}" / run_stem


def run_record(
    listing: Listing,
    manifest: Manifest,
    tool: Tool,
    inputs: dict[str, Any],
    *,
    reviewed: bool,
) -> dict[str, Any]:
    """A provenance record for an add-on run, written by the core."""
    return {
        "schema_version": 1,
        "addon": {
            "id": manifest.id,
            "name": manifest.name,
            "repo": listing.repo,
            "commit": listing.commit,
            "licence": manifest.licence,
            "reviewed": reviewed,
        },
        "tool": {"id": tool.id, "label": tool.label, "makes": tool.makes},
        "inputs": dict(inputs),
        "models": [
            {
                "name": str(model.get("name", "")),
                "licence": str(model.get("licence", "")),
                "licence_url": str(model.get("licence_url", "")),
                "size": str(model.get("size", "")),
            }
            for model in manifest.models
        ],
    }
