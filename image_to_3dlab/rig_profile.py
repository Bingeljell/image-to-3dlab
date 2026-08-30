"""Load versioned semantic-to-Rigify joint mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
PROFILE_ROOT = REPO / "rigs" / "profiles"


class RigProfileError(ValueError):
    pass


def load_rig_profile(profile_id: str, root: Path = PROFILE_ROOT) -> dict[str, Any]:
    if not profile_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in profile_id):
        raise RigProfileError("rig profile id contains unsafe characters")
    path = root / f"{profile_id}.json"
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RigProfileError(f"cannot load rig profile {profile_id!r}: {exc}") from exc
    return validate_rig_profile(source, expected_id=profile_id)


def validate_rig_profile(source: Any, expected_id: str | None = None) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise RigProfileError("rig profile root must be an object")
    if set(source) != {
        "schemaVersion", "id", "semanticProfile", "blenderVersion",
        "blenderMetarigNames", "joints",
    }:
        raise RigProfileError("rig profile root fields do not match schema v1")
    if source["schemaVersion"] != 1:
        raise RigProfileError("rig profile schemaVersion must be 1")
    profile_id = _string(source["id"], "id")
    if expected_id is not None and profile_id != expected_id:
        raise RigProfileError(f"rig profile id is {profile_id!r}, expected {expected_id!r}")
    semantic_profile = _string(source["semanticProfile"], "semanticProfile")
    blender_version = _string(source["blenderVersion"], "blenderVersion")
    names = source["blenderMetarigNames"]
    if not isinstance(names, list) or not names:
        raise RigProfileError("blenderMetarigNames must be a non-empty array")
    names = [_string(name, "blenderMetarigNames") for name in names]
    raw_joints = source["joints"]
    if not isinstance(raw_joints, Mapping) or not raw_joints:
        raise RigProfileError("joints must be a non-empty object")

    joints: dict[str, dict[str, Any]] = {}
    required = {"label", "sourceBone", "parent", "mirrorOf", "targets"}
    for joint_id, value in raw_joints.items():
        if not isinstance(value, Mapping) or set(value) != required:
            raise RigProfileError(f"joint {joint_id!r} fields do not match schema v1")
        targets = value["targets"]
        if not isinstance(targets, list) or not targets:
            raise RigProfileError(f"joint {joint_id!r} needs at least one Blender target")
        normalized_targets = []
        for target in targets:
            if not isinstance(target, Mapping) or set(target) != {"bone", "endpoint"}:
                raise RigProfileError(f"joint {joint_id!r} has an invalid Blender target")
            endpoint = target["endpoint"]
            if endpoint not in {"head", "tail"}:
                raise RigProfileError(f"joint {joint_id!r} endpoint must be head or tail")
            normalized_targets.append({"bone": _string(target["bone"], "bone"), "endpoint": endpoint})
        joints[joint_id] = {
            "label": _string(value["label"], "label"),
            "sourceBone": _string(value["sourceBone"], "sourceBone"),
            "parent": _nullable_string(value["parent"], "parent"),
            "mirrorOf": _nullable_string(value["mirrorOf"], "mirrorOf"),
            "targets": normalized_targets,
        }
    for joint_id, joint in joints.items():
        for field in ("parent", "mirrorOf"):
            if joint[field] is not None and joint[field] not in joints:
                raise RigProfileError(f"joint {joint_id!r} {field} references an unknown joint")
    return {
        "schemaVersion": 1,
        "id": profile_id,
        "semanticProfile": semantic_profile,
        "blenderVersion": blender_version,
        "blenderMetarigNames": names,
        "joints": joints,
    }


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RigProfileError(f"{field} must be a non-empty string")
    return value


def _nullable_string(value: Any, field: str) -> str | None:
    return None if value is None else _string(value, field)
