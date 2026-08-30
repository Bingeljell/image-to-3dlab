"""Validate browser rig sidecars and plan authoritative Blender corrections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

FINGERPRINT_PREFIX = "sha256:"
ROOT_KEYS = {
    "schemaVersion", "rigProfile", "assetFingerprint", "coordinateSpace",
    "mirror", "joints", "corrections", "binding",
}
JOINT_KEYS = {"label", "position", "sourceBone", "parent", "mirrorOf"}
CORRECTION_KEYS = {"sourcePosition", "targetPosition", "delta", "mirrored"}
EPSILON = 1e-6


class RigSidecarError(ValueError):
    """A sidecar is malformed, stale, or inconsistent with its source asset."""


@dataclass(frozen=True)
class JointCorrection:
    joint_id: str
    source_bone: str
    source_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    delta: tuple[float, float, float]
    mirrored: bool
    targets: tuple["BoneTarget", ...] = ()


@dataclass(frozen=True)
class BoneTarget:
    bone_id: str
    bone_name: str
    endpoint: str


def asset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return FINGERPRINT_PREFIX + digest.hexdigest()


def load_sidecar(path: Path) -> dict[str, Any]:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RigSidecarError(f"cannot read rig sidecar: {exc}") from exc
    return validate_sidecar(source)


def validate_sidecar(source: Any) -> dict[str, Any]:
    root = _mapping(source, "root")
    _allow_keys(root, ROOT_KEYS, "root")
    if root.get("schemaVersion") != 1:
        _fail("schemaVersion must be 1")
    rig_profile = _string(root.get("rigProfile"), "rigProfile")
    fingerprint = _fingerprint(root.get("assetFingerprint"))
    if root.get("coordinateSpace") != "armature-local":
        _fail('coordinateSpace must be "armature-local"')

    mirror = _mapping(root.get("mirror"), "mirror")
    _allow_keys(mirror, {"axis", "origin"}, "mirror")
    axis = mirror.get("axis")
    if axis not in {"X", "Y", "Z"}:
        _fail("mirror.axis must be X, Y, or Z")
    origin = _number(mirror.get("origin"), "mirror.origin")

    raw_joints = _mapping(root.get("joints"), "joints")
    if not raw_joints:
        _fail("joints must contain at least one joint")
    joints: dict[str, dict[str, Any]] = {}
    for joint_id, value in raw_joints.items():
        _string(joint_id, "joint id")
        joint = _mapping(value, f"joints.{joint_id}")
        _allow_keys(joint, JOINT_KEYS, f"joints.{joint_id}")
        joints[joint_id] = {
            "label": _string(joint.get("label"), f"joints.{joint_id}.label"),
            "position": list(_vector(joint.get("position"), f"joints.{joint_id}.position")),
            "sourceBone": _string(joint.get("sourceBone"), f"joints.{joint_id}.sourceBone"),
            "parent": _nullable_string(joint.get("parent"), f"joints.{joint_id}.parent"),
            "mirrorOf": _nullable_string(joint.get("mirrorOf"), f"joints.{joint_id}.mirrorOf"),
        }
    for joint_id, joint in joints.items():
        for field in ("parent", "mirrorOf"):
            reference = joint[field]
            if reference is not None and reference not in joints:
                _fail(f"joints.{joint_id}.{field} references unknown joint")

    raw_corrections = root.get("corrections", {})
    corrections: dict[str, dict[str, Any]] = {}
    for joint_id, value in _mapping(raw_corrections, "corrections").items():
        if joint_id not in joints:
            _fail(f"corrections.{joint_id} references unknown joint")
        correction = _mapping(value, f"corrections.{joint_id}")
        _allow_keys(correction, CORRECTION_KEYS, f"corrections.{joint_id}")
        source_position = _vector(
            correction.get("sourcePosition"), f"corrections.{joint_id}.sourcePosition"
        )
        target_position = _vector(
            correction.get("targetPosition"), f"corrections.{joint_id}.targetPosition"
        )
        delta = _vector(correction.get("delta"), f"corrections.{joint_id}.delta")
        mirrored = correction.get("mirrored")
        if not isinstance(mirrored, bool):
            _fail(f"corrections.{joint_id}.mirrored must be boolean")
        if not _same_vector(source_position, joints[joint_id]["position"]):
            _fail(f"corrections.{joint_id}.sourcePosition does not match joint position")
        expected = tuple(target_position[i] - source_position[i] for i in range(3))
        if not _same_vector(delta, expected):
            _fail(f"corrections.{joint_id}.delta does not match targetPosition - sourcePosition")
        corrections[joint_id] = {
            "sourcePosition": list(source_position),
            "targetPosition": list(target_position),
            "delta": list(delta),
            "mirrored": mirrored,
        }

    binding = _validate_binding(root.get("binding"), joints)

    return {
        "schemaVersion": 1,
        "rigProfile": rig_profile,
        "assetFingerprint": fingerprint,
        "coordinateSpace": "armature-local",
        "mirror": {"axis": axis, "origin": origin},
        "joints": joints,
        "corrections": corrections,
        "binding": binding,
    }


def verify_asset(sidecar: Mapping[str, Any], asset_path: Path) -> str:
    actual = asset_fingerprint(asset_path)
    expected = sidecar["assetFingerprint"]
    if actual != expected:
        raise RigSidecarError(
            f"asset fingerprint mismatch: sidecar expects {expected}, uploaded GLB is {actual}"
        )
    return actual


def verify_scene(sidecar: Mapping[str, Any], scene_path: Path) -> str:
    binding = sidecar.get("binding")
    if not binding:
        raise RigSidecarError("Rig sidecar: binding manifest is required for Blender rebind")
    actual = asset_fingerprint(scene_path)
    expected = binding["sceneFingerprint"]
    if actual != expected:
        raise RigSidecarError(
            f"scene fingerprint mismatch: sidecar expects {expected}, uploaded BLEND is {actual}"
        )
    return actual


def plan_corrections(sidecar: Mapping[str, Any]) -> list[JointCorrection]:
    return [
        JointCorrection(
            joint_id=joint_id,
            source_bone=sidecar["joints"][joint_id]["sourceBone"],
            source_position=tuple(correction["sourcePosition"]),
            target_position=tuple(correction["targetPosition"]),
            delta=tuple(correction["delta"]),
            mirrored=correction["mirrored"],
            targets=tuple(
                BoneTarget(
                    bone_id=target["boneId"],
                    bone_name=target["boneName"],
                    endpoint=target["endpoint"],
                )
                for target in (sidecar.get("binding") or {}).get("joints", {})
                .get(joint_id, {}).get("targets", [])
            ),
        )
        for joint_id, correction in sidecar["corrections"].items()
    ]


def _validate_binding(value: Any, joints: Mapping[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    binding = _mapping(value, "binding")
    required = {"adapter", "sceneFingerprint", "metarigObjectId", "joints"}
    if set(binding) != required:
        _fail("binding fields do not match schema v1")
    raw_joints = _mapping(binding["joints"], "binding.joints")
    normalized_joints = {}
    for joint_id, value in raw_joints.items():
        if joint_id not in joints:
            _fail(f"binding.joints.{joint_id} references unknown joint")
        target_group = _mapping(value, f"binding.joints.{joint_id}")
        if set(target_group) != {"targets"}:
            _fail(f"binding.joints.{joint_id} fields do not match schema v1")
        raw_targets = target_group["targets"]
        if not isinstance(raw_targets, list) or not raw_targets:
            _fail(f"binding.joints.{joint_id}.targets must contain at least one target")
        targets = []
        for index, value in enumerate(raw_targets):
            path = f"binding.joints.{joint_id}.targets.{index}"
            target = _mapping(value, path)
            if set(target) != {"boneId", "boneName", "endpoint"}:
                _fail(f"{path} fields do not match schema v1")
            if target["endpoint"] not in {"head", "tail"}:
                _fail(f"{path}.endpoint must be head or tail")
            targets.append({
                "boneId": _string(target["boneId"], f"{path}.boneId"),
                "boneName": _string(target["boneName"], f"{path}.boneName"),
                "endpoint": target["endpoint"],
            })
        normalized_joints[joint_id] = {"targets": targets}
    return {
        "adapter": _string(binding["adapter"], "binding.adapter"),
        "sceneFingerprint": _fingerprint(binding["sceneFingerprint"]),
        "metarigObjectId": _string(binding["metarigObjectId"], "binding.metarigObjectId"),
        "joints": normalized_joints,
    }


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return value


def _allow_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(f"{path} contains unknown fields: {', '.join(unknown)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path} must be a non-empty string")
    return value


def _nullable_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be finite")
    number = float(value)
    if number != number or abs(number) == float("inf"):
        _fail(f"{path} must be finite")
    return number


def _vector(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        _fail(f"{path} must contain exactly three finite numbers")
    return tuple(_number(item, path) for item in value)


def _fingerprint(value: Any) -> str:
    fingerprint = _string(value, "assetFingerprint")
    digest = fingerprint.removeprefix(FINGERPRINT_PREFIX)
    if not fingerprint.startswith(FINGERPRINT_PREFIX) or len(digest) != 64:
        _fail("assetFingerprint must be sha256 followed by 64 lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in digest):
        _fail("assetFingerprint must be sha256 followed by 64 lowercase hexadecimal characters")
    return fingerprint


def _same_vector(left: Any, right: Any) -> bool:
    return all(abs(float(left[index]) - float(right[index])) <= EPSILON for index in range(3))


def _fail(message: str) -> None:
    raise RigSidecarError(f"Rig sidecar: {message}")
