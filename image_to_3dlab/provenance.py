from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LicenseProfile:
    classification: str
    folder: str
    license_name: str
    license_url: str
    conditions: tuple[str, ...]


LICENSES = {
    "sf3d": LicenseProfile(
        classification="commercial-conditional",
        folder="conditional",
        license_name="Stability AI Community License",
        license_url="https://huggingface.co/stabilityai/stable-fast-3d/blob/main/LICENSE.md",
        conditions=(
            "Commercial registration required.",
            "Enterprise license required at or above the applicable annual-revenue threshold.",
        ),
    ),
    "hunyuan-comfyui": LicenseProfile(
        classification="territory-restricted",
        folder="conditional",
        license_name="Tencent Hunyuan 3D 2.1 Community License",
        license_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE",
        conditions=(
            "Use and outputs are not licensed in the EU, UK, or South Korea.",
            "Additional distribution, attribution, and commercial conditions apply.",
        ),
    ),
    "trellis2": LicenseProfile(
        classification="commercial-conditional",
        folder="conditional",
        license_name="MIT (TRELLIS.2) + DINOv3 License",
        license_url="https://huggingface.co/microsoft/TRELLIS.2-4B",
        conditions=(
            "TRELLIS.2 model code and weights are MIT licensed.",
            "The DINOv3 image encoder is governed by the separate DINOv3 License.",
            "BRIA RMBG-2.0 is disabled and must not be loaded in commercial runs.",
        ),
    ),
}


def load_run_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("run manifest must be an object with schema_version 1")
    return data


def validate_run_policy(
    backend: str,
    use_case: str,
    distribution: str,
    allow_conditional: bool,
) -> LicenseProfile:
    if use_case not in {"showcase", "game"}:
        raise ValueError("use_case must be 'showcase' or 'game'")
    if distribution not in {"private", "public", "worldwide"}:
        raise ValueError("distribution must be 'private', 'public', or 'worldwide'")
    try:
        profile = LICENSES[backend]
    except KeyError as exc:
        raise ValueError(
            f"backend {backend!r} has no reviewed license profile"
        ) from exc
    if profile.classification != "commercial-clear" and not allow_conditional:
        raise ValueError(
            f"{backend} is {profile.classification}; manifest disallows conditional models"
        )
    if (
        use_case == "game"
        and distribution == "worldwide"
        and backend == "hunyuan-comfyui"
    ):
        raise ValueError(
            "Hunyuan outputs are not cleared for worldwide game distribution"
        )
    return profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pipeline_revision() -> dict[str, Any] | None:
    """This repository's commit, and whether the tree was dirty when the run started.

    Distinct from `backend_revision`, which is the vendored backend checkout. The
    patches in `scripts/patch_trellis_*.py` change what the backend *does* while living
    here, so the backend SHA alone does not identify the code that produced an asset.

    Without this, runs are only comparable by parameters -- and parameters can lie. The
    pangolin generated on 2026-08-02 declares `bake_target_faces: 200000`, but the fix
    that made that value take effect on Metal landed five days later, so the number in
    its sidecar was never what ran.

    `dirty` matters as much as the commit: uncommitted changes mean the SHA does not
    pin the code, and a benchmark built on such a run is not reproducible.
    """
    repo = Path(__file__).resolve().parents[1]
    commit = _git_revision(repo)
    if commit is None:
        return None
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": commit, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("torch", "torchvision", "rembg", "transformers", "trimesh"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def finalize_output(
    generated: Path,
    image: Path,
    output_root: Path,
    backend: str,
    profile: LicenseProfile,
    intent: dict[str, Any],
    parameters: dict[str, Any],
    source_manifest: Path | None,
    backend_repo: Path | None = None,
) -> tuple[Path, Path]:
    run_id = uuid.uuid4().hex[:12]
    destination_dir = output_root.resolve() / profile.folder
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image.stem}__{backend}__{profile.classification}__{run_id}"
    destination = destination_dir / f"{stem}{generated.suffix.lower()}"
    generated.replace(destination)

    record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "input": {
            "path": str(image),
            "sha256": sha256_file(image),
            "source": "user-provided",
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "classification": profile.classification,
        },
        "model": {"backend": backend, "parameters": parameters},
        "license": {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        },
        "components": (
            [
                {
                    "component": "rembg/u2net",
                    "purpose": "background removal",
                    "license": "MIT code / Apache-2.0 U-2-Net",
                    "commercial_status": "commercial-clear",
                }
            ]
            if backend in {"sf3d", "trellis2"}
            else []
        )
        + (
            [
                {
                    "component": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "purpose": "image conditioning",
                    "license": "DINOv3 License",
                    "commercial_status": "commercial-conditional",
                    "license_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
                },
                {
                    "component": "BRIA RMBG-2.0",
                    "purpose": "background removal",
                    "loaded": False,
                    "commercial_status": "blocked",
                    "note": "Explicitly disabled by repository patch.",
                },
            ]
            if backend == "trellis2"
            else []
        ),
        "software": {
            "packages": _package_versions(),
            "backend_revision": _git_revision(backend_repo) if backend_repo else None,
            "pipeline_revision": _pipeline_revision(),
        },
        "source_manifest": {
            "path": str(source_manifest.resolve()),
            "sha256": sha256_file(source_manifest),
        }
        if source_manifest
        else None,
    }
    sidecar = destination.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return destination, sidecar
