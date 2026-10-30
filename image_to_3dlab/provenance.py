from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from image_to_3dlab.matte import matte_model


@dataclass(frozen=True)
class LicenseProfile:
    classification: str
    folder: str
    license_name: str
    license_url: str
    conditions: tuple[str, ...]
    models: tuple[str, ...]


# Qwen confirmed on 2026-09-21 that outputs are not part of the licensed Materials, so the
# pictures belong to whoever made them. The model runs on the user's own machine and its
# weights are never redistributed by us. The statement is cited so the record stands alone.
QWEN_OUTPUT_STATEMENT = "https://x.com/QwenDevs/status/2101917379785838660"
QWEN_OUTPUT_RIGHTS = (
    "The pictures you generate are yours: Qwen confirmed that outputs are not part of the "
    f"licensed Materials ({QWEN_OUTPUT_STATEMENT}, 21 Sept 2026). The model runs on your own "
    "machine; its weights are not redistributed."
)


# Whichever remover this machine runs (image_to_3dlab/matte.py picks it); both are MIT.
MATTE_LICENSES = {
    "birefnet-general-lite": "MIT (BiRefNet)",
    "u2net": "MIT code / Apache-2.0 U-2-Net",
}

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
        models=("stabilityai/stable-fast-3d",),
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
        models=("tencent/Hunyuan3D-2.1",),
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
        models=("microsoft/TRELLIS.2-4B",),
    ),
    # Same two licences as TRELLIS.2, so the same class: an MIT model with a DINOv3 encoder.
    "pixal3d": LicenseProfile(
        classification="commercial-conditional",
        folder="conditional",
        license_name="MIT (code + flow weights); DINOv3 License (bundled encoder)",
        license_url="https://huggingface.co/raven38/pixal3d-sv-q8_0-v1",
        conditions=(
            "Pixal3D code and flow weights are MIT licensed.",
            "The bundled DINOv3 image encoder is governed by the separate DINOv3 License.",
        ),
        models=("raven38/pixal3d-sv-q8_0-v1",),
    ),
    "qwen-image-2.1": LicenseProfile(
        # Qwen confirmed the outputs are the user's, and the model runs on their own machine;
        # the one condition left is the credit, so it sits with the other conditional models.
        classification="commercial-conditional",
        folder="conditional",
        license_name="Qwen Research License Agreement",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
        conditions=(
            QWEN_OUTPUT_RIGHTS,
            "Attribution: say 'Built with Qwen'.",
        ),
        models=("Qwen/Qwen-Image-2.1",),
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
    model_identity: str | None = None,
    allow_unverified: bool = False,
    commercial_intent: bool = False,
    allow_research_only: bool = True,
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
    if commercial_intent and profile.classification != "commercial-clear":
        raise ValueError(
            f"{backend} is {profile.classification}; commercial_intent does not allow it"
        )
    # research-only is stricter than conditional and is not covered by allow_conditional.
    # Running a non-commercial model is refused for anything but a private showcase,
    # whatever the manifest says. (Its outputs are the user's; running it is the limit.)
    if profile.classification == "research-only" and not allow_research_only:
        raise ValueError(
            f"{backend} is research-only; license_policy.allow_research_only is false"
        )
    if profile.classification == "research-only" and not (
        use_case == "showcase" and distribution == "private"
    ):
        raise ValueError(
            f"{backend} is research-only (non-commercial); it is permitted only for a "
            f"private showcase, not for use_case={use_case!r} distribution={distribution!r}"
        )
    if (
        use_case == "game"
        and distribution == "worldwide"
        and backend == "hunyuan-comfyui"
    ):
        raise ValueError(
            "Hunyuan outputs are not cleared for worldwide game distribution"
        )
    if not allow_unverified and (
        model_identity is None or model_identity not in profile.models
    ):
        if model_identity is None:
            raise ValueError(
                f"{backend} has no reviewed model identity; declare it in a run manifest "
                "with license_policy.allow_unverified: true if you have verified the "
                "licence yourself"
            )
        raise ValueError(
            f"model identity {model_identity!r} is not in the reviewed licence profile for "
            f"{backend}; declare it in a run manifest with license_policy.allow_unverified: true "
            f"if you have verified the licence yourself"
        )
    return profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The in-repo MLX port is MIT. The weights it loads are not.
HUNYUAN_MLX_CODE_COMPONENT = {
    "component": "ZimengXiong/hunyuan3d-mlx (in-repo shape+paint port)",
    "purpose": "MLX generation code tracked in this repository",
    "license": "MIT",
    "license_url": "hunyuan_mlx/LICENSE",
    "commercial_status": "commercial-clear",
}


def _hunyuan_weight_component(component: str, purpose: str, weights: str | None) -> dict[str, Any]:
    """One Tencent weight entry. `weights` is the file or repo this run actually loaded."""
    profile = LICENSES["hunyuan-comfyui"]
    record: dict[str, Any] = {
        "component": component,
        "purpose": purpose,
        "license": profile.license_name,
        "license_url": profile.license_url,
        "commercial_status": "territory-restricted",
        "conditions": list(profile.conditions),
    }
    if weights is not None:
        record["weights"] = weights
    return record


def write_hunyuan_sidecar(
    output: Path,
    image: Path,
    *,
    backend: str,
    parameters: dict[str, Any] | None = None,
    shape: bool = False,
    paint: bool = False,
    shape_weights: str | None = None,
    paint_weights: str | None = None,
    include_code: bool = False,
) -> Path:
    """Write a schema-1 licence sidecar beside a Hunyuan export.

    A shape-only export passes `shape` and leaves `paint` false, so the record
    does not claim a paint model. A paint export names the weights it loaded.
    """
    if not shape and not paint:
        raise ValueError("a Hunyuan export must name the shape or paint weights it loaded")
    output = Path(output)
    image = Path(image)
    profile = LICENSES["hunyuan-comfyui"]
    components: list[dict[str, Any]] = []
    if include_code:
        components.append(dict(HUNYUAN_MLX_CODE_COMPONENT))
    if shape:
        components.append(_hunyuan_weight_component(
            "Tencent Hunyuan3D-2/2.1 shape weights",
            "shape-stage diffusion model",
            shape_weights,
        ))
    if paint:
        components.append(_hunyuan_weight_component(
            "Tencent Hunyuan3D-2/2.1 paint weights (MLX-ready conversion)",
            "paint-stage texture generation model",
            paint_weights,
        ))
    record = {
        "schema_version": 1,
        "backend": backend,
        "input": {
            "path": str(image),
            "sha256": sha256_file(image),
            "source": "user-provided",
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "classification": profile.classification,
        },
        "model": {
            "backend": backend,
            "parameters": parameters or {},
        },
        "license": {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        },
        "components": components,
    }
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return sidecar


def hunyuan_paint_run_dir(outputs_root: Path, stem: str) -> Path:
    """A folder this process owns. Two runs do not share textured_mesh_pbr.glb."""
    safe = stem.replace("/", "_").replace("\\", "_") or "paint"
    return Path(outputs_root) / f"{safe}-{os.getpid()}"


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
    model_identity: str | None = None,
) -> tuple[Path, Path]:
    run_id = uuid.uuid4().hex[:12]
    verified = model_identity is not None and model_identity in profile.models
    classification = profile.classification if verified else "unverified"
    destination_dir = output_root.resolve() / profile.folder
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image.stem}__{backend}__{classification}__{run_id}"
    destination = destination_dir / f"{stem}{generated.suffix.lower()}"
    generated.replace(destination)

    if verified:
        license_record: dict[str, Any] = {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        }
    else:
        license_record = {
            "name": "Unverified — user-declared model",
            "verified": False,
            "conditions": [
                "Verify this model's licence yourself; this run used an identity not in the reviewed profile."
            ],
        }

    model_record: dict[str, Any] = {"backend": backend, "parameters": parameters}
    if model_identity is not None:
        model_record["identity"] = model_identity

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
            "classification": classification,
        },
        "model": model_record,
        "license": license_record,
        "components": (
            [
                {
                    "component": f"rembg/{matte_model()}",
                    "purpose": "background removal",
                    "license": MATTE_LICENSES[matte_model()],
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
