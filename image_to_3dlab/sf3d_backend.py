from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SF3DOptions:
    repo: Path
    model: str
    texture_resolution: int = 1024
    foreground_ratio: float = 0.85
    remesh: str = "none"
    target_vertices: int = -1
    cpu_fallback: bool = True


def _is_mps_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "mps" in message and any(
        word in message for word in ("out of memory", "oom", "allocation")
    )


def _load_sf3d(repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / "sf3d" / "system.py").is_file():
        raise RuntimeError(
            f"SF3D checkout not found at {repo}. Run scripts/bootstrap_macos.sh first, "
            "or pass --sf3d-repo."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from sf3d.system import SF3D
        from sf3d.utils import remove_background, resize_foreground
    except ImportError as exc:
        raise RuntimeError(
            "SF3D or a compiled extension could not be imported. Run scripts/bootstrap_macos.sh. "
            f"Original error: {exc}"
        ) from exc
    return SF3D, remove_background, resize_foreground


def _run(
    image_path: Path, output_path: Path, options: SF3DOptions, device: str
) -> Path:
    import rembg
    import torch
    from PIL import Image

    SF3D, remove_background, resize_foreground = _load_sf3d(options.repo)
    model = SF3D.from_pretrained(
        options.model, config_name="config.yaml", weight_name="model.safetensors"
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGBA")
    image = remove_background(image, rembg.new_session())
    image = resize_foreground(image, options.foreground_ratio)
    prepared = output_path.with_name(f"{output_path.stem}_input.png")
    image.save(prepared)

    try:
        with torch.inference_mode():
            mesh, _ = model.run_image(
                [image],
                bake_resolution=options.texture_resolution,
                remesh=options.remesh,
                vertex_count=options.target_vertices,
            )
        mesh.export(str(output_path), include_normals=True)
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return output_path


def generate_sf3d(image: Path, output_dir: Path, options: SF3DOptions) -> Path:
    import torch

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    forced_cpu = os.environ.get("SF3D_USE_CPU") == "1"
    device = "cpu" if forced_cpu or not torch.backends.mps.is_available() else "mps"
    output_path = output_dir.resolve() / f"{image.stem}_sf3d.glb"
    try:
        return _run(image, output_path, options, device)
    except RuntimeError as exc:
        if device != "mps" or not options.cpu_fallback or not _is_mps_oom(exc):
            raise
        print("MPS ran out of memory; retrying SF3D on CPU.", file=sys.stderr)
        os.environ["SF3D_USE_CPU"] = "1"
        torch.mps.empty_cache()
        gc.collect()
        return _run(image, output_path, options, "cpu")
