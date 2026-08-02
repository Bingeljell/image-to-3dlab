from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrellisOptions:
    repo: Path
    seed: int = 42
    pipeline_type: str = "512"
    texture_size: int = 1024
    bake_target_faces: int = 50_000
    steps: int | None = None


@dataclass(frozen=True)
class TrellisResult:
    asset: Path
    texture_backend: str


def _prepare_rgba(image: Path, destination: Path) -> Path:
    import rembg
    from PIL import Image

    source = Image.open(image).convert("RGBA")
    alpha = source.getchannel("A")
    if alpha.getextrema()[0] == 255:
        source = rembg.remove(source, session=rembg.new_session("u2net"))
    if source.getchannel("A").getextrema()[0] == 255:
        raise RuntimeError(
            "TRELLIS preprocessing did not produce a transparent foreground"
        )
    source.save(destination)
    return destination


def generate_trellis(
    image: Path, output_dir: Path, options: TrellisOptions
) -> TrellisResult:
    repo = options.repo.expanduser().resolve()
    python = repo / ".venv" / "bin" / "python"
    generator = repo / "generate.py"
    if not python.is_file() or not generator.is_file():
        raise RuntimeError(
            f"TRELLIS Mac environment not found at {repo}. "
            "Run scripts/bootstrap_trellis_macos.sh first."
        )
    patched_pipeline = (
        repo / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    )
    if "pipeline.rembg_model = None" not in patched_pipeline.read_text():
        raise RuntimeError("TRELLIS BRIA-disable patch is missing; refusing to run")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / f"{image.stem}_trellis_input.png"
    _prepare_rgba(image, prepared)
    output_base = output_dir / f"{image.stem}_trellis2"
    cpu_basecolor = Path(f"{output_base}_basecolor.png")
    cpu_basecolor.unlink(missing_ok=True)
    command = [
        str(python),
        str(generator),
        str(prepared),
        "--seed",
        str(options.seed),
        "--output",
        str(output_base),
        "--pipeline-type",
        options.pipeline_type,
        "--texture-size",
        str(options.texture_size),
        "--bake-target-faces",
        str(options.bake_target_faces),
    ]
    if options.steps is not None:
        command.extend(("--steps", str(options.steps)))
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    xcode_developer_dir = Path("/Applications/Xcode.app/Contents/Developer")
    if xcode_developer_dir.is_dir():
        env.setdefault("DEVELOPER_DIR", str(xcode_developer_dir))
    try:
        subprocess.run(command, cwd=repo, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"TRELLIS generation failed with exit code {exc.returncode}"
        ) from exc
    result = output_base.with_suffix(".glb")
    if not result.is_file():
        raise RuntimeError(f"TRELLIS reported success but did not create {result}")
    texture_backend = "kdtree-cpu" if cpu_basecolor.is_file() else "metal-o-voxel"
    return TrellisResult(asset=result, texture_backend=texture_backend)
