from __future__ import annotations

import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

_GLB_MAGIC = 0x46546C67
_GLB_JSON_CHUNK = 0x4E4F534A


@dataclass(frozen=True)
class TrellisOptions:
    repo: Path
    seed: int = 42
    pipeline_type: str = "512"
    texture_size: int = 1024
    bake_target_faces: int = 50_000
    steps: int | None = None
    normalize_material: bool = True
    material_mode: str = "matte"


@dataclass(frozen=True)
class TrellisResult:
    asset: Path
    texture_backend: str
    material_normalized: bool = False
    material_mode: str | None = None


def _normalize_glb_material(path: Path, mode: str = "matte") -> int:
    """Rewrite a GLB's material JSON so it renders correctly.

    TRELLIS exports each material with ``alphaMode=BLEND`` and ``metallicFactor=1``
    plus a metallic-roughness texture. Together these make a dense mesh render as
    transparent, mirror-like shards instead of the baked albedo. We patch only the
    JSON chunk in place; geometry and texture buffers are left byte-for-byte intact.

    Both modes force ``alphaMode`` to ``OPAQUE`` (the actual cause of the glass
    look). They then differ in how they treat metalness:

    - ``matte`` (default): drop all metalness (``metallicFactor`` 0, matte
      roughness, metallic-roughness texture removed). Best for organic subjects
      whose shading is already baked into the albedo (e.g. foliage, fur).
    - ``pbr``: keep the baked metallic-roughness so genuinely metallic subjects
      (brass, chrome) keep their sheen under environment lighting.

    Returns the number of material properties changed.
    """
    if mode not in {"matte", "pbr"}:
        raise ValueError(f"material mode must be 'matte' or 'pbr', got {mode!r}")
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise RuntimeError(f"{path} is not a binary glTF (GLB) file")
    offset = 12
    chunks: list[list] = []
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunks.append([chunk_type, data[offset + 8 : offset + 8 + chunk_len]])
        offset += 8 + chunk_len
    try:
        json_index = next(i for i, c in enumerate(chunks) if c[0] == _GLB_JSON_CHUNK)
    except StopIteration as exc:
        raise RuntimeError(f"{path} has no glTF JSON chunk") from exc

    gltf = json.loads(chunks[json_index][1].decode("utf-8"))
    changed = 0
    for material in gltf.get("materials", []):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        if material.get("alphaMode", "OPAQUE") != "OPAQUE":
            material["alphaMode"] = "OPAQUE"
            changed += 1
        material.pop("alphaCutoff", None)
        if mode == "matte":
            if pbr.get("metallicFactor") != 0.0:
                pbr["metallicFactor"] = 0.0
                changed += 1
            pbr["roughnessFactor"] = 1.0
            if pbr.pop("metallicRoughnessTexture", None) is not None:
                changed += 1
        # "pbr" mode keeps metalness/roughness and the metallic-roughness texture.
    chunks[json_index][1] = json.dumps(gltf, separators=(",", ":")).encode("utf-8")

    body = b""
    for chunk_type, chunk_data in chunks:
        pad = (4 - (len(chunk_data) % 4)) % 4
        filler = b"\x20" if chunk_type == _GLB_JSON_CHUNK else b"\x00"
        chunk_data = chunk_data + filler * pad
        body += struct.pack("<II", len(chunk_data), chunk_type) + chunk_data
    path.write_bytes(struct.pack("<III", _GLB_MAGIC, version, 12 + len(body)) + body)
    return changed


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
    material_normalized = False
    material_mode = None
    if options.normalize_material:
        _normalize_glb_material(result, options.material_mode)
        material_normalized = True
        material_mode = options.material_mode
    return TrellisResult(
        asset=result,
        texture_backend=texture_backend,
        material_normalized=material_normalized,
        material_mode=material_mode,
    )
