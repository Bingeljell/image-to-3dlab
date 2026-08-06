from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .provenance import (
    finalize_output,
    load_run_manifest,
    validate_run_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn one image into a 3D asset with SF3D or ComfyUI/Hunyuan3D."
    )
    parser.add_argument("image", nargs="?", type=Path, help="PNG/JPEG/WebP input image")
    parser.add_argument(
        "--run-manifest", type=Path, help="JSON run manifest (schema v1)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Run SF3D in this process")
    mode.add_argument(
        "--quality",
        action="store_true",
        help="Run an API-format Hunyuan3D ComfyUI workflow",
    )
    mode.add_argument(
        "--trellis", action="store_true", help="Run TRELLIS.2 through the Mac port"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--view",
        dest="views",
        type=Path,
        action="append",
        help=(
            "extra view of the same subject in the same pose (repeatable, trellis2 "
            "only). Views must orbit the camera, not change the pose."
        ),
    )

    fast = parser.add_argument_group("SF3D options")
    fast.add_argument("--sf3d-repo", type=Path, default=Path("vendor/stable-fast-3d"))
    fast.add_argument("--model", default="stabilityai/stable-fast-3d")
    fast.add_argument("--texture-resolution", type=int, default=1024)
    fast.add_argument("--foreground-ratio", type=float, default=0.85)
    fast.add_argument("--remesh", choices=("none", "triangle", "quad"), default="none")
    fast.add_argument("--target-vertices", type=int, default=-1)
    fast.add_argument("--cpu", action="store_true", help="Force SF3D to CPU")
    fast.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Do not retry on CPU after an MPS OOM",
    )

    trellis = parser.add_argument_group("TRELLIS.2 options")
    trellis.add_argument(
        "--trellis-repo", type=Path, default=Path("vendor/trellis-mac")
    )
    trellis.add_argument("--seed", type=int, default=42)
    trellis.add_argument(
        "--pipeline-type", choices=("512", "1024", "1024_cascade"), default="512"
    )
    trellis.add_argument(
        "--trellis-texture-size", choices=(512, 1024, 2048), type=int, default=1024
    )
    trellis.add_argument(
        "--trellis-bake-target-faces",
        type=int,
        default=50_000,
        help="Triangle budget used by the CPU UV/texture fallback",
    )
    trellis.add_argument("--steps", type=int)
    trellis.add_argument(
        "--trellis-raw-material",
        dest="trellis_normalize_material",
        action="store_false",
        help="Keep TRELLIS's raw material instead of normalizing it (leaves it transparent)",
    )
    trellis.add_argument(
        "--trellis-material-mode",
        choices=("matte", "pbr"),
        default="matte",
        help="matte drops metalness (organic subjects); pbr keeps it (brass/chrome)",
    )

    quality = parser.add_argument_group("ComfyUI/Hunyuan3D options")
    quality.add_argument(
        "--workflow", type=Path, help="Workflow exported with ComfyUI Export (API)"
    )
    quality.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    quality.add_argument(
        "--image-node", help="LoadImage node ID; auto-detected when unambiguous"
    )
    quality.add_argument(
        "--output-node", help="Output node ID; otherwise all outputs are searched"
    )
    quality.add_argument(
        "--timeout", type=float, default=3600, help="Job timeout in seconds"
    )
    quality.add_argument("--poll-interval", type=float, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = None
    manifest_data = None
    if args.run_manifest:
        source_manifest = args.run_manifest.expanduser().resolve()
        try:
            manifest_data = load_run_manifest(source_manifest)
            model = manifest_data["model"]
            backend = model["backend"]
            # A manifest may name one image, or several views of the same subject in
            # the same pose. Extra views replace guessing about unseen sides.
            manifest_input = manifest_data["input"]
            if "views" in manifest_input:
                args.views = [
                    source_manifest.parent / view for view in manifest_input["views"]
                ]
                args.image = args.views[0]
            else:
                args.image = source_manifest.parent / manifest_input["path"]
            args.fast = backend == "sf3d"
            args.quality = backend == "hunyuan-comfyui"
            args.trellis = backend == "trellis2"
            if not (args.fast or args.quality or args.trellis):
                raise ValueError(f"backend {backend!r} is not implemented yet")
            parameters = model.get("parameters", {})
            args.model = model.get("id", args.model)
            for key in (
                "texture_resolution",
                "foreground_ratio",
                "remesh",
                "target_vertices",
            ):
                if key in parameters:
                    setattr(args, key, parameters[key])
            for key in ("seed", "pipeline_type", "steps"):
                if key in parameters:
                    setattr(args, key, parameters[key])
            if "texture_size" in parameters:
                args.trellis_texture_size = parameters["texture_size"]
            if "bake_target_faces" in parameters:
                args.trellis_bake_target_faces = parameters["bake_target_faces"]
            if "normalize_material" in parameters:
                args.trellis_normalize_material = bool(parameters["normalize_material"])
            if "material_mode" in parameters:
                args.trellis_material_mode = parameters["material_mode"]
            args.output_dir = Path(
                manifest_data.get("output", {}).get("directory", args.output_dir)
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"error: invalid run manifest: {exc}", file=sys.stderr)
            return 2
    elif args.image is None or sum((args.fast, args.quality, args.trellis)) != 1:
        print(
            "error: provide --run-manifest, or IMAGE with exactly one mode",
            file=sys.stderr,
        )
        return 2

    views = [v.expanduser().resolve() for v in (args.views or [args.image])]
    missing = [str(v) for v in views if not v.is_file()]
    if missing:
        print(f"error: input image does not exist: {', '.join(missing)}", file=sys.stderr)
        return 2
    # Provenance and output naming follow the first view.
    image = views[0]
    if len(views) > 1 and not args.trellis:
        print("error: multiple views are only supported by the trellis2 backend", file=sys.stderr)
        return 2
    intent = {
        "use_case": manifest_data.get("use_case", "showcase")
        if manifest_data
        else "showcase",
        "distribution": manifest_data.get("distribution", "private")
        if manifest_data
        else "private",
        "commercial_intent": manifest_data.get("commercial_intent", False)
        if manifest_data
        else False,
    }
    backend = "sf3d" if args.fast else "trellis2" if args.trellis else "hunyuan-comfyui"
    policy = manifest_data.get("license_policy", {}) if manifest_data else {}
    try:
        profile = validate_run_policy(
            backend,
            intent["use_case"],
            intent["distribution"],
            policy.get("allow_conditional", True),
        )
    except ValueError as exc:
        print(f"error: license policy: {exc}", file=sys.stderr)
        return 2
    working_output_dir = args.output_dir / ".working"
    working_output_dir.mkdir(parents=True, exist_ok=True)
    trellis_texture_backend = None
    trellis_material_normalized = None
    trellis_material_mode = None

    if not 0 < args.foreground_ratio <= 1:
        print("error: --foreground-ratio must be in (0, 1]", file=sys.stderr)
        return 2
    if args.texture_resolution <= 0:
        print("error: --texture-resolution must be positive", file=sys.stderr)
        return 2
    if args.trellis_bake_target_faces <= 0:
        print("error: --trellis-bake-target-faces must be positive", file=sys.stderr)
        return 2

    try:
        if args.fast:
            if args.cpu:
                os.environ["SF3D_USE_CPU"] = "1"
            from .sf3d_backend import SF3DOptions, generate_sf3d

            result = generate_sf3d(
                image,
                working_output_dir,
                SF3DOptions(
                    repo=args.sf3d_repo,
                    model=args.model,
                    texture_resolution=args.texture_resolution,
                    foreground_ratio=args.foreground_ratio,
                    remesh=args.remesh,
                    target_vertices=args.target_vertices,
                    cpu_fallback=not args.no_cpu_fallback,
                ),
            )
        elif args.trellis:
            from .trellis_backend import TrellisOptions, generate_trellis

            trellis_result = generate_trellis(
                views,
                working_output_dir,
                TrellisOptions(
                    repo=args.trellis_repo,
                    seed=args.seed,
                    pipeline_type=args.pipeline_type,
                    texture_size=args.trellis_texture_size,
                    bake_target_faces=args.trellis_bake_target_faces,
                    steps=args.steps,
                    normalize_material=args.trellis_normalize_material,
                    material_mode=args.trellis_material_mode,
                ),
            )
            result = trellis_result.asset
            trellis_texture_backend = trellis_result.texture_backend
            trellis_material_normalized = trellis_result.material_normalized
            trellis_material_mode = trellis_result.material_mode
        else:
            if args.workflow is None:
                raise ValueError(
                    "--quality requires --workflow (ComfyUI API-format JSON)"
                )
            from .comfyui_backend import ComfyUIClient

            result = ComfyUIClient(args.comfy_url).generate(
                image=image,
                workflow_path=args.workflow,
                output_dir=args.output_dir,
                image_node=args.image_node,
                output_node=args.output_node,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parameters = {
        "model": args.model if args.fast or args.trellis else None,
        "texture_resolution": args.texture_resolution if args.fast else None,
        "foreground_ratio": args.foreground_ratio if args.fast else None,
        "remesh": args.remesh if args.fast else None,
        "target_vertices": args.target_vertices if args.fast else None,
        "seed": args.seed if args.trellis else None,
        "pipeline_type": args.pipeline_type if args.trellis else None,
        "texture_size": args.trellis_texture_size if args.trellis else None,
        "bake_target_faces": args.trellis_bake_target_faces if args.trellis else None,
        "texture_backend": trellis_texture_backend if args.trellis else None,
        "material_normalized": trellis_material_normalized if args.trellis else None,
        "material_mode": trellis_material_mode if args.trellis else None,
        "steps": args.steps if args.trellis else None,
    }
    result, sidecar = finalize_output(
        generated=result,
        image=image,
        output_root=args.output_dir,
        backend=backend,
        profile=profile,
        intent=intent,
        parameters=parameters,
        source_manifest=source_manifest,
        backend_repo=(
            args.sf3d_repo if args.fast else args.trellis_repo if args.trellis else None
        ),
    )
    print(result)
    print(sidecar)
    return 0
