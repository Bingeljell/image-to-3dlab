#!/usr/bin/env python3
"""Disable TRELLIS' configured background model for license-controlled runs."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "vendor" / "trellis-mac" / "TRELLIS.2" / "trellis2" / "pipelines"

_LIVE_CONSTRUCTOR = re.compile(r"getattr\(\s*rembg\b")


def patch(path: Path) -> None:
    source = path.read_text()
    needle = (
        "        pipeline.rembg_model = getattr(rembg, "
        "args['rembg_model']['name'])(**args['rembg_model']['args'])"
    )
    replacement = (
        "        # image-to-3dlab disables separately licensed background models.\n"
        "        # Inputs must already contain a non-opaque alpha channel.\n"
        "        pipeline.rembg_model = None"
    )
    if needle in source:
        # A live constructor is replaced even when the note is already present
        # (before or after it); only add the note if it is missing.
        stub = (
            "        pipeline.rembg_model = None"
            if replacement in source
            else replacement
        )
        source = source.replace(needle, stub)
    elif replacement not in source:
        raise RuntimeError(
            f"expected TRELLIS background-model hook not found in {path}"
        )
    live = [
        line
        for line in source.splitlines()
        if not line.lstrip().startswith("#") and _LIVE_CONSTRUCTOR.search(line)
    ]
    if live:
        raise RuntimeError(
            f"live background-model constructor remains in {path}: {live[0].strip()}"
        )
    path.write_text(source)


def patch_cpu_bake_budget(path: Path) -> None:
    source = path.read_text()
    option_anchor = (
        '    parser.add_argument(\n        "--no-texture", action="store_true",\n'
    )
    option = (
        "    parser.add_argument(\n"
        '        "--bake-target-faces", type=int, default=50000,\n'
        '        help="Triangle budget for the CPU UV/texture fallback",\n'
        "    )\n"
    )
    if '"--bake-target-faces"' not in source:
        if option_anchor not in source:
            raise RuntimeError(
                f"expected TRELLIS CLI option anchor not found in {path}"
            )
        source = source.replace(option_anchor, option + option_anchor)

    budget_needle = "            target_faces = min(200000, len(faces))"
    budget_replacement = (
        "            target_faces = min(args.bake_target_faces, len(faces))"
    )
    if budget_needle in source:
        source = source.replace(budget_needle, budget_replacement)
    elif budget_replacement not in source:
        raise RuntimeError(f"expected TRELLIS bake budget hook not found in {path}")

    # The Metal bake path has its own budget line reading `faces_np`, so the CPU
    # replacement above never matched it and every Metal run silently pinned the
    # mesh at 200,000 faces. The 200,000 is a crash guard against mtlbvh failing
    # on large meshes, so it stays as a ceiling — only lower requests are honoured.
    metal_needle = "                target_faces = min(200000, len(faces_np))"
    metal_replacement = (
        "                target_faces = min("
        "args.bake_target_faces, 200000, len(faces_np))"
    )
    if metal_needle in source:
        source = source.replace(metal_needle, metal_replacement)
    elif metal_replacement not in source:
        raise RuntimeError(
            f"expected TRELLIS Metal bake budget hook not found in {path}"
        )
    path.write_text(source)


def main() -> None:
    for filename in ("trellis2_image_to_3d.py", "trellis2_texturing.py"):
        patch(PIPELINES / filename)
    patch_cpu_bake_budget(ROOT / "vendor" / "trellis-mac" / "generate.py")
    print(
        "Patched TRELLIS for pre-matted RGBA inputs and a practical CPU bake budget; "
        "BRIA will not load."
    )


if __name__ == "__main__":
    main()
