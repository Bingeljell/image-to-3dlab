#!/usr/bin/env python3
"""Disable TRELLIS' configured background model for license-controlled runs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "vendor" / "trellis-mac" / "TRELLIS.2" / "trellis2" / "pipelines"


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
    if replacement in source:
        return
    if needle not in source:
        raise RuntimeError(
            f"expected TRELLIS background-model hook not found in {path}"
        )
    path.write_text(source.replace(needle, replacement))


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
    path.write_text(source)


for filename in ("trellis2_image_to_3d.py", "trellis2_texturing.py"):
    patch(PIPELINES / filename)

patch_cpu_bake_budget(ROOT / "vendor" / "trellis-mac" / "generate.py")

print(
    "Patched TRELLIS for pre-matted RGBA inputs and a practical CPU bake budget; "
    "BRIA will not load."
)
