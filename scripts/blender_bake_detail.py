"""Bake the original's surface detail onto a finished mesh: normal map + metallic-roughness.

Run with:
    blender --background --python-exit-code 1 --python scripts/blender_bake_detail.py -- \
        SOURCE.glb TARGET.glb OUT.glb [size]

SOURCE is the generated high-poly asset, TARGET the retopologised (or repainted) mesh
with its own UVs, OUT the target with the new maps wired into its material.

**Why.** Retopology keeps the shape but throws away the fine relief (muscle, rivets, belt
buckles) and the base-colour-only transfer drops the source's metallic-roughness map, so
skin and leather pick up a flat metallic sheen and metal loses its contrast. Measured on
the orc, 2026-09-27: 942,928 -> 19,974 triangles looked soft and washed out; with these
two maps baked it reads close to the original.

**Why a separate stage, after the repaint.** The Hunyuan repaint re-unwraps the mesh, so a
map baked onto the retopo's UVs would land in the wrong place on the painted asset. This
bakes onto whatever mesh is current, and leaves a repaint's own metallic-roughness alone:
that one matches the new colours.

**Alignment is checked, not assumed.** Both files go through the same glTF importer, so
orientation agrees; scale and position are fitted from the bounding boxes. If the
per-axis size ratios disagree (a rotated or different mesh), the bake is refused: the
first orc attempt baked across a 2.06 spread and produced garbage with no error.

**Normal sign.** Generated meshes are inconsistently wound, so some rays come back
reversed. The fix is `resolve_tangent_sign` from `blender_bake_normals.py`, imported, not
copied.

**Tangents ship with the map.** A tangent-space normal map only reads right against the
tangents it was baked with. Left out, every engine generates its own, and the glTF
validator warns on every file that those may not match; exported, it reports nothing.

Headless on purpose: a Cycles bake through the live Blender socket blocks the GUI and
clobbers the open scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blender_bake_normals import axis_size_ratios, resolve_tangent_sign
from compress_glb_textures import build_glb, parse_glb

MIN_SIZE, MAX_SIZE = 256, 8192
# How far the three axis ratios may disagree before the two meshes are treated as not the
# same shape. The aligned orc read 1.0049; a 90-degree turn reads well above 1.4.
MAX_SPREAD = 1.05
# Ray reach as a fraction of the asset's largest dimension. Same as the retopo transfer:
# short enough not to hit the far side of a limb, long enough to find the old surface.
RAY_FRACTION = 0.02
FLOAT = 5126


class Args(NamedTuple):
    source: str
    target: str
    output: str
    size: int


class Fit(NamedTuple):
    ok: bool
    scale: float
    spread: float


def parse_args(argv: list[str]) -> Args:
    """Positional arguments after Blender's `--`."""
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(rest) < 3:
        raise SystemExit("usage: blender --background --python-exit-code 1 --python "
                         "scripts/blender_bake_detail.py "
                         "-- SOURCE.glb TARGET.glb OUT.glb [size]")
    size = int(rest[3]) if len(rest) > 3 else 2048
    if not MIN_SIZE <= size <= MAX_SIZE:
        raise SystemExit(f"size must be within {MIN_SIZE}..{MAX_SIZE}, got {size}")
    return Args(rest[0], rest[1], rest[2], size)


def fit(high_size: tuple[float, float, float], low_size: tuple[float, float, float]) -> Fit:
    """Uniform scale that maps the source onto the target, and whether to trust it."""
    ratios = axis_size_ratios(high_size, low_size, (0, 1, 2))
    if min(ratios) <= 0:
        return Fit(False, 0.0, float("inf"))
    spread = max(ratios) / min(ratios)
    return Fit(spread <= MAX_SPREAD, sum(ratios) / 3.0, spread)


def transfer_metallic_roughness(source_has_map: bool, target_has_map: bool) -> bool:
    """Bake the source's map across only onto a target that has none of its own."""
    return source_has_map and not target_has_map


def ray_reach(dimensions: tuple[float, float, float]) -> float:
    return max(dimensions) * RAY_FRACTION


def repair_zero_tangents(glb: bytes) -> tuple[bytes, int]:
    """Give every zero-length tangent in a GLB a unit direction across its normal.

    Blender's exporter leaves a few tangents at zero length: on a test prop sheet, one or
    two vertices in 3 files of 27, each on a lone triangle that is a UV island of its own.
    The glTF validator counts each one as an error, and gltfpack already repairs them in
    the files it writes, so this does the same: any unit direction perpendicular to the
    normal, handedness positive. One triangle's shading is all it can affect.
    """
    import numpy as np

    document, chunk = parse_glb(glb)
    if not chunk:
        return glb, 0
    binary = bytearray(chunk)

    def view(index: int, width: int) -> np.ndarray:
        accessor = document["accessors"][index]
        if accessor["componentType"] != FLOAT:
            raise ValueError("expected float normals and tangents")
        buffer_view = document["bufferViews"][accessor["bufferView"]]
        offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = buffer_view.get("byteStride", 4 * width)
        return np.ndarray((accessor["count"], width), dtype="<f4", buffer=binary,
                          offset=offset, strides=(stride, 4))

    repaired = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "TANGENT" not in attributes or "NORMAL" not in attributes:
                continue
            tangents = view(attributes["TANGENT"], 4)
            normals = view(attributes["NORMAL"], 3)
            # Written as "not long enough" so a NaN tangent counts as broken too.
            zero = ~(np.linalg.norm(tangents[:, :3], axis=1) >= 1e-6)
            for i in np.flatnonzero(zero):
                normal = normals[i].astype(np.float64)
                helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                tangent = np.cross(normal, helper)
                length = np.linalg.norm(tangent)
                # A zero normal has no "across"; any unit vector beats a NaN.
                tangents[i] = (*(tangent / length if length > 1e-12 else helper), 1.0)
            repaired += int(zero.sum())
    if not repaired:
        return glb, 0
    return build_glb(document, bytes(binary)), repaired


def partial_path(final: Path) -> Path:
    """Where the exporter writes, beside OUT, until its tangents are repaired."""
    return final.with_name(f"{final.stem}.partial.glb")


def finish_export(partial: Path, final: Path) -> int:
    """Repair the exported GLB's tangents and only then put it at `final`.

    Finish's --resume trusts any OUT that is not empty, so a bake cut short between the
    export and the repair must leave no OUT behind. Returns how many were repaired.
    """
    glb, repaired = repair_zero_tangents(partial.read_bytes())
    if repaired:
        partial.write_bytes(glb)
    partial.replace(final)
    return repaired


# --- Blender side (thin) ---------------------------------------------------------------

def _mr_image(material):
    """The image feeding Metallic/Roughness, following the importer's Separate Color."""
    if material is None or not material.use_nodes:
        return None
    bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None
    for socket in ("Metallic", "Roughness"):
        for link in bsdf.inputs[socket].links:
            node = link.from_node
            while node.type != "TEX_IMAGE":
                if not node.inputs or not node.inputs[0].links:
                    break
                node = node.inputs[0].links[0].from_node
            if node.type == "TEX_IMAGE" and node.image:
                return node.image
    return None


def _import(bpy, path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def _bbox(objects):
    from mathutils import Vector
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], world[i])
                hi[i] = max(hi[i], world[i])
    return lo, hi


def _target_image(bpy, target, name, size):
    """A non-colour image set as the active bake node in every material on the target."""
    image = bpy.data.images.new(name, width=size, height=size, alpha=False)
    image.colorspace_settings.name = "Non-Color"
    nodes = []
    for material in target.data.materials:
        tree = material.node_tree
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        for other in tree.nodes:
            other.select = False
        node.select = True
        tree.nodes.active = node
        nodes.append((material, node))
    return image, nodes


def _bake(bpy, sources, target, kind):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in sources:
        obj.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.bake(type=kind)


def main() -> int:
    import bpy
    import numpy as np
    from mathutils import Matrix

    args = parse_args(list(sys.argv))
    bpy.ops.wm.read_factory_settings(use_empty=True)

    sources = _import(bpy, args.source)
    targets = _import(bpy, args.target)
    if len(targets) != 1:
        raise SystemExit(f"expected one mesh in {args.target}, found {len(targets)}")
    target = targets[0]
    if not target.data.uv_layers:
        raise SystemExit(f"{args.target} has no UVs to bake onto")
    if not target.data.materials:
        target.data.materials.append(bpy.data.materials.new("BAKE_MAT"))
    for material in target.data.materials:
        material.use_nodes = True

    hlo, hhi = _bbox(sources)
    llo, lhi = _bbox([target])
    shape = fit(tuple(hhi - hlo), tuple(lhi - llo))
    if not shape.ok:
        raise SystemExit(f"source and target do not line up (axis spread {shape.spread:.3f}, "
                         f"limit {MAX_SPREAD}); refusing to bake garbage")
    offset = (llo + lhi) / 2 - (hlo + hhi) / 2 * shape.scale
    move = Matrix.Translation(offset) @ Matrix.Scale(shape.scale, 4)
    for obj in sources:
        obj.matrix_world = move @ obj.matrix_world
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.margin = 16
    bake.use_clear = True
    bake.cage_extrusion = ray_reach(tuple(target.dimensions))
    bake.max_ray_distance = bake.cage_extrusion * 2.0
    bake.normal_space = "TANGENT"

    # --- normal map ------------------------------------------------------------------
    normal_img, normal_nodes = _target_image(bpy, target, "detail_normal", args.size)
    _bake(bpy, sources, target, "NORMAL")
    pixels = np.empty(args.size * args.size * 4, dtype=np.float32)
    normal_img.pixels.foreach_get(pixels)
    rgba = pixels.reshape(args.size, args.size, 4)
    fixed, flipped = resolve_tangent_sign((rgba[..., :3] * 255.0).round())
    rgba[..., :3] = fixed.astype(np.float32) / 255.0
    normal_img.pixels.foreach_set(rgba.ravel())
    normal_img.pack()

    # --- metallic-roughness ------------------------------------------------------------
    source_maps = {id(m): _mr_image(m) for o in sources for m in o.data.materials if m}
    target_has_map = any(_mr_image(m) for m in target.data.materials)
    moved_mr = transfer_metallic_roughness(any(source_maps.values()), target_has_map)
    mr_nodes = []
    if moved_mr:
        for obj in sources:
            for slot in obj.material_slots:
                original = slot.material
                emit = bpy.data.materials.new("EMIT_MR")
                emit.use_nodes = True
                nodes, links = emit.node_tree.nodes, emit.node_tree.links
                for node in list(nodes):
                    if node.type != "OUTPUT_MATERIAL":
                        nodes.remove(node)
                output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
                emission = nodes.new("ShaderNodeEmission")
                image = source_maps.get(id(original)) if original else None
                if image is not None:
                    tex = nodes.new("ShaderNodeTexImage")
                    tex.image = image
                    links.new(tex.outputs["Color"], emission.inputs["Color"])
                else:
                    # No map on this part: bake its flat factors, in glTF's G=rough B=metal.
                    bsdf = next((n for n in original.node_tree.nodes
                                 if n.type == "BSDF_PRINCIPLED"), None) if original else None
                    rough = bsdf.inputs["Roughness"].default_value if bsdf else 0.5
                    metal = bsdf.inputs["Metallic"].default_value if bsdf else 0.0
                    emission.inputs["Color"].default_value = (0.0, rough, metal, 1.0)
                links.new(emission.outputs["Emission"], output.inputs["Surface"])
                slot.material = emit
        for material, node in normal_nodes:
            node.select = False
        mr_img, mr_nodes = _target_image(bpy, target, "detail_mr", args.size)
        _bake(bpy, sources, target, "EMIT")
        mr_img.pack()

    # --- wire the maps into the target's material(s) -----------------------------------
    for material, node in normal_nodes:
        tree = material.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.space = "TANGENT"
        tree.links.new(node.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    for material, node in mr_nodes:
        tree = material.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        split = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(node.outputs["Color"], split.inputs["Color"])
        tree.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
        tree.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])

    for obj in sources:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    partial = partial_path(Path(args.output))
    bpy.ops.export_scene.gltf(filepath=str(partial), export_format="GLB", use_selection=True,
                              export_tangents=True)
    repaired = finish_export(partial, Path(args.output))

    print("BAKE_DETAIL::" + json.dumps({
        "source": args.source, "target": args.target, "output": args.output,
        "size": args.size, "scale": round(shape.scale, 5), "spread": round(shape.spread, 4),
        "normal_flipped_fraction": round(flipped, 4),
        "zero_tangents_repaired": repaired,
        "metallic_roughness": "transferred" if moved_mr else (
            "kept target's own" if target_has_map else "none on source"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
