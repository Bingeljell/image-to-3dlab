"""Bake a normal map from an aligned high-poly mesh in a private scene.

Run (internal): bundled by blender_bake_normals.py.
"""

from __future__ import annotations

import json
import math
import time

if __package__:
    from .blender_scene import clear_private_scene
elif __name__ != "_assetfurnace_worker":
    from blender_scene import clear_private_scene


# glTF is Y-up and Blender is Z-up, so the importer maps (x, y, z) -> (x, -z, y). The
# high-poly PLY never goes through that: it is written straight out of the generator and
# the PLY importer applies no conversion. Left uncorrected the two meshes sit 90 degrees
# apart, which a bounding-box fit cannot repair — it would simply squash one to match the
# other's dimensions and bake nonsense.
AXIS_PERMUTATIONS = {
    "none": (0, 1, 2),
    "gltf_to_blender": (0, 2, 1),
}


def axis_size_ratios(
    high_size: tuple[float, float, float],
    low_size: tuple[float, float, float],
    permutation: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Per-axis size ratio after permuting the high-poly's axes.

    All three land near 1.0 only when the permutation is right, which makes this a
    direct test for orientation rather than a guess.
    """
    permuted = tuple(high_size[i] for i in permutation)
    return tuple(
        (low_size[i] / permuted[i]) if permuted[i] > 1e-9 else 0.0 for i in range(3)
    )


def best_permutation(
    high_size: tuple[float, float, float],
    low_size: tuple[float, float, float],
) -> tuple[str, float]:
    """Pick the axis mapping whose size ratios are most uniform.

    Returns the name and the spread (max ratio / min ratio). A spread near 1.0 means the
    two meshes really are the same shape in that orientation; a large spread means
    neither candidate fits and the bake should not be trusted.
    """
    best_name, best_spread = None, float("inf")
    for name, perm in AXIS_PERMUTATIONS.items():
        ratios = axis_size_ratios(high_size, low_size, perm)
        if min(ratios) <= 0:
            continue
        spread = max(ratios) / min(ratios)
        if spread < best_spread:
            best_name, best_spread = name, spread
    if best_name is None:
        raise ValueError("no usable axis permutation; is one mesh degenerate?")
    return best_name, best_spread


def axis_fit(high_size, low_size) -> tuple[list[float], float, float]:
    """Return the measured ratios, their spread, and the uniform fit scale."""
    ratios = list(axis_size_ratios(high_size, low_size, AXIS_PERMUTATIONS["none"]))
    spread = max(ratios) / min(ratios) if min(ratios) > 0 else 999.0
    return ratios, spread, sum(ratios) / 3.0


def run(arguments: dict[str, object], *, bpy=None, vector=None) -> dict:
    """Bake in a private scene and always restore the user's window."""
    if bpy is None:
        import bpy
    if vector is None:
        from mathutils import Vector as vector

    t0 = time.time()
    name = "IMG3D normal bake"
    window = bpy.context.window
    previous_scene = window.scene
    render_scene = bpy.data.scenes.get(name) or bpy.data.scenes.new(name)
    window.scene = render_scene
    try:
        if render_scene.world is None:
            render_scene.world = bpy.data.worlds.new(render_scene.name + " World")
        clear_private_scene(bpy, render_scene)
        return bake_normals(bpy, vector, render_scene, t0, **arguments)
    finally:
        window.scene = previous_scene


def bbox(o, Vector):
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        for i in range(3):
            lo[i] = min(lo[i], w[i])
            hi[i] = max(hi[i], w[i])
    return lo, hi

def bake_normals(
    bpy, Vector, render_scene, t0, *, low_path, high_path, out_path, size, ray,
    smooth_source, device, max_ray, decimate_to, samples, rot_x,
) -> dict:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=low_path)
    low = [o for o in render_scene.objects if o not in before and o.type == "MESH"][0]  # noqa: RUF015 - preserve missing-mesh error
    low.name = "LOWPOLY"

    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=high_path)
    high = [o for o in render_scene.objects if o not in before and o.type == "MESH"][0]  # noqa: RUF015 - preserve missing-mesh error
    high.name = "HIGHPOLY"
    high_tris_in = len(high.data.polygons)

    # Shade the source smooth. A PLY dumped from the decode carries positions and faces and
    # no vertex normals, so Blender shades it flat and every one of its million-odd facets
    # bakes as a single constant normal. That is not detail, it is confetti: the first bake
    # here came out as rainbow noise over the real relief, with 47% of texels pointing into
    # the surface. Interpolated normals are what the high-poly is for.
    if smooth_source:
        bpy.context.view_layer.objects.active = high
        bpy.ops.object.select_all(action="DESELECT")
        high.select_set(True)
        bpy.ops.object.shade_smooth()

    # Optional decimation of the *bake source*. 6M triangles is slow to ray-cast against;
    # a 1-2M source keeps nearly all the detail a 2048 map can represent.
    if decimate_to and high_tris_in > decimate_to:
        bpy.context.view_layer.objects.active = high
        mod = high.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = decimate_to / high_tris_in
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Orientation first, then fit. A bounding-box fit cannot repair a rotation — it would
    # just squash the high-poly to match the low-poly's dimensions. The permutation is
    # decided outside Blender and verified by per-axis size ratios, which land near 1.0
    # only when the orientation is actually right.
    if rot_x:
        high.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.context.view_layer.update()

    llo, lhi = bbox(low, Vector)
    hlo, hhi = bbox(high, Vector)
    ratios, spread, scale = axis_fit(hhi - hlo, lhi - llo)
    high.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    hlo, hhi = bbox(high, Vector)
    high.location = high.location + ((llo + lhi) / 2 - (hlo + hhi) / 2)
    bpy.context.view_layer.update()
    hlo, hhi = bbox(high, Vector)
    align_err = max(abs((hlo[i] + hhi[i]) / 2 - (llo[i] + lhi[i]) / 2) for i in range(3))

    # The bake target: a non-colour image wired into the low-poly's material.
    img = bpy.data.images.new("NORMAL_BAKE", width=size, height=size, alpha=False,
                              float_buffer=False)
    img.colorspace_settings.name = "Non-Color"

    mat = low.data.materials[0] if low.data.materials else None
    if mat is None:
        mat = bpy.data.materials.new("LowMat")
        low.data.materials.append(mat)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.nodes.active = node

    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    try:
        scn.cycles.device = device
    except Exception:  # noqa: S110 - keep Blender’s default if the device is unavailable.
        pass
    scn.cycles.samples = samples
    scn.render.bake.use_selected_to_active = True
    # Extrusion and distance are different quantities and setting them equal is a trap:
    # rays then start at the cage extrusion outside the surface and stop exactly *at* it, so anything below —
    # every crevice, every recess, which is most of what a normal map is for — is out of
    # reach, and each ray records whatever it grazed on the way in. That is what filled the
    # Snag's first maps with per-island confetti. 0 means no limit, which is the usual setup.
    scn.render.bake.cage_extrusion = ray
    scn.render.bake.max_ray_distance = max_ray
    scn.render.bake.margin = 16
    scn.render.bake.use_clear = True

    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low          # active = bake TARGET

    err = None
    try:
        bpy.ops.object.bake(type="NORMAL")
    except RuntimeError as exc:
        err = str(exc)

    if err is None:
        img.filepath_raw = out_path
        img.file_format = "PNG"
        img.save()

    report = {
        "high_tris_in": high_tris_in,
        "high_tris_baked": len(high.data.polygons),
        "low_tris": len(low.data.polygons),
        "uniform_scale_applied": round(scale, 5),
        "axis_ratios": [round(r, 4) for r in ratios],
        "orientation_spread": round(spread, 4),
        "alignment_error": round(align_err, 5),
        "ray_distance": ray,
        "max_ray_distance": max_ray,
        "smooth_source": smooth_source,
        "device": device,
        "error": err,
        "written": None if err else out_path,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report, indent=2))
    return report
