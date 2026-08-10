"""Give a generated mesh's zero-thickness sheets real thickness, headless.

Run with:
    blender --background --python scripts/blender_solidify.py -- IN.glb OUT.glb [thickness]

TRELLIS renders fine detail -- fur, armour plates, carved relief, woven coils -- as
surfaces with no back face. They are invisible from behind, so backface culling makes
the model see-through, and a sprite bake with a transparent film turns them into actual
alpha. These are not tears; nothing is missing. They need thickness, not patching.

Measured before -> after, as boundary-loop perimeter over the mesh diagonal:
clockwork pangolin 97.82 -> 0.00, moss fox 126.58 -> 0.00, basalt monolith 44.78 -> 0.00.
Roughly 4x the faces, which is irrelevant for a sprite bake and worth decimating for a
real-time asset.

`offset = 0.0` grows thickness both ways from each sheet rather than along the normal.
That matters: winding is inconsistent on these meshes (docs/open-questions.md question
2), so extruding along the normal would push some plates inward and some outward.

Headless on purpose. Driving this through the live Blender socket blocks the GUI's
handler, which locks up the user's session and loses the reply.
"""

from __future__ import annotations

import sys

import bpy


def solidify(path_in: str, path_out: str, fraction: float) -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=path_in)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh in {path_in}")

    for obj in meshes:
        before = len(obj.data.polygons)
        modifier = obj.modifiers.new("Solidify", "SOLIDIFY")
        # Thickness as a fraction of the asset's largest dimension, so the same number
        # behaves the same on any subject regardless of its scale.
        modifier.thickness = fraction * max(obj.dimensions)
        modifier.offset = 0.0
        modifier.use_rim = True
        modifier.use_rim_only = False
        # Read the thickness before applying: afterwards the modifier is gone and the
        # reference reads back as zero, which makes the log claim nothing happened.
        applied = modifier.thickness
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Solidify")
        print(f"SOLIDIFY:: {obj.name} {before} -> {len(obj.data.polygons)} faces "
              f"(thickness {applied:.5f})")

    for obj in bpy.data.objects:
        obj.select_set(obj.type == "MESH")
    bpy.ops.export_scene.gltf(filepath=path_out, use_selection=True,
                              export_format="GLB", export_yup=True)
    print(f"SOLIDIFY:: wrote {path_out}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    path_in, path_out = argv[0], argv[1]
    fraction = float(argv[2]) if len(argv) > 2 else 0.004
    solidify(path_in, path_out, fraction)


if __name__ == "__main__":
    main()
