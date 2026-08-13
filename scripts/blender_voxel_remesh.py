"""Rebuild one closed skin from the decode, using Blender's voxel remesh. Headless.

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python scripts/blender_voxel_remesh.py -- in.glb out.glb 0.004

**The problem this solves.** TRELLIS's decode is not a surface. Every moss frond and fur
tuft is a separate flat shard with gaps between them, so you can see straight through the
creature — from behind the tail you see the face, and the eyeballs show as loose spheres
through the muzzle. Measured: 130,373 connected components, 62,700 of them inside-out, and
a median of 6 ray crossings through the torso where a single skin gives 2.

`to_glb`'s remesh exists to collapse that into one isosurface, which is why the reference
output is solid. On this Mac port that remesh emits a wireframe cage instead. Blender's
voxel remesh is the same class of operation — resample to a volume, extract one surface —
in an implementation that works.

**What it costs.** Voxel remesh is a resampling: fine detail below the voxel size is lost,
and the result is a uniform-density mesh with no UVs, so the texture has to be re-baked
afterwards. That is the trade for getting a closed surface at all.

Voxel size is in the mesh's own units. The decode sits in roughly a unit box, so 0.004 is
about 250 voxels across the widest axis — fine enough to keep the silhouette, coarse enough
to bridge the gaps between shards.
"""

import sys

import bpy


def parse_args(argv: list[str]) -> tuple[str, str, float]:
    """Arguments after Blender's `--` separator: input, output, voxel size."""
    if "--" not in argv:
        raise SystemExit("pass arguments after --")
    rest = argv[argv.index("--") + 1:]
    if len(rest) < 2:
        raise SystemExit("usage: ... -- input.glb output.glb [voxel_size]")
    return rest[0], rest[1], float(rest[2]) if len(rest) > 2 else 0.004


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def voxel_remesh(obj, voxel_size: float):
    """Apply a voxel remesh, returning (faces_before, faces_after).

    `mode='VOXEL'` rebuilds the surface from a signed distance volume, which is precisely
    the operation that turns separated shards into one skin. QuadriFlow — the other remesh
    mode — refuses meshes like this, as recorded in earlier work.
    """
    before = len(obj.data.polygons)
    modifier = obj.modifiers.new(name="i2l_voxel", type="REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = voxel_size
    # Off: adaptivity decimates flat regions, reintroducing uneven density we do not want
    # while we are still judging whether the surface closes at all.
    modifier.adaptivity = 0.0
    modifier.use_smooth_shade = False

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return before, len(obj.data.polygons)


def main() -> None:
    source, target, voxel_size = parse_args(sys.argv)
    clear_scene()

    bpy.ops.import_scene.gltf(filepath=source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh in {source}")
    obj = max(meshes, key=lambda o: len(o.data.polygons))

    # Join the rest in, so shards living in separate objects are remeshed together.
    others = [o for o in meshes if o is not obj]
    if others:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()

    print(f"[i2l] input {len(obj.data.polygons):,} faces, voxel_size {voxel_size}", flush=True)
    before, after = voxel_remesh(obj, voxel_size)
    print(f"[i2l] remeshed {before:,} -> {after:,} faces", flush=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=target, use_selection=True, export_format="GLB",
        export_materials="NONE", export_normals=True,
    )
    print(f"[i2l] wrote {target}", flush=True)


if __name__ == "__main__":
    main()
