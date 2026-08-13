"""Pull a clean remeshed surface back onto the original decode. Headless.

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python scripts/blender_shrinkwrap.py -- remeshed.glb decode.glb out.glb

**This is `project_back`.** TRELLIS's `remesh_narrow_band_dc` takes a `project_back`
parameter that snaps rebuilt vertices onto the original surface, which is how the reference
keeps its individual moss fronds after remeshing. Blender's voxel remesh has no equivalent:
it resamples and stops, so detail below the voxel size is gone.

The sweep makes the gap concrete — no single voxel size is both clean and accurate:

    voxel 0.004  volume +0.0067 (control +0.0052)  but cratered
    voxel 0.012  smooth                            but volume +0.0806, 15x inflated

Coarse voxels bridge the gaps between the decode's shards and inflate the form; fine voxels
keep the form and carve pits between shards. Shrinkwrap resolves it: take the coarse mesh's
clean topology, then move every vertex onto the real surface.

`NEAREST_SURFACEPOINT` rather than `PROJECT`: projection casts along a direction and misses
wherever the decode has gaps, dropping vertices through the holes. Nearest-point always
lands somewhere on the surface, which is the safer behaviour on a mesh made of shards.
"""

import sys

import bpy


def parse_args(argv: list[str]) -> tuple[str, str, str, float]:
    """Arguments after `--`: remeshed, decode, output, optional offset."""
    if "--" not in argv:
        raise SystemExit("pass arguments after --")
    rest = argv[argv.index("--") + 1:]
    if len(rest) < 3:
        raise SystemExit("usage: ... -- remeshed.glb decode.glb out.glb [offset]")
    return rest[0], rest[1], rest[2], float(rest[3]) if len(rest) > 3 else 0.0


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def import_largest(path: str, name: str):
    """Import a GLB and return its biggest mesh, joined with any siblings."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    fresh = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not fresh:
        raise SystemExit(f"no mesh in {path}")
    obj = max(fresh, key=lambda o: len(o.data.polygons))
    if len(fresh) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in fresh:
            o.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
    obj.name = name
    return obj


def shrinkwrap(obj, target, offset: float = 0.0):
    """Move every vertex of `obj` onto `target`. Returns the modifier's mode."""
    modifier = obj.modifiers.new(name="i2l_shrinkwrap", type="SHRINKWRAP")
    modifier.target = target
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.offset = offset
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return modifier.wrap_method


def main() -> None:
    remeshed_path, decode_path, target_path, offset = parse_args(sys.argv)
    clear_scene()

    remeshed = import_largest(remeshed_path, "remeshed")
    decode = import_largest(decode_path, "decode")
    print(f"[i2l] remeshed {len(remeshed.data.polygons):,} faces, "
          f"decode {len(decode.data.polygons):,} faces", flush=True)

    shrinkwrap(remeshed, decode, offset)
    print(f"[i2l] shrinkwrapped onto the decode (offset {offset})", flush=True)

    bpy.ops.object.select_all(action="DESELECT")
    remeshed.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=target_path, use_selection=True, export_format="GLB",
        export_materials="NONE", export_normals=True,
    )
    print(f"[i2l] wrote {target_path}", flush=True)


if __name__ == "__main__":
    main()
