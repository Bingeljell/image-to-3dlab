"""Stand a single prop upright and centre its base, headless.

Run (headless): blender --background --python-exit-code 1 --python scripts/blender_upright.py -- IN.glb OUT.glb

Uses the same bounded tilt search as prop sheets. Irregular shapes may need the
switch turned off. Does not change the heading about the vertical axis.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.blender_split_props import best_tilt, tilt_matrix


def upright_points(points, enabled=True):
    """Undo a prop's camera tilt and ground its base in Blender's Z-up frame."""
    points = np.asarray(points, dtype=float)
    if not enabled:
        return points.copy()
    transform = upright_matrix(points)
    return points @ transform[:3, :3].T + transform[:3, 3]


def upright_matrix(points):
    """One rigid transform, shared by the vertex test and Blender's mesh operation."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('expected finite 3D points')
    rotation = tilt_matrix(*best_tilt(points))
    result = points @ rotation.T
    low, high = result.min(0), result.max(0)
    origin = (low + high) / 2
    origin[2] = low[2]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, -origin
    return transform


def main():
    import argparse

    import bpy
    from mathutils import Matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    if any(obj.type == 'ARMATURE' for obj in bpy.data.objects):
        raise ValueError('stand props upright before rigging')
    objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    points, worlds = [], []
    for obj in objects:
        points.extend([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
        worlds.append(obj.matrix_world.copy())
    correction = Matrix(upright_matrix(points).tolist())
    for obj, world in zip(objects, worlds):
        obj.data = obj.data.copy()
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
        obj.data.transform(correction @ world)
        obj.data.update()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.output), export_format='GLB', use_selection=True)
    from image_to_3dlab.glb_turn import build_glb, parse_glb
    original, _ = parse_glb(args.source.read_bytes())
    document, binary = parse_glb(args.output.read_bytes())
    document.setdefault('asset', {}).update(original.get('asset', {}))
    args.output.write_bytes(build_glb(document, binary))


if __name__ == '__main__':
    main()
