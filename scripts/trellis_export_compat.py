"""Write TRELLIS mesh exports and configure optional packing."""

import inspect
import os
from pathlib import Path


def write_highpoly(path, vertices, faces):
    """Write a little-endian binary PLY with triangle faces."""
    import numpy as np

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    verts = np.ascontiguousarray(vertices, dtype="<f4")
    triangles = np.ascontiguousarray(faces, dtype="<i4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    counts = np.full((len(triangles), 1), 3, dtype="u1")
    rows = np.hstack([counts, triangles.view("u1").reshape(len(triangles), 12)])
    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(verts.tobytes())
        handle.write(rows.tobytes())
    print(f"  high-poly written: {destination} ({len(verts):,} verts, {len(triangles):,} tris)")


def save_decoded_mesh(path, mesh):
    """Save the decoded geometry and material tensors on CPU for a later bake."""
    import torch

    torch.save(
        {
            "vertices": mesh.vertices.cpu(),
            "faces": mesh.faces.cpu(),
            "attrs": mesh.attrs.cpu(),
            "coords": mesh.coords.cpu(),
            "layout": mesh.layout,
            "voxel_size": mesh.voxel_size,
        },
        path,
    )
    print(f"Decode cached: {path} ({mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces)")


def packing_kwargs(to_glb, brute_force):
    """Forward the packing preference when the installed baker supports it."""
    if "xatlas_pack_charts_kwargs" in inspect.signature(to_glb).parameters:
        return {"xatlas_pack_charts_kwargs": {"brute_force": brute_force}}
    if brute_force:
        print("  Warning: this o_voxel cannot forward packing options; "
              "run scripts/patch_ovoxel_pack_options.py to gain ~12% atlas coverage")
    return {}


def save_checkpoint(tag, mesh=None, vertices=None, faces=None):
    """Persist contiguous CPU geometry when I2L_CHECKPOINT_PREFIX is set."""
    prefix = os.environ.get("I2L_CHECKPOINT_PREFIX")
    if not prefix:
        return
    if mesh is not None:
        vertices, faces = mesh.read()
    if vertices is None or faces is None:
        raise RuntimeError(f"checkpoint {tag} has no geometry")

    import torch

    path = Path(f"{prefix}_{tag}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vertices": vertices.detach().cpu().contiguous(),
            "faces": faces.detach().cpu().contiguous(),
        },
        path,
    )
    print(f"[i2l] checkpoint {tag}: {path}", flush=True)
