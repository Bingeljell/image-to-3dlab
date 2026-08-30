"""Voxel-proxy weight transfer used by the headless rig rebind worker."""

from __future__ import annotations

from typing import Any


def derived_voxel_size(mesh, fraction: float = 0.012) -> float:
    if not 0.001 <= fraction <= 0.05:
        raise ValueError("voxel fraction must be between 0.001 and 0.05")
    extent = max(float(value) for value in mesh.dimensions)
    if extent <= 0:
        raise RuntimeError(f"mesh {mesh.name} has zero extent")
    return extent * fraction


def transfer_weights(bpy, mesh, rig, *, voxel_fraction: float = 0.012,
                     weight_threshold: float = 0.001) -> dict[str, Any]:
    """Reweight one mesh through a throwaway watertight proxy and return diagnostics."""
    voxel_size = derived_voxel_size(mesh, voxel_fraction)
    stale = [obj for obj in bpy.data.objects if obj.name == f"I2L_WEIGHT_PROXY_{mesh.name}"]
    for obj in stale:
        bpy.data.objects.remove(obj, do_unlink=True)

    world_matrix = mesh.matrix_world.copy()
    proxy = mesh.copy()
    proxy.data = mesh.data.copy()
    proxy.name = f"I2L_WEIGHT_PROXY_{mesh.name}"
    proxy.parent = None
    proxy.matrix_world = world_matrix
    bpy.context.scene.collection.objects.link(proxy)
    for modifier in list(proxy.modifiers):
        proxy.modifiers.remove(modifier)
    for group in list(proxy.vertex_groups):
        proxy.vertex_groups.remove(group)

    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    modifier = proxy.modifiers.new("I2L voxel proxy", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    proxy_faces = len(proxy.data.polygons)
    if proxy_faces == 0:
        raise RuntimeError(f"voxel remesh produced an empty proxy for {mesh.name}")

    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        raise RuntimeError(f"voxel proxy automatic weighting failed for {mesh.name}: {exc}") from exc
    weighted_proxy_groups = _nonempty_groups(proxy, weight_threshold)
    if not weighted_proxy_groups:
        raise RuntimeError(f"voxel proxy produced no non-empty weight groups for {mesh.name}")

    for group in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(group)
    for modifier in list(mesh.modifiers):
        if modifier.type in {"ARMATURE", "DATA_TRANSFER"}:
            mesh.modifiers.remove(modifier)
    mesh.parent = None
    mesh.matrix_world = world_matrix

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    transfer = mesh.modifiers.new("I2L weight transfer", "DATA_TRANSFER")
    transfer.object = proxy
    transfer.use_vert_data = True
    transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    transfer.vert_mapping = "POLYINTERP_NEAREST"
    bpy.ops.object.datalayout_transfer(modifier=transfer.name)
    bpy.ops.object.modifier_apply(modifier=transfer.name)

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_NAME")
    mesh_groups = _nonempty_groups(mesh, weight_threshold)
    unweighted = _unweighted_vertices(mesh, weight_threshold)
    bpy.data.objects.remove(proxy, do_unlink=True)
    if not mesh_groups:
        raise RuntimeError(f"weight transfer produced no non-empty groups for {mesh.name}")
    if unweighted == len(mesh.data.vertices):
        raise RuntimeError(f"weight transfer left every vertex unweighted for {mesh.name}")
    return {
        "mesh": mesh.name,
        "voxelSize": voxel_size,
        "proxyFaces": proxy_faces,
        "proxyGroups": len(weighted_proxy_groups),
        "meshGroups": len(mesh_groups),
        "unweightedVertices": unweighted,
        "totalVertices": len(mesh.data.vertices),
    }


def _nonempty_groups(mesh, threshold: float) -> set[str]:
    names = {group.index: group.name for group in mesh.vertex_groups}
    return {
        names[weight.group]
        for vertex in mesh.data.vertices
        for weight in vertex.groups
        if weight.weight >= threshold and weight.group in names
    }


def _unweighted_vertices(mesh, threshold: float) -> int:
    return sum(
        1 for vertex in mesh.data.vertices
        if not any(weight.weight >= threshold for weight in vertex.groups)
    )
