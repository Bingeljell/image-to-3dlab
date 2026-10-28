"""Voxel-proxy weight transfer used by the headless rig rebind worker.

Run (internal): not run on its own; blender_rebind.py, blender_bind_rig.py and blender_quadruped_pipeline.py import it"""

from __future__ import annotations

from typing import Any


def derived_voxel_size(mesh, fraction: float = 0.012) -> float:
    if not 0.001 <= fraction <= 0.05:
        raise ValueError("voxel fraction must be between 0.001 and 0.05")
    extent = max(float(value) for value in mesh.dimensions)
    if extent <= 0:
        raise RuntimeError(f"mesh {mesh.name} has zero extent")
    return extent * fraction


def largest_component(vertex_count: int, edges) -> set[int]:
    """Indices of the biggest connected island in an undirected vertex graph.

    Ties go to the island containing the lowest vertex index, so the result is stable
    across runs. Isolated vertices count as islands of one.
    """
    adjacency: dict[int, list[int]] = {index: [] for index in range(vertex_count)}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    seen: set[int] = set()
    best: set[int] = set()
    for start in range(vertex_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        island = {start}
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    island.add(neighbour)
                    stack.append(neighbour)
        if len(island) > len(best):
            best = island
    return best


def strip_loose_islands(bmesh, mesh_data) -> dict[str, int]:
    """Reduce a proxy to its largest connected island before bone heat runs.

    Bone heat solves one linear system across the whole surface. A voxel remesh of a
    generated decode routinely leaves a few orphan specks floating off the body -- the
    moss fox produced 13 isolated 8-vertex cubes -- and an island with no bone inside it
    makes that system singular. Blender then reports "failed to find solution for one or
    more bones" and leaves *every* group empty, not just the island's. The proxy is
    throwaway geometry that only ever carries weights, so discarding the specks costs
    nothing.
    """
    before = len(mesh_data.vertices)
    mesh = bmesh.new()
    try:
        mesh.from_mesh(mesh_data)
        mesh.verts.ensure_lookup_table()
        keep = largest_component(
            len(mesh.verts),
            [(edge.verts[0].index, edge.verts[1].index) for edge in mesh.edges],
        )
        doomed = [vert for vert in mesh.verts if vert.index not in keep]
        if doomed:
            bmesh.ops.delete(mesh, geom=doomed, context="VERTS")
            mesh.to_mesh(mesh_data)
            mesh_data.update()
    finally:
        mesh.free()
    return {"before": before, "after": len(mesh_data.vertices),
            "removed": before - len(mesh_data.vertices)}


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
    if len(proxy.data.polygons) == 0:
        raise RuntimeError(f"voxel remesh produced an empty proxy for {mesh.name}")

    import bmesh

    islands = strip_loose_islands(bmesh, proxy.data)
    proxy_faces = len(proxy.data.polygons)
    if proxy_faces == 0:
        raise RuntimeError(f"island cleanup emptied the proxy for {mesh.name}")

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
        "proxyLooseVertsRemoved": islands["removed"],
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
