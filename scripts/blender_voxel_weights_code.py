"""The Blender-side program for `blender_voxel_weights.py`.

Kept in its own module so the (long) generated source is not buried inside an f-string
in the CLI, which makes both halves hard to read and hard to lint.
"""

from __future__ import annotations

TEMPLATE = '''
import bpy, json

VOXEL = {voxel_size}
THRESH = {weight_threshold}
KEEP_PROXY = {keep_proxy}

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("PROXY_")]
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if len(mesh_objs) != 1 or len(arms) != 1:
    raise RuntimeError("need exactly one mesh and one armature, found %d and %d"
                       % (len(mesh_objs), len(arms)))
mesh, arm = mesh_objs[0], arms[0]
mesh.hide_select = False

for stale in [o for o in bpy.data.objects if o.name.startswith("PROXY_")]:
    bpy.data.objects.remove(stale, do_unlink=True)

# --- 1. voxel proxy -------------------------------------------------------------
proxy = mesh.copy()
proxy.data = mesh.data.copy()
proxy.name = "PROXY_weights"
proxy.parent = None
bpy.context.scene.collection.objects.link(proxy)
for mod in list(proxy.modifiers):
    proxy.modifiers.remove(mod)
for vg in list(proxy.vertex_groups):
    proxy.vertex_groups.remove(vg)

bpy.ops.object.select_all(action="DESELECT")
proxy.select_set(True)
bpy.context.view_layer.objects.active = proxy
rem = proxy.modifiers.new("Remesh", "REMESH")
rem.mode = "VOXEL"
rem.voxel_size = VOXEL
bpy.ops.object.modifier_apply(modifier=rem.name)
proxy_faces = len(proxy.data.polygons)

# --- 2. heat-weight the proxy ---------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
proxy.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
proxy_error = None
try:
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
except RuntimeError as exc:
    proxy_error = str(exc)

proxy_nonempty = 0
for vg in proxy.vertex_groups:
    for v in proxy.data.vertices:
        if any(g.group == vg.index and g.weight >= THRESH for g in v.groups):
            proxy_nonempty += 1
            break

# --- 3. transfer weights proxy -> real mesh -------------------------------------
for vg in list(mesh.vertex_groups):
    mesh.vertex_groups.remove(vg)
for mod in list(mesh.modifiers):
    if mod.type in {{"ARMATURE", "DATA_TRANSFER"}}:
        mesh.modifiers.remove(mod)
mesh.parent = None

bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
dt = mesh.modifiers.new("WeightTransfer", "DATA_TRANSFER")
dt.object = proxy
dt.use_vert_data = True
dt.data_types_verts = {{"VGROUP_WEIGHTS"}}
dt.vert_mapping = "POLYINTERP_NEAREST"
bpy.ops.object.datalayout_transfer(modifier=dt.name)
bpy.ops.object.modifier_apply(modifier=dt.name)

# --- 4. bind the real mesh with the transferred groups --------------------------
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_NAME")

if not KEEP_PROXY:
    bpy.data.objects.remove(proxy, do_unlink=True)

# --- 5. verification ------------------------------------------------------------
markers = {{o.name[6:]: o.matrix_world.translation.copy()
           for o in bpy.data.objects if o.name.startswith("JOINT_")}}
groups = {{vg.index: vg.name for vg in mesh.vertex_groups}}
zs = [v.co.z for v in mesh.data.vertices]
span = max(zs) - min(zs)
head_z = markers["head"].z if "head" in markers else max(zs)

influence = {{}}
for v in mesh.data.vertices:
    for g in v.groups:
        if g.weight >= THRESH:
            name = groups.get(g.group)
            if name:
                rec = influence.setdefault(name, [0, -1e9])
                rec[0] += 1
                rec[1] = max(rec[1], v.co.z)

report = {{}}
for name in groups.values():
    rec = influence.get(name)
    report[name] = {{
        "verts": rec[0] if rec else 0,
        "max_z": round(rec[1], 4) if rec else None,
        "reaches_head": bool(rec and rec[1] > head_z - span * 0.05),
    }}

legs = [n for n in report if "front" in n or "back" in n]
contaminated = sorted(n for n in legs if report[n]["reaches_head"])
empty = sorted(n for n in report if report[n]["verts"] == 0)
unweighted = sum(1 for v in mesh.data.vertices
                 if not any(g.weight >= THRESH for g in v.groups))

print(json.dumps({{
    "proxy_faces": proxy_faces,
    "proxy_heat_error": proxy_error,
    "proxy_groups_with_weight": proxy_nonempty,
    "mesh_vertex_groups": len(mesh.vertex_groups),
    "empty_groups": empty,
    "legs_reaching_head": contaminated,
    "unweighted_vertices": unweighted,
    "total_vertices": len(mesh.data.vertices),
    "verdict": ("ENVELOPE-LIKE BLEED" if contaminated
                else "EMPTY GROUPS" if empty
                else "CLEAN"),
    "report": report,
}}, indent=2))
'''


def build_code(voxel_size: float, weight_threshold: float, keep_proxy: bool) -> str:
    return TEMPLATE.format(
        voxel_size=voxel_size,
        weight_threshold=weight_threshold,
        keep_proxy=bool(keep_proxy),
    )
