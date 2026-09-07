"""Repair ear binding and create reviewed feline combat action variants."""
import sys
from _rpc import send
CODE=r'''
import bpy,os,json,math
from mathutils import Vector,Quaternion
s=bpy.context.scene;r=bpy.data.objects['rig'];m=bpy.data.objects['geometry_0']
folder=os.path.dirname(bpy.data.filepath)
target=os.path.join(folder,'forest-flicker-rigify-polished-v2.blend')
backup=os.path.join(folder,'forest-flicker-before-polish.blend')
if not os.path.exists(backup):
    assert not os.path.exists(target)
    bpy.ops.wm.save_as_mainfile(filepath=backup,copy=True)
def curves(a): return [f for l in a.layers for st in l.strips for bag in st.channelbags for f in bag.fcurves]
def env(f,pts):
    if f<=pts[0][0]:return pts[0][1]
    for (t,v),(u,w) in zip(pts,pts[1:]):
        if f<=u:
            x=(f-t)/(u-t);x=x*x*(3-2*x)
            return v+(w-v)*x
    return pts[-1][1]
def linear(f,pts):
    for (t,v),(u,w) in zip(pts,pts[1:]):
        if f<=u:return v+(w-v)*(f-t)/(u-t)
    return pts[-1][1]
headnames=['DEF-spine.008','DEF-spine.009','DEF-spine.010','DEF-spine.011']
headids={m.vertex_groups[n].index for n in headnames}
fallback=[.124,.59,.274,.012]
changed=0;badtip=0
already_fixed=bpy.data.actions.get('ForestFlicker_Attack_RightSwipe_Polished') is not None
for v in ([] if already_fixed else m.data.vertices):
    p=m.matrix_world@v.co
    if p.y>=-.5 or p.z<=.95:continue
    weights={g.group:g.weight for g in v.groups if g.weight>0}
    if not weights:continue
    foreign=sum(w for i,w in weights.items() if i not in headids)
    if foreign<1e-6:continue
    alpha=min(1,max(0,(p.z-.95)/.15));alpha=alpha*alpha*(3-2*alpha)
    totalhead=sum(weights.get(m.vertex_groups[n].index,0) for n in headnames)
    norm=[weights.get(m.vertex_groups[n].index,0)/totalhead for n in headnames] if totalhead>1e-6 else fallback
    if totalhead<1e-6 and p.x<0 and p.z>1.45:badtip+=1
    updated={i:w*(1-alpha) for i,w in weights.items()}
    for n,w in zip(headnames,norm):
        i=m.vertex_groups[n].index;updated[i]=updated.get(i,0)+alpha*w
    total=sum(updated.values())
    for i in weights:m.vertex_groups[i].remove([v.index])
    for i,w in updated.items():
        if w>1e-7:m.vertex_groups[i].add([v.index],w/total,'REPLACE')
    changed+=1
muzzle_ids=[v.index for v in m.data.vertices if (m.matrix_world@v.co).y<-.8 and .2<(m.matrix_world@v.co).z<.55]
for suffix,count in [('Attack_RightSwipe',32),('Special_DoublePawSlam',44)]:
    old=bpy.data.actions['ForestFlicker_'+suffix];old.use_fake_user=True
    name=old.name+'_Polished'
    prior=bpy.data.actions.get(name)
    if prior:
        if r.animation_data.action==prior:r.animation_data.action=None
        bpy.data.actions.remove(prior)
    a=old.copy();a.name=name;a.use_fake_user=True
    timing=[(1,1),(10,12),(13,16),(16,20),(23,24),(32,32)] if count==32 else [(1,1),(10,10),(18,18),(21,20),(24,24),(29,28),(44,44)]
    for src,dst in zip(curves(old),curves(a)):
        values=[src.evaluate(linear(f,timing)) for f in range(1,count+1)]
        dst.keyframe_points.clear()
        for f,v in enumerate(values,1):dst.keyframe_points.insert(f,v,options={'FAST'})
        dst.update()
    r.animation_data.action=a
    if a.slots:r.animation_data.action_slot=a.slots[0]
    def offset(name,xyz):
        b=r.pose.bones[name];mat=b.matrix.copy();mat.translation+=Vector(xyz);b.matrix=mat
        b.keyframe_insert('location',frame=f,group=name)
        bpy.context.view_layer.update()
    def turn(name,axis,deg):
        b=r.pose.bones[name];b.rotation_quaternion=b.rotation_quaternion@Quaternion(axis,math.radians(deg))
        b.keyframe_insert('rotation_quaternion',frame=f,group=name)
    for f in range(1,count+1):
        s.frame_set(f);bpy.context.view_layer.update()
        if count==32:
            load=env(f,[(1,0),(9,1),(12,.75),(16,0),(32,0)])
            strike=env(f,[(1,0),(12,0),(16,1),(20,.65),(28,0),(32,0)])
            offset('torso',(.025*load,-.04*strike,-.025*load))
            turn('chest',(0,1,0),-7*load+11*strike)
            turn('spine.003',(0,1,0),8*load-10*strike)
            bpy.context.view_layer.update()
            offset('front_foot_ik.R',(-.11*load+.11*strike,.035*load-.16*strike,.025*load+.10*strike))
        else:
            crouch=env(f,[(1,0),(9,1),(13,0),(44,0)])
            lift=env(f,[(1,0),(12,0),(19,1),(21,1),(24,0),(44,0)])
            land=env(f,[(1,0),(23,0),(25,1),(29,.4),(36,0),(44,0)])
            reach=env(f,[(1,0),(13,0),(20,1),(27,1),(38,0),(44,0)])
            offset('torso',(0,.035*crouch-.065*reach,-.065*crouch+.075*lift-.055*land))
            turn('chest',(1,0,0),-4*crouch+5*lift-7*land)
            turn('spine.003',(1,0,0),-5*lift+7*land)
            bpy.context.view_layer.update()
            for side in ('L','R'):
                offset('front_foot_ik.'+side,(0,-.11*reach,.04*lift))
            # Keep the face above the contact plane as the forelegs absorb impact.
            for attempt in range(3):
                ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
                low=min((ev.matrix_world@mesh.vertices[i].co).z for i in muzzle_ids)
                ev.to_mesh_clear()
                if low>=.04:break
                offset('torso',(0,0,.041-low))
    for fc in curves(a):
        for k in fc.keyframe_points:k.interpolation='LINEAR'
    a.use_frame_range=True;a.frame_start=1;a.frame_end=count
    a['polish_notes']='Preserved feline stance; stronger anticipation, torso drive, contact and recovery.'
r.animation_data.action=bpy.data.actions['ForestFlicker_Attack_RightSwipe_Polished']
s.frame_start=1;s.frame_end=32;s.use_preview_range=False;s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'ear_vertices_cleaned':changed,'wrong_right_tip_vertices':badtip,'actions':[a.name for a in bpy.data.actions if 'Polished' in a.name]}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
