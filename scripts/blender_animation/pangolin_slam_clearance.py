from _rpc import send
CODE=r'''
import bpy,math,json
from mathutils import Quaternion,Vector
r=bpy.data.objects['rig'];s=bpy.context.scene;m=bpy.data.objects['geometry_0']
src=bpy.data.actions['Pangolin_Special_RearSlam_Sprite_v04'];a=bpy.data.actions['Pangolin_Special_RearSlam_Sprite_v05_Heavy']
feet=['front_foot_ik.L','front_foot_ik.R','foot_ik.L','foot_ik.R']
samples={}
def lowest():
    ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();z=min((ev.matrix_world@v.co).z for v in me.vertices);ev.to_mesh_clear();return z
r.animation_data.action=src
for f in range(1,121):
    s.frame_set(f);bpy.context.view_layer.update()
    samples[f]=(r.pose.bones['head'].rotation_quaternion.copy(),{n:r.pose.bones[n].matrix.copy() for n in feet},lowest(),r.pose.bones['torso'].location.copy())
def env(f,pts):
    for (t,v),(u,w) in zip(pts,pts[1:]):
        if f<=u:
            x=max(0,(f-t)/(u-t));x=x*x*(3-2*x);return v+(w-v)*x
    return pts[-1][1]
r.animation_data.action=a;checks=[]
for f in range(1,121):
    s.frame_set(f);q,matrices,floor,torso=samples[f]
    load=env(f,[(1,0),(16,1),(24,0)])
    hit=env(f,[(1,0),(46,0),(48,1),(50,1),(54,-.18),(59,.28),(72,0)])
    b=r.pose.bones['torso'];b.location=torso+Vector((0,0,-.012*load-.025*hit));b.keyframe_insert('location',frame=f,group=b.name)
    bpy.context.view_layer.update()
    for n,mat in matrices.items():
        b=r.pose.bones[n];b.matrix=mat;bpy.context.view_layer.update()
        b.keyframe_insert('location',frame=f,group=n);b.keyframe_insert('rotation_quaternion',frame=f,group=n)
    b=r.pose.bones['head'];b.rotation_quaternion=q;bpy.context.view_layer.update()
    # Lift the chin only as needed; do not translate the rig or disturb planted feet.
    correction=0
    while lowest()<floor-.003 and correction<18:
        correction+=.5;b.rotation_quaternion=q@Quaternion((1,0,0),math.radians(-correction));bpy.context.view_layer.update()
    b.keyframe_insert('rotation_quaternion',frame=f,group=b.name)
    if f in [18,48,49,50,54,59]:checks.append((f,correction,lowest(),floor))
for l in a.layers:
 for st in l.strips:
  for bag in st.channelbags:
   for fc in bag.fcurves:
    if len(fc.keyframe_points)>=120:
     for k in fc.keyframe_points:k.interpolation='LINEAR'
r.animation_data.action=bpy.data.actions['Pangolin_Attack_TailSwipe_CW180_Sprite_v04_Whip'];s.frame_set(75)
bpy.ops.wm.save_as_mainfile(filepath='/Users/nikhilshahane/projects/worklings-blender-work/clockwork-pangolin-rigify-impact-v1.blend')
print(json.dumps(checks))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
