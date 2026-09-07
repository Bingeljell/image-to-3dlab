"""SUPERSEDED experiment: foot-only correction did not resolve the thigh twist.

Historical reference only; use forest_hind_chain_fix.py as the accepted starting
point. Running this still edits and saves the live scene.
"""
from _rpc import send
CODE=r'''
import bpy,os,json
from mathutils import Vector,Matrix
r=bpy.data.objects['rig'];s=bpy.context.scene
source=r.animation_data.action
assert source.name=='ForestFlicker_Death_44f'
source.name='ForestFlicker_Death_44f_before_paw_fix';source.use_fake_user=True
a=source.copy();a.name='ForestFlicker_Death_44f';r.animation_data.action=a
b=r.pose.bones['foot_fk.R'];previous=None
for f in range(1,45):
    s.frame_set(f);bpy.context.view_layer.update()
    mat=b.matrix.copy();q=mat.to_quaternion()
    t=max(0,min(1,(f-20)/15));t=t*t*(3-2*t)
    direction=Vector((-.4,-1,-.3)).normalized()
    swing=(q@Vector((0,1,0))).rotation_difference(direction)
    loc=b.location.copy();scale=b.scale.copy()
    b.matrix=Matrix.LocRotScale(mat.translation,q.slerp(swing@q,t),mat.to_scale())
    b.location=loc;b.scale=scale
    if previous and b.rotation_quaternion.dot(previous)<0:b.rotation_quaternion.negate()
    previous=b.rotation_quaternion.copy()
    b.keyframe_insert('rotation_quaternion',frame=f,group=b.name)
s.frame_set(44)
target=os.path.join(os.path.dirname(bpy.data.filepath),'forest-flicker-rigify-polished-v6.blend')
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'frames':list(a.frame_range),'modified_bone':b.name}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
