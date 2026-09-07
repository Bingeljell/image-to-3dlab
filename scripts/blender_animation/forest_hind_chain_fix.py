from _rpc import send
CODE=r'''
import bpy,os,math,json
from mathutils import Quaternion,Vector,Matrix
r=bpy.data.objects['rig'];s=bpy.context.scene
source=bpy.data.actions['ForestFlicker_Death_44f']
r.animation_data.action=source;s.frame_set(1)
names=['thigh_fk.R','shin_fk.R','foot_fk.R','toe.R']
names=[n for n in names if n in r.pose.bones]
base={n:r.pose.bones[n].rotation_quaternion.copy() for n in names}
source.name='ForestFlicker_Death_44f_v7_archive';source.use_fake_user=True
a=source.copy();a.name='ForestFlicker_Death_44f';r.animation_data.action=a
def smooth(f,start,end):
    t=max(0,min(1,(f-start)/(end-start)));return t*t*(3-2*t)
for f in range(1,45):
    s.frame_set(f)
    t=smooth(f,14,34)
    for n,angle in [('thigh_fk.R',0),('shin_fk.R',0),('foot_fk.R',0)]:
        b=r.pose.bones[n]
        b.rotation_quaternion=base[n]@Quaternion((1,0,0),math.radians(angle*t))
        bpy.context.view_layer.update()
        mat=b.matrix.copy();loc=b.location.copy();scale=b.scale.copy()
        y=Vector({'thigh_fk.R':(-1,-.7,-.12),'shin_fk.R':(-.25,1,-.1),'foot_fk.R':(-1,-.65,-.08)}[n]).normalized()
        q=mat.to_quaternion();targetq=(q@Vector((0,1,0))).rotation_difference(y)@q
        b.matrix=Matrix.LocRotScale(mat.translation,mat.to_quaternion().slerp(targetq,t),mat.to_scale())
        b.location=loc;b.scale=scale
        b.keyframe_insert('rotation_quaternion',frame=f,group=n)
    bpy.context.view_layer.update()
    b=r.pose.bones['toe.R'];b.rotation_quaternion=base['toe.R']
    b.keyframe_insert('rotation_quaternion',frame=f,group=b.name)
s.frame_set(31)
target=os.path.join(os.path.dirname(bpy.data.filepath),'forest-flicker-rigify-polished-v8.blend')
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'action':a.name,'frame':s.frame_current}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
