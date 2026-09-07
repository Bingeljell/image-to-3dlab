from _rpc import send
CODE=r'''
import bpy,json
from mathutils import Quaternion
r=bpy.data.objects['rig'];s=bpy.context.scene
src=bpy.data.actions['Pangolin_Attack_TailSwipe_CW180_Sprite_v05_WhipHold'];src.use_fake_user=True
name='Pangolin_Attack_TailSwipe_CW180_Sprite_v06_SoftRecoil'
assert name not in bpy.data.actions
a=src.copy();a.name=name;a.use_fake_user=True;r.animation_data.action=a
tail=['spine.003','spine.002','spine.001','spine']
poses={}
for f in [75,77]:
 s.frame_set(f);poses[f]={n:r.pose.bones[n].rotation_quaternion.copy() for n in tail}
for f,amount in [(79,.8),(80,.6),(81,.3),(82,0)]:
 s.frame_set(f)
 for n in tail:
  b=r.pose.bones[n];b.rotation_quaternion=poses[75][n].slerp(poses[77][n],amount)
  b.keyframe_insert('rotation_quaternion',frame=f,group=n)
for marker in list(a.pose_markers):
 if marker.frame in [79,80,81,82]:a.pose_markers.remove(marker)
for label,f in [('Recoil 80%',79),('Recoil 60%',80),('Recoil 30%',81),('Recoiled',82)]:a.pose_markers.new(label).frame=f
s.frame_start=1;s.frame_end=100;s.frame_set(79)
path='/Users/nikhilshahane/projects/worklings-blender-work/clockwork-pangolin-rigify-impact-v3.blend'
bpy.ops.wm.save_as_mainfile(filepath=path)
print(json.dumps({'saved':path,'action':a.name}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
