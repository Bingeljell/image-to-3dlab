from _rpc import send
CODE=r'''
import bpy,os,math,json
from mathutils import Quaternion,Vector
r=bpy.data.objects['rig'];s=bpy.context.scene
folder=os.path.dirname(bpy.data.filepath)
backup=os.path.join(folder,'clockwork-pangolin-before-impact.blend')
if not os.path.exists(backup):bpy.ops.wm.save_as_mainfile(filepath=backup,copy=True)
def curves(a):
    return [fc for l in a.layers for st in l.strips for bag in st.channelbags for fc in bag.fcurves]
def sample(a,frames,names):
    r.animation_data.action=a;out={}
    for f in frames:
        s.frame_set(f);bpy.context.view_layer.update()
        out[f]={n:(r.pose.bones[n].location.copy(),r.pose.bones[n].rotation_quaternion.copy(),r.pose.bones[n].scale.copy()) for n in names}
    return out
def env(f,pts):
    for (t,v),(u,w) in zip(pts,pts[1:]):
        if f<=u:
            x=max(0,(f-t)/(u-t));x=x*x*(3-2*x);return v+(w-v)*x
    return pts[-1][1]
tail=['spine.003','spine.002','spine.001','spine']
src=bpy.data.actions['Pangolin_Attack_TailSwipe_CW180_Sprite_v03'];src.use_fake_user=True
poses=sample(src,[1,60,70],tail)
a=src.copy();a.name='Pangolin_Attack_TailSwipe_CW180_Sprite_v04_Whip';a.use_fake_user=True
r.animation_data.action=a
# Preserve the full root turn; move the tail strike to after its completion at 74.
for fc in curves(a):
    if any(('pose.bones["'+n+'"]') in fc.data_path for n in ['torso','hips','chest']):
        for k in fc.keyframe_points:
            old=k.co.x
            if old<=60:new=44+(old-44)*31/16
            elif old<=70:new=75+(old-60)*2/10
            elif old<=74:new=77+(old-70)*3/4
            elif old<=86:new=80+(old-74)*6/12
            else:new=old
            delta=new-old;k.co.x=new;k.handle_left.x+=delta;k.handle_right.x+=delta
        fc.update()
for f in range(1,101):
    s.frame_set(f)
    for i,n in enumerate(tail):
        neutral=poses[1][n][1];coil=poses[60][n][1]
        extended=neutral.slerp(poses[70][n][1],[.9,.55,.22,.08][i])
        if f<=75:q=neutral.slerp(coil,env(f,[(1,0),(32,0),(60,1),(75,1)]))
        elif f<=78:q=coil.slerp(extended,{76:.5,77:1,78:.5}[f])
        elif f<=81:q=coil.slerp(extended,env(f,[(78,.5),(81,0)]))
        else:q=coil.slerp(neutral,env(f,[(81,0),(91,1),(100,1)]))
        b=r.pose.bones[n];b.rotation_quaternion=q;b.keyframe_insert('rotation_quaternion',frame=f,group=n)
for fc in curves(a):
    if any(('pose.bones["'+n+'"]') in fc.data_path for n in tail):
        for k in fc.keyframe_points:k.interpolation='LINEAR'
for label,f in [('Turn complete',74),('Whip 0%',75),('Whip 50%',76),('Whip 100%',77),('Recoil 50%',78),('Recoiled',81)]:a.pose_markers.new(label).frame=f
a.use_frame_range=True;a.frame_start=1;a.frame_end=100
whip=a
src=bpy.data.actions['Pangolin_Special_RearSlam_Sprite_v04'];src.use_fake_user=True
names=['torso','chest','head']+tail
poses=sample(src,range(1,121),names)
a=src.copy();a.name='Pangolin_Special_RearSlam_Sprite_v05_Heavy';a.use_fake_user=True;r.animation_data.action=a
for f in range(1,121):
    s.frame_set(f)
    # A deeper load, then compression held for three frames and one restrained recoil.
    load=env(f,[(1,0),(16,1),(24,0)])
    hit=env(f,[(1,0),(46,0),(48,1),(50,1),(54,-.18),(59,.28),(72,0)])
    b=r.pose.bones['torso'];b.location=poses[f]['torso'][0]+Vector((0,0,-.018*load-.055*hit))
    b.rotation_quaternion=poses[f]['torso'][1]
    b.keyframe_insert('location',frame=f,group=b.name)
    b=r.pose.bones['chest'];b.rotation_quaternion=poses[f]['chest'][1]@Quaternion((1,0,0),math.radians(2*load+3*hit))
    b.keyframe_insert('rotation_quaternion',frame=f,group=b.name)
    # Tail mass settles after contact, with the tip responding last.
    for i,n in enumerate(tail):
        lag=env(f,[(1,0),(47+i,0),(51+i,-1),(57+i,.3),(65+i,0)])
        b=r.pose.bones[n];b.rotation_quaternion=poses[f][n][1]@Quaternion((1,0,0),math.radians((3+i*2)*lag))
        b.keyframe_insert('rotation_quaternion',frame=f,group=n)
for fc in curves(a):
    for k in fc.keyframe_points:
        if len(fc.keyframe_points)>100:k.interpolation='LINEAR'
for label,f in [('Loaded',16),('Apex',44),('Contact',48),('Weight held',50),('Small recoil',54),('Settle',65)]:a.pose_markers.new(label).frame=f
a.use_frame_range=True;a.frame_start=1;a.frame_end=120
r.animation_data.action=whip;s.frame_start=1;s.frame_end=100;s.frame_set(75)
target=os.path.join(folder,'clockwork-pangolin-rigify-impact-v1.blend');bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'backup':backup,'actions':[whip.name,a.name]}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
