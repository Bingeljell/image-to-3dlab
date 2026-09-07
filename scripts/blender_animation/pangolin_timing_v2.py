from _rpc import send
CODE=r'''
import bpy,json
r=bpy.data.objects['rig'];s=bpy.context.scene
def curves(a):
    return [fc for l in a.layers for st in l.strips for bag in st.channelbags for fc in bag.fcurves]
def map_time(f,points):
    for (t,v),(u,w) in zip(points,points[1:]):
        if f<=u:return v+(w-v)*(f-t)/(u-t)
    return points[-1][1]
def retime(srcname,name,end,points,markers):
    src=bpy.data.actions[srcname];src.use_fake_user=True
    assert name not in bpy.data.actions
    a=src.copy();a.name=name;a.use_fake_user=True
    for srcfc,fc in zip(curves(src),curves(a)):
        values=[srcfc.evaluate(map_time(f,points)) for f in range(1,end+1)]
        fc.keyframe_points.clear();fc.keyframe_points.add(end)
        fc.keyframe_points.foreach_set('co',[v for f,y in enumerate(values,1) for v in (f,y)])
        for k in fc.keyframe_points:k.interpolation='LINEAR'
        fc.update()
    for marker in list(a.pose_markers):a.pose_markers.remove(marker)
    for label,f in markers:a.pose_markers.new(label).frame=f
    a.use_frame_range=True;a.frame_start=1;a.frame_end=end
    return a
whip=retime('Pangolin_Attack_TailSwipe_CW180_Sprite_v04_Whip','Pangolin_Attack_TailSwipe_CW180_Sprite_v05_WhipHold',100,
 [(1,1),(77,77),(78,77),(92,91),(100,100)],
 [('Turn complete',74),('Whip 0%',75),('Whip 50%',76),('Full extension',77),('Full extension hold',78),('Recoil 50%',79),('Recoiled',82)])
slam=retime('Pangolin_Special_RearSlam_Sprite_v05_Heavy','Pangolin_Special_RearSlam_Sprite_v06_Heavy60',60,
 [(1,1),(8,18),(18,38),(22,44),(24,46),(26,48),(28,50),(32,54),(37,59),(43,72),(51,92),(60,120)],
 [('Loaded',8),('Apex',22),('Drop',24),('Contact',26),('Weight held',28),('Small recoil',32),('Settle',43),('Ready',60)])
r.animation_data.action=slam;s.frame_start=1;s.frame_end=60;s.frame_set(26)
target='/Users/nikhilshahane/projects/worklings-blender-work/clockwork-pangolin-rigify-impact-v2.blend'
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'actions':[whip.name,slam.name],'whip_hold_max_curve_difference':max(abs(fc.evaluate(77)-fc.evaluate(78)) for fc in curves(whip))}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
