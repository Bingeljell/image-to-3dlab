from _rpc import send
CODE=r'''
import bpy,json
s=bpy.context.scene;r=bpy.data.objects['rig'];death=bpy.data.actions['ForestFlicker_Death_44f']
root=r.pose.bones['root'];fixed=[]
for a in bpy.data.actions:
    if a==death:continue
    paths={f.data_path for l in a.layers for st in l.strips for bag in st.channelbags for f in bag.fcurves}
    r.animation_data.action=a
    end=int(a.frame_range[1])
    for prop,value in [('location',(0,0,0)),('rotation_quaternion',(1,0,0,0)),('scale',(1,1,1))]:
        if root.path_from_id(prop) in paths:continue
        setattr(root,prop,value)
        for f in (1,end):root.keyframe_insert(prop,frame=f,group='root')
    fixed.append(a.name)
r.animation_data.action=death
def pose():return {b.name:[v for row in b.matrix_basis for v in row] for b in r.pose.bones}
s.frame_set(41);p=pose();s.frame_set(44);q=pose()
err=max(abs(x-y) for n in p for x,y in zip(p[n],q[n]))
assert err<1e-6
r.animation_data.action=bpy.data.actions['ForestFlicker_Walk_Feline'];s.frame_set(1)
assert abs(root.rotation_quaternion.w-1)<1e-6 and root.location.length<1e-6
r.animation_data.action=death;s.frame_start=1;s.frame_end=44;s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print(json.dumps({'stillness_error':err,'walk_root_reset':'passed','death_frames':list(death.frame_range),'root_defaults_added_to':fixed,'saved':bpy.data.filepath}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=60))
