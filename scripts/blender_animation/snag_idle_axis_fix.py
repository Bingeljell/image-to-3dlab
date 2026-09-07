"""Preserve idle motion and explicitly key missing pose channels."""
from _rpc import send
CODE=r'''
import bpy,os,json
from mathutils import Matrix
s=bpy.context.scene; r=bpy.data.objects['SnagRig']
folder=os.path.dirname(bpy.data.filepath)
target=os.path.join(folder,'snag_straightroots_animated_v15.blend')
assert not os.path.exists(target)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder,'snag_idle_before_axis_fix.blend'),copy=True)
r.animation_data.action=None
for b in r.pose.bones: b.matrix_basis=Matrix.Identity(4)
r.animation_data.action=bpy.data.actions['Attack_Whip_24f_Review']
s.frame_set(1); bpy.context.view_layer.update()
baseline={b.name:{p:list(getattr(b,p)) for p in ('location','rotation_quaternion' if b.rotation_mode=='QUATERNION' else 'rotation_euler','scale')} for b in r.pose.bones}
old=bpy.data.actions['Idle']; old.name='Idle_BeforeAxisFix'; old.use_fake_user=True
a=old.copy(); a.name='Idle'; a.use_fake_user=True
r.animation_data.action=a
curves=[fc for l in a.layers for st in l.strips for bag in st.channelbags for fc in bag.fcurves]
existing={(fc.data_path,fc.array_index) for fc in curves}
added=[]
for b in r.pose.bones:
    for prop,values in baseline[b.name].items():
        path=b.path_from_id(prop)
        for i,value in enumerate(values):
            if (path,i) in existing: continue
            getattr(b,prop)[i]=value
            for frame in (1,25): b.keyframe_insert(prop,index=i,frame=frame,group=b.name)
            added.append((path,i))
def pose(): return {b.name:[x for row in b.matrix_basis for x in row] for b in r.pose.bones}
s.frame_set(7); clean=pose()
r.animation_data.action=bpy.data.actions['Death_48f_Review']; s.frame_set(40)
r.animation_data.action=a; s.frame_set(7); after=pose()
error=max(abs(x-y) for n in clean for x,y in zip(clean[n],after[n]))
newcurves=[fc for l in a.layers for st in l.strips for bag in st.channelbags for fc in bag.fcurves]
lookup={(fc.data_path,fc.array_index):fc for fc in newcurves}
motion_error=max(abs(fc.evaluate(f)-lookup[(fc.data_path,fc.array_index)].evaluate(f)) for fc in curves for f in range(1,26))
assert error<1e-6 and motion_error<1e-6
s.frame_start=1; s.frame_end=24; s.use_preview_range=False; s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=target)
def snap():
    bpy.ops.screen.screenshot(filepath=os.path.join(folder,'idle_axis_fixed.png'))
bpy.app.timers.register(snap,first_interval=.5)
print(json.dumps({'saved':target,'action':a.name,'original':old.name,'added_channels':len(added),'death_carryover_error':error,'existing_motion_error':motion_error,'root_rotation':list(r.pose.bones['root'].rotation_quaternion),'body_rotation':list(r.pose.bones['body'].rotation_euler)}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=60))
