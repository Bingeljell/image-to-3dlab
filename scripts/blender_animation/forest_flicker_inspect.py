"""Temporarily sample existing actions, then restore the live pose."""
from _rpc import send
CODE=r'''
import bpy,os,json
s=bpy.context.scene; r=bpy.data.objects['rig']
saved_frame=s.frame_current; saved_action=r.animation_data.action
saved_pose={b.name:b.matrix_basis.copy() for b in r.pose.bones}
saved_props={b.name:{k:v for k,v in b.items() if isinstance(v,(float,int,str))} for b in r.pose.bones}
jobs=[]
for label,suffix,frames in [('idle','Idle_BreatheLook',(1,48,96)),('walk','Walk_Feline',(1,9,17)),('swipe','Attack_RightSwipe',(1,12,18)),('slam','Special_DoublePawSlam',(1,16,26)),('damage','Damage_Wince_TailDown',(1,10,18))]:
    for f in frames: jobs.append((label,'ForestFlicker_'+suffix,f))
queue=iter(jobs); state={'pending':None}
os.makedirs('/private/tmp/forest_flicker_review',exist_ok=True)
def capture():
    if state['pending']:
        label,_,f=state['pending']
        bpy.ops.screen.screenshot(filepath=f'/private/tmp/forest_flicker_review/{label}_{f}.png')
    try: label,name,f=next(queue)
    except StopIteration:
        r.animation_data.action=saved_action
        s.frame_set(saved_frame)
        for b in r.pose.bones:
            b.matrix_basis=saved_pose[b.name]
            for k,v in saved_props[b.name].items(): b[k]=v
        bpy.context.view_layer.update()
        return None
    for b in r.pose.bones:
        b.matrix_basis=saved_pose[b.name]
        for k,v in saved_props[b.name].items(): b[k]=v
    r.animation_data.action=bpy.data.actions[name]
    if r.animation_data.action.slots: r.animation_data.action_slot=r.animation_data.action.slots[0]
    s.frame_set(f); bpy.context.view_layer.update()
    state['pending']=(label,name,f)
    for ar in bpy.context.screen.areas: ar.tag_redraw()
    return .65
bpy.app.timers.register(capture,first_interval=.1)
print('Sampling 15 poses; restoring original pose and action afterward.')
'''
print(send(CODE,'127.0.0.1',9876,timeout=30))
