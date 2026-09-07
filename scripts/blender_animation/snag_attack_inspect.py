"""Read the active whip action and capture poses without editing animation."""
from _rpc import send

CODE = r'''
import bpy, json, os
from mathutils import Vector
s=bpy.context.scene
r=bpy.data.objects['SnagRig']
a=r.animation_data.action
assert a.name=='Attack_Whip', a.name
original=s.frame_current
area=max((ar for ar in bpy.context.screen.areas if ar.type=='VIEW_3D'),key=lambda ar:ar.width)
view=area.spaces.active.region_3d
saved_view=(view.view_rotation.copy(),view.view_location.copy(),view.view_distance,view.view_perspective)
view.view_rotation=Vector((1.1,-2,1)).to_track_quat('Z','Y')
view.view_location=Vector((0,-.1,0))
view.view_distance=1.8
view.view_perspective='PERSP'
curves=[fc for layer in a.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]
print(json.dumps({'file':bpy.data.filepath,'frame':original,'fps':s.render.fps,'range':list(a.frame_range),'keys':sorted(set(float(k.co.x) for fc in curves for k in fc.keyframe_points)),'motion':[{ 'path':fc.data_path,'index':fc.array_index,'min':round(min(k.co.y for k in fc.keyframe_points),3),'max':round(max(k.co.y for k in fc.keyframe_points),3)} for fc in curves if fc.keyframe_points and max(k.co.y for k in fc.keyframe_points)-min(k.co.y for k in fc.keyframe_points)>.02]}))
frames=iter([1,5,9,12,15,18,21,24])
folder=os.path.join(os.path.dirname(bpy.data.filepath),'attack_inspection')
os.makedirs(folder,exist_ok=True)
state={'capture':False,'frame':1}
def attack_pose_capture():
    if state['capture']:
        bpy.ops.screen.screenshot(filepath=os.path.join(folder,f"attack_{state['frame']:02d}.png"))
        state['capture']=False
    try: f=next(frames)
    except StopIteration:
        s.frame_set(original)
        view.view_rotation,view.view_location,view.view_distance,view.view_perspective=saved_view
        return None
    s.frame_set(f)
    for area in bpy.context.screen.areas:
        if area.type=='VIEW_3D': area.tag_redraw()
    state['frame']=f
    state['capture']=True
    return .7
bpy.app.timers.register(attack_pose_capture,first_interval=.1)
'''

print(send(CODE,'127.0.0.1',9876,timeout=30))
