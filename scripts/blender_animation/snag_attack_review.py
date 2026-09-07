"""Build the approved 24-frame whip review in the live Blender scene."""
import sys
from _rpc import send

BUILD = r'''
import bpy, math, os, json
from mathutils import Quaternion, Vector, Matrix
s=bpy.context.scene
r=bpy.data.objects['SnagRig']
folder=os.path.dirname(bpy.data.filepath)
target=os.path.join(folder,'snag_straightroots_animated_v14.blend')
assert r.animation_data.action.name in ('Attack_Whip','Attack_Whip_24f_Review')
backup=os.path.join(folder,'snag_attack_before_review.blend')
if not os.path.exists(backup):
    assert not os.path.exists(target), 'v14 already exists'
    bpy.ops.wm.save_as_mainfile(filepath=backup,copy=True)
old=bpy.data.actions['Attack_Whip']
old.use_fake_user=True
r.animation_data.action=None
for b in r.pose.bones: b.matrix_basis=Matrix.Identity(4)
r.animation_data.action=old
s.frame_set(1)
bpy.context.view_layer.update()
base={b.name:(b.location.copy(),b.rotation_quaternion.copy(),b.rotation_euler.copy(),b.scale.copy()) for b in r.pose.bones}
directions={b.name:(b.tail-b.head).normalized() for b in r.pose.bones}
tails={b.name:b.tail.copy() for b in r.pose.bones}
prior=bpy.data.actions.get('Attack_Whip_24f_Review')
if prior: bpy.data.actions.remove(prior)
a=bpy.data.actions.new('Attack_Whip_24f_Review')
a.use_fake_user=True
r.animation_data.action=a
def env(f,points):
    if f<=points[0][0]: return points[0][1]
    for (t0,v0),(t1,v1) in zip(points,points[1:]):
        if f<=t1:
            u=(f-t0)/(t1-t0); u=u*u*(3-2*u)
            return v0+(v1-v0)*u
    return points[-1][1]
def orient(b,desired):
    current=(b.tail-b.head).normalized()
    q=current.rotation_difference(desired.normalized())@b.matrix.to_quaternion()
    b.matrix=Matrix.LocRotScale(b.matrix.translation,q,b.matrix.to_scale())
    b.location=base[b.name][0]
    b.scale=base[b.name][3]
    b.rotation_quaternion.normalize()
    bpy.context.view_layer.update()
wind_angles=[15,40,75,110,145,180,215,240]
strike_angles=[-8,-6,-4,-1,2,5,8,12]
previous={}
tip_positions=[]
for f in range(1,25):
    s.frame_set(f)
    for b in r.pose.bones:
        loc,q,e,sc=base[b.name]
        b.location=loc; b.rotation_quaternion=q; b.rotation_euler=e; b.scale=sc
    body=r.pose.bones['body']
    lean=env(f,[(1,0),(7,-8),(9,-3),(11,10),(13,12),(16,5),(20,-2),(24,0)])
    twist=env(f,[(1,0),(7,-6),(10,-2),(13,6),(17,2),(24,0)])
    body.rotation_euler.x=math.radians(lean)
    body.rotation_euler.y=math.radians(twist)
    forward=env(f,[(1,0),(7,-.025),(10,.01),(13,.035),(16,.025),(24,0)])
    body.location=base['body'][0]+body.bone.matrix_local.to_3x3().inverted()@Vector((0,-forward,0))
    bpy.context.view_layer.update()
    # Outer roots retain their planted shape while the torso supplies the force.
    brace=env(f,[(1,0),(5,.9),(15,.9),(24,0)])
    for side in ('L','R'):
        for i in range(1,9):
            b=r.pose.bones[f'tentacle_{side}_{i:02d}']
            current=(b.tail-b.head).normalized()
            anchored=directions[b.name]
            orient(b,current.lerp(anchored,brace))
    for i in range(1,9):
        b=r.pose.bones[f'tentacle_C_{i:02d}']
        initial=directions[b.name]
        start_angle=math.degrees(math.atan2(initial.z,-initial.y))
        # The coil opens from the shoulder outward: the tip releases last.
        lag=(i-1)*.42
        wind=env(f,[(1,0),(3,.12),(7,1),(9,1)])
        launch=env(f,[(1,0),(8+lag,0),(10+lag,1)])
        recover=env(f,[(1,0),(16+lag*.35,0),(24,1)])
        angle=start_angle+(wind_angles[i-1]-start_angle)*wind
        angle=angle+(strike_angles[i-1]-angle)*launch
        overshoot=env(f,[(1,0),(13+lag*.25,0),(15+lag*.25,1),(18+lag*.25,0)])
        angle-=overshoot*(12+2*i)
        angle=angle+(start_angle-angle)*recover
        lateral=env(f,[(1,initial.x),(7,-.65),(11,-.1),(14,.14),(18,.06),(24,initial.x)])
        rad=math.radians(angle)
        desired=Vector((lateral,-math.cos(rad),math.sin(rad)))
        # Endpoints match the original ready stance exactly.
        if f in (1,24): desired=initial
        orient(b,desired)
    if f in (1,24):
        for b in r.pose.bones:
            loc,q,e,sc=base[b.name]
            b.location=loc; b.rotation_quaternion=q; b.rotation_euler=e; b.scale=sc
    bpy.context.view_layer.update()
    tip_positions.append((f,list(r.pose.bones['tentacle_C_08'].tail)))
    for b in r.pose.bones:
        if b.rotation_mode=='QUATERNION':
            if b.name in previous and b.rotation_quaternion.dot(previous[b.name])<0: b.rotation_quaternion.negate()
            previous[b.name]=b.rotation_quaternion.copy()
        for prop in ('location','rotation_quaternion' if b.rotation_mode=='QUATERNION' else 'rotation_euler','scale'):
            b.keyframe_insert(prop,frame=f,group=b.name)
for layer in a.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for fc in bag.fcurves:
                for k in fc.keyframe_points: k.interpolation='LINEAR'
a.use_frame_range=True; a.frame_start=1; a.frame_end=24
for name,f in [('Ready',1),('Loaded',7),('Drive',10),('Crack',13),('Follow through',16),('Recover',24)]:
    marker=a.pose_markers.new(name); marker.frame=f
a['review_notes']='24fps: braced body windup; centre root coils; base leads, tip releases last; overshoot and return. All pose transforms keyed to avoid action carryover.'
s.frame_start=1; s.frame_end=24; s.use_preview_range=False
s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'action':a.name,'tip_positions':tip_positions}))
'''

CHECK = r'''
import bpy,json
r=bpy.data.objects['SnagRig']; s=bpy.context.scene
def pose(): return {b.name:[x for row in b.matrix_basis for x in row] for b in r.pose.bones}
s.frame_set(1); first=pose()
s.frame_set(24); last=pose()
err=max(abs(x-y) for n in first for x,y in zip(first[n],last[n]))
review=r.animation_data.action
r.animation_data.action=bpy.data.actions['Death_48f_Review']; s.frame_set(40)
r.animation_data.action=review; s.frame_set(1)
again=pose()
carry=max(abs(x-y) for n in first for x,y in zip(first[n],again[n]))
print(json.dumps({'range':list(review.frame_range),'ready_pose_return_error':err,'death_to_attack_carryover_error':carry,'actions':list(bpy.data.actions.keys())}))
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
'''

print(send(CHECK if len(sys.argv)>1 and sys.argv[1]=='check' else BUILD,'127.0.0.1',9876,timeout=60))
