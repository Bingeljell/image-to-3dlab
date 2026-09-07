"""Run the Snag death review steps against the local Blender session."""
import sys
from _rpc import send

INSPECT = r'''
import bpy, json
s=bpy.context.scene
r=bpy.data.objects['SnagRig']
a=r.animation_data.action
def curves(a):
    return [fc for layer in a.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]
print(json.dumps({'fps':s.render.fps,'frame':s.frame_current,'action':a.name,'bones':[{'name':b.name,'parent':b.parent.name if b.parent else None,'head':list(b.head_local),'tail':list(b.tail_local)} for b in r.data.bones], 'curves':[{'path':f.data_path,'index':f.array_index,'keys':[(list(k.co),k.interpolation) for k in f.keyframe_points]} for f in curves(a)]}))
'''

BUILD = r'''
import bpy, math, json, os
from mathutils import Quaternion, Vector
s=bpy.context.scene
r=bpy.data.objects['SnagRig']
m=bpy.data.objects['geometry_0']
folder=os.path.dirname(bpy.data.filepath)
target=os.path.join(folder,'snag_straightroots_animated_v13.blend')
assert r.animation_data.action.name in ('Death','Death_48f_Review')
if not os.path.exists(os.path.join(folder,'snag_death_before_review.blend')):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder,'snag_death_before_review.blend'),copy=True)
old=bpy.data.actions['Death']
r.animation_data.action=old
prior=bpy.data.actions.get('Death_48f_Review')
if prior: bpy.data.actions.remove(prior)
old.use_fake_user=True
s.frame_set(1)
base={b.name:(b.location.copy(),b.rotation_quaternion.copy(),b.rotation_euler.copy(),b.scale.copy()) for b in r.pose.bones}
def bottom():
    ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=ev.to_mesh()
    z=min((ev.matrix_world@v.co).z for v in mesh.vertices)
    ev.to_mesh_clear()
    return z
floor=bottom()
a=bpy.data.actions.new('Death_48f_Review')
a.use_fake_user=True
r.animation_data.action=a
def env(f,points):
    if f<=points[0][0]: return points[0][1]
    for (t0,v0),(t1,v1) in zip(points,points[1:]):
        if f<=t1:
            u=(f-t0)/(t1-t0)
            u=u*u*(3-2*u)
            return v0+(v1-v0)*u
    return points[-1][1]
def rot(axis,degrees): return Quaternion(axis,math.radians(degrees))
previous_q={}
settled_root=None
for f in range(1,49):
    s.frame_set(f)
    for b in r.pose.bones:
        loc,q,e,sc=base[b.name]
        b.location=loc; b.rotation_quaternion=q; b.rotation_euler=e; b.scale=sc
    brace=env(f,[(1,0),(6,1),(10,.6),(16,0)])
    fail=env(f,[(1,0),(8,0),(12,1),(16,.5),(23,1)])
    fall=env(f,[(1,0),(16,0),(19,.12),(22,.52),(25,1),(28,.94),(33,.98)])
    root=r.pose.bones['root']
    roll=env(f,[(1,0),(7,-3),(12,9),(16,5),(19,28),(22,88),(25,158),(28,149),(33,154)])
    root.rotation_quaternion=base['root'][1]@rot((0,0,1),roll)
    root.location.x=env(f,[(1,0),(16,0),(25,-.045),(33,-.052)])
    body=r.pose.bones['body']
    body.rotation_euler.x=math.radians(-5*brace+15*fall)
    body.rotation_euler.z=math.radians(3*fail)
    body.scale=(1+.035*fall,1+.025*brace-.12*fall,1+.02*fall)
    for side in ('L','C','R'):
        for i in range(1,9):
            name=f'tentacle_{side}_{i:02d}'
            b=r.pose.bones[name]
            q=base[name][1]
            delay={'L':0,'C':2,'R':4}[side]+max(0,i-3)*.65
            settle=env(f-delay,[(1,0),(17,0),(25,1.06),(29,.94),(32,1)])
            # Left folds first, right remains extended as the last support.
            if side=='L':
                bend=(-10*fail if 2<=i<=5 else 0)+(-5*settle if 3<=i<=6 else 0)
                splay=7*settle if i<=3 else 0
                b.rotation_quaternion=q@rot((1,0,0),bend+2*brace)@rot((0,0,1),splay)
            elif side=='R':
                release=(.45 if i in (3,4,5,6) else .20 if i==8 else .12)*settle
                b.rotation_quaternion=q.slerp(Quaternion(),max(0,min(1,release)))@rot((1,0,0),-3*brace)@rot((0,0,1),-5*settle if i<=3 else 0)
            else:
                b.rotation_quaternion=q@rot((1,0,0),(-5*settle if 3<=i<=6 else 0)+3*brace)
            if side=='R' and i>=7:
                twitch=env(f,[(1,0),(35,0),(37,1),(38,-.25),(40,0)])
                b.rotation_quaternion=b.rotation_quaternion@rot((1,0,0),twitch*(7 if i==8 else 3))
            b.rotation_quaternion.normalize()
    bpy.context.view_layer.update()
    # As the body rolls, the roots lose muscle tension and drape toward the floor.
    # Solve each segment in pose space so a flipped body does not leave roots pointing skyward.
    for side in ('L','C','R'):
        for i in range(1,9):
            delay={'L':0,'C':1,'R':2}[side]+max(0,i-3)*.35
            drape=env(f-delay,[(1,0),(19,0),(26,.8),(31,1)])
            if drape<=0: continue
            b=r.pose.bones[f'tentacle_{side}_{i:02d}']
            head=b.head.copy()
            direction=b.tail-head
            lateral={'L':-.85,'C':.12,'R':.95}[side]
            down=max(-1.6,min(.05,(floor+.035-head.z)/max(.035,direction.length)))
            desired=Vector((lateral,-1,down)).normalized()
            swing=Quaternion().slerp(direction.normalized().rotation_difference(desired),drape)
            mat=b.matrix.copy()
            q=swing@mat.to_quaternion()
            from mathutils import Matrix
            b.matrix=Matrix.LocRotScale(mat.translation,q,mat.to_scale())
            if side=='R' and i>=7:
                twitch=env(f,[(1,0),(35,0),(37,1),(38,-.25),(40,0)])
                b.rotation_quaternion=b.rotation_quaternion@rot((1,0,0),twitch*(7 if i==8 else 3))
            bpy.context.view_layer.update()
    # Ground the performance at its original sole level, avoiding a sinking corpse.
    dz=floor-bottom()
    root.location+=root.bone.matrix_local.to_3x3().inverted()@Vector((0,0,dz))
    if f==33: settled_root=root.location.copy()
    if f>33: root.location=settled_root
    bpy.context.view_layer.update()
    for b in r.pose.bones:
        if b.rotation_mode=='QUATERNION':
            if b.name in previous_q and b.rotation_quaternion.dot(previous_q[b.name])<0:
                b.rotation_quaternion.negate()
            previous_q[b.name]=b.rotation_quaternion.copy()
        b.keyframe_insert('location',frame=f,group=b.name)
        b.keyframe_insert('rotation_quaternion' if b.rotation_mode=='QUATERNION' else 'rotation_euler',frame=f,group=b.name)
        b.keyframe_insert('scale',frame=f,group=b.name)
for layer in a.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for fc in bag.fcurves:
                for k in fc.keyframe_points: k.interpolation='LINEAR'
a.use_frame_range=True
a.frame_start=1; a.frame_end=48
s.frame_start=1; s.frame_end=48
s.use_preview_range=False
a['review_notes']='48f at 24fps: brace, left support fails, attempted recovery, sideways collapse, staggered roots, right-tip twitch, stillness.'
s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'action':a.name,'original_action':old.name,'floor':floor,'frames':list(a.frame_range)}))
'''

SNAPSHOT = r'''
import bpy, os
s=bpy.context.scene
s.frame_set(FRAME)
for area in bpy.context.screen.areas:
    if area.type=='VIEW_3D': area.tag_redraw()
def capture_review():
    bpy.ops.screen.screenshot(filepath=os.path.join(os.path.dirname(bpy.data.filepath),'death_review_FRAME.png'))
bpy.app.timers.register(capture_review,first_interval=.5)
print('Screenshot scheduled')
'''

VALIDATE = r'''
import bpy, json
s=bpy.context.scene
r=bpy.data.objects['SnagRig']
a=r.animation_data.action
states={}
for f in (1,16,25,33,35,37,40,41,48):
    s.frame_set(f)
    states[f]={b.name:list(b.matrix_basis) for b in []}
    states[f]={b.name:[v for row in b.matrix_basis for v in row] for b in r.pose.bones}
still=max(abs(x-y) for name in states[41] for x,y in zip(states[41][name],states[48][name]))
body_twitch=max(abs(x-y) for name in ('body','root') for x,y in zip(states[35][name],states[37][name]))
print(json.dumps({'action':a.name,'frame_range':list(a.frame_range),'stillness_error':still,'body_motion_during_tip_twitch':body_twitch,'actions':list(bpy.data.actions.keys())}))
s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
'''

if __name__ == '__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'inspect'
    code=BUILD if mode=='build' else VALIDATE if mode=='validate' else SNAPSHOT.replace('FRAME',sys.argv[2]) if mode=='snapshot' else INSPECT
    print(send(code, '127.0.0.1', 9876, timeout=60))
