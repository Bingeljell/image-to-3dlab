"""Author the intermediate 44-frame death (v5), not the accepted v8 leg pose.

See FOREST_FLICKER.md before replaying; later chain corrections are separate.
"""
from _rpc import send
CODE=r'''
import bpy,math,os,json
from mathutils import Quaternion,Vector,Matrix
s=bpy.context.scene;r=bpy.data.objects['rig'];m=bpy.data.objects['geometry_0']
folder=os.path.dirname(bpy.data.filepath);target=os.path.join(folder,'forest-flicker-rigify-polished-v5.blend')
backup=os.path.join(folder,'forest-flicker-before-death.blend')
if not os.path.exists(backup):
    assert not os.path.exists(target)
    bpy.ops.wm.save_as_mainfile(filepath=backup,copy=True)
r.animation_data.action=bpy.data.actions['ForestFlicker_Damage_Wince_TailDown'];s.frame_set(10);bpy.context.view_layer.update()
pain={n:(r.pose.bones[n].location.copy(),r.pose.bones[n].rotation_quaternion.copy()) for n in ('torso','chest','neck','head','spine.003','spine.002','spine.001','spine')}
r.animation_data.action=bpy.data.actions['ForestFlicker_Attack_RightSwipe_Polished']
r.pose.bones['root'].matrix_basis=Matrix.Identity(4)
s.frame_set(1);bpy.context.view_layer.update()
if 'ear_flop.R' not in r.data.bones:
    bpy.context.view_layer.objects.active=r;r.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb=r.data.edit_bones.new('ear_flop.R');eb.head=(-.25,-.67,.98);eb.tail=(-.339,-.668,1.554)
    eb.parent=r.data.edit_bones['DEF-spine.009'];eb.use_deform=True
    bpy.ops.object.mode_set(mode='OBJECT')
    vg=m.vertex_groups.new(name='ear_flop.R')
    for v in m.data.vertices:
        co=m.matrix_world@v.co
        if co.x<-.08 and co.y<-.5 and co.z>.98:
            t=min(1,max(0,(co.z-.98)/.27));w=t*t*(3-2*t)
            old=[(g.group,g.weight) for g in v.groups]
            for idx,weight in old:m.vertex_groups[idx].add([v.index],weight*(1-w),'REPLACE')
            vg.add([v.index],w,'REPLACE')
r.pose.bones['ear_flop.R'].rotation_mode='QUATERNION'
r.pose.bones['ear_flop.R'].matrix_basis=Matrix.Identity(4)
# Match FK controls to the current IK pose before switching for the collapse.
for prefix in ('front_',''):
    for side in ('L','R'):
        for part in ('thigh','shin','foot'):
            fk=r.pose.bones.get(prefix+part+'_fk.'+side)
            org=r.pose.bones.get('ORG-'+prefix+part+'.'+side)
            if fk and org:fk.matrix=org.matrix.copy();bpy.context.view_layer.update()
        r.pose.bones[prefix+'thigh_parent.'+side]['IK_FK']=1.0
bpy.context.view_layer.update()
controls=[b for b in r.pose.bones if not b.name.startswith(('DEF-','MCH-','ORG-','VIS_'))]
base={b.name:(b.location.copy(),b.rotation_quaternion.copy(),b.rotation_euler.copy(),b.scale.copy()) for b in controls}
def minimum():
    ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    z=min((ev.matrix_world@v.co).z for v in mesh.vertices);ev.to_mesh_clear();return z
floor=minimum()
name='ForestFlicker_Death_44f'
prior=bpy.data.actions.get(name)
if prior:
    r.animation_data.action=None
    if not bpy.data.actions.get(name+'_v1'):prior.name=name+'_v1';prior.use_fake_user=True
    else:bpy.data.actions.remove(prior)
a=bpy.data.actions.new(name);a.use_fake_user=True;r.animation_data.action=a
def env(f,pts):
    if f<=pts[0][0]:return pts[0][1]
    for (t,v),(u,w) in zip(pts,pts[1:]):
        if f<=u:
            x=(f-t)/(u-t);x=x*x*(3-2*x);return v+(w-v)*x
    return pts[-1][1]
def rot(name,axis,deg):
    b=r.pose.bones.get(name)
    if b:b.rotation_quaternion=base[name][1]@Quaternion(axis,math.radians(deg))
previous={};settled=None
for f in range(1,45):
    s.frame_set(f)
    for b in controls:
        loc,q,e,sc=base[b.name];b.location=loc;b.rotation_quaternion=q;b.rotation_euler=e;b.scale=sc
    recoil=env(f,[(1,0),(5,1),(9,.2),(14,0)])
    buckle=env(f,[(1,0),(7,0),(11,1),(14,.65),(23,1)])
    fall=env(f,[(1,0),(14,0),(18,.15),(22,.7),(26,1),(30,.97),(33,1)])
    roll=env(f,[(1,0),(5,-3),(11,10),(14,6),(18,24),(22,62),(26,94),(29,89),(33,92)])
    rot('root',(0,1,0),roll)
    root=r.pose.bones['root'];root.location.x=env(f,[(1,0),(14,0),(26,.17),(33,.18)])
    torso=r.pose.bones['torso']
    torso.location+=torso.bone.matrix_local.to_3x3().inverted()@Vector((0,.025*recoil,-.06*buckle-.07*fall))
    hurt=env(f,[(1,0),(8,1),(14,1),(27,.9),(35,.9)])
    for n,(pl,pq) in pain.items():
        b=r.pose.bones[n];b.location=base[n][0].lerp(pl,hurt);b.rotation_quaternion=base[n][1].slerp(pq,hurt)
    torso.location+=torso.bone.matrix_local.to_3x3().inverted()@Vector((0,0,-.06*buckle-.07*fall))
    # The right foreleg folds first; the left makes the last attempt to catch.
    rot('front_thigh_fk.R',(1,0,0),-18*buckle-8*fall)
    rot('front_shin_fk.R',(1,0,0),32*buckle+10*fall)
    rot('front_foot_fk.R',(1,0,0),-14*fall)
    delayed=env(f,[(1,0),(15,0),(22,.4),(29,1),(33,1)])
    rot('front_thigh_fk.L',(1,0,0),-8*buckle-20*delayed)
    rot('front_shin_fk.L',(1,0,0),10*buckle+24*delayed)
    rot('front_foot_fk.L',(1,0,0),-9*delayed)
    rot('thigh_fk.R',(1,0,0),-20*fall);rot('shin_fk.R',(1,0,0),25*fall)
    rot('thigh_fk.L',(1,0,0),-12*fall);rot('shin_fk.L',(1,0,0),18*fall)
    drape=env(f,[(1,0),(20,0),(28,.8),(35,1)])
    for n,angle in [('front_thigh_fk.R',55),('front_shin_fk.R',-15),('thigh_fk.R',48),('shin_fk.R',-12)]:
        r.pose.bones[n].rotation_quaternion @= Quaternion((0,0,1),math.radians(angle*drape))
    r.pose.bones['front_foot_fk.R'].rotation_quaternion @= Quaternion((1,0,0),math.radians(22*drape))
    rot('ear_flop.R',(0,0,1),env(f,[(1,0),(19,0),(25,-10),(31,-52),(35,-46)]))
    tail=env(f,[(1,0),(15,0),(23,5),(28,-4),(33,1),(35,0)])
    for i,n in enumerate(('spine.003','spine.002','spine.001','spine')):
        r.pose.bones[n].rotation_quaternion @= Quaternion((1,0,0),math.radians(tail*(.35+i*.2)))
    twitch=env(f,[(1,0),(36,0),(38,1),(39,-.2),(41,0)])
    r.pose.bones['spine'].rotation_quaternion @= Quaternion((0,0,1),math.radians(4*twitch))
    bpy.context.view_layer.update()
    for n,direction in [('front_thigh_fk.R',(-1,-.3,.12)),('front_shin_fk.R',(-.7,-1,.15)),('thigh_fk.R',(-1,.4,.15)),('shin_fk.R',(-.7,-1,.15))]:
        b=r.pose.bones[n];mat=b.matrix.copy();q=mat.to_quaternion()
        swing=(q@Vector((0,1,0))).rotation_difference(Vector(direction).normalized())
        desired=q.slerp(swing@q,min(1,drape))
        b.matrix=Matrix.LocRotScale(mat.translation,desired,mat.to_scale())
        bpy.context.view_layer.update()
    if f<=35:
        root.location.z+=floor-minimum()
        if f==35:settled=root.location.copy()
    else:root.location=settled
    bpy.context.view_layer.update()
    for b in controls:
        if b.rotation_mode=='QUATERNION':
            if b.name in previous and b.rotation_quaternion.dot(previous[b.name])<0:b.rotation_quaternion.negate()
            previous[b.name]=b.rotation_quaternion.copy()
        for p in ('location','rotation_quaternion' if b.rotation_mode=='QUATERNION' else 'rotation_euler','scale'):
            b.keyframe_insert(p,frame=f,group=b.name)
    for prefix in ('front_',''):
        for side in ('L','R'):
            b=r.pose.bones[prefix+'thigh_parent.'+side];b['IK_FK']=1.0
            b.keyframe_insert('["IK_FK"]',frame=f,group=b.name)
for l in a.layers:
    for st in l.strips:
        for bag in st.channelbags:
            for fc in bag.fcurves:
                for k in fc.keyframe_points:k.interpolation='LINEAR'
a.use_frame_range=True;a.frame_start=1;a.frame_end=44
for label,f in [('Flinch',5),('Failed catch',14),('Collapse',22),('Settle',35),('Tail twitch',38),('Still',41)]:
    a.pose_markers.new(label).frame=f
for other in bpy.data.actions:
    if other==a:continue
    r.animation_data.action=other
    b=r.pose.bones['ear_flop.R'];b.rotation_quaternion=Quaternion()
    for f in (1,int(other.frame_range[1])):b.keyframe_insert('rotation_quaternion',frame=f,group=b.name)
r.animation_data.action=a
s.frame_start=1;s.frame_end=44;s.use_preview_range=False;s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=target)
print(json.dumps({'saved':target,'action':name,'floor':floor,'frames':list(a.frame_range)}))
'''
print(send(CODE,'127.0.0.1',9876,timeout=120))
