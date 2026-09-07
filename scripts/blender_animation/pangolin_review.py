import bpy,json,sys
from pathlib import Path
from mathutils import Vector
s=bpy.context.scene;r=bpy.data.objects['rig']
s.render.engine='BLENDER_WORKBENCH'
s.render.resolution_x=640;s.render.resolution_y=480;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'
sh=s.display.shading;sh.light='STUDIO';sh.studio_light='paint.sl';sh.color_type='TEXTURE'
sh.show_shadows=True;sh.show_cavity=True;sh.background_type='WORLD';s.world.color=(.06,.07,.08)
s.view_settings.view_transform='Standard'
cam=bpy.data.objects.new('ReviewCamera',bpy.data.cameras.new('ReviewCamera'));s.collection.objects.link(cam);s.camera=cam
cam.location=(4,-3,2.6);cam.rotation_euler=(Vector((0,.3,.5))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=3.8
out=Path('/private/tmp/pangolin-impact' if '--impact' in sys.argv else '/private/tmp/pangolin-review');out.mkdir(exist_ok=True)
if '--v2' in sys.argv:out=Path('/private/tmp/pangolin-impact-v2');out.mkdir(exist_ok=True)
jobs=[('swipe','Pangolin_Attack_TailSwipe_CW180_Sprite_v04_Whip',100),('slam','Pangolin_Special_RearSlam_Sprite_v05_Heavy',120)] if '--impact' in sys.argv else [('swipe','Pangolin_Attack_TailSwipe_CW180_Sprite_v03',100),('slam','Pangolin_Special_RearSlam_Sprite_v04',120)]
if '--v2' in sys.argv:jobs=[('swipe','Pangolin_Attack_TailSwipe_CW180_Sprite_v05_WhipHold',100),('slam','Pangolin_Special_RearSlam_Sprite_v06_Heavy60',60)]
if '--v3' in sys.argv:
    out=Path('/private/tmp/pangolin-impact-v3');out.mkdir(exist_ok=True)
    jobs=[('swipe','Pangolin_Attack_TailSwipe_CW180_Sprite_v06_SoftRecoil',100)]
for label,name,end in jobs:
    if '--slam-only' in sys.argv and label!='slam':continue
    r.animation_data.action=bpy.data.actions[name]
    s.frame_start=1;s.frame_end=end;s.render.filepath=str(out/(label+'_'))
    bpy.ops.render.render(animation=True)
