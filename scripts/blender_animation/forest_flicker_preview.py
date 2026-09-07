"""Render local combat review clips without modifying the saved asset."""
import bpy
import sys
from pathlib import Path
from mathutils import Vector
s=bpy.context.scene;r=bpy.data.objects['rig']
out=Path('/Users/nikhilshahane/projects/image-to-3dlab/output/forest-flicker-review')
s.render.engine='BLENDER_WORKBENCH'
s.render.resolution_x=960;s.render.resolution_y=720;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'
sh=s.display.shading
sh.light='STUDIO';sh.studio_light='paint.sl';sh.color_type='TEXTURE';sh.show_specular_highlight=False
sh.show_cavity=True;sh.cavity_type='BOTH';sh.background_type='WORLD';s.world.color=(.06,.07,.08)
s.view_settings.view_transform='Standard';s.view_settings.look='None'
s.camera.location=(3,-4,2.25)
s.camera.rotation_euler=(Vector((0,0,.8))-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO';s.camera.data.ortho_scale=3.6
jobs=[('death','ForestFlicker_Death_44f',44)] if '--death' in sys.argv else [('swipe','ForestFlicker_Attack_RightSwipe_Polished',32),('slam','ForestFlicker_Special_DoublePawSlam_Polished',44)]
if __name__ == '__main__':
    for label,name,end in jobs:
        folder=out/label;folder.mkdir(parents=True,exist_ok=True)
        r.animation_data.action=bpy.data.actions[name]
        if r.animation_data.action.slots:r.animation_data.action_slot=r.animation_data.action.slots[0]
        s.frame_start=1;s.frame_end=end;s.render.filepath=str(folder/'frame_')
        bpy.ops.render.render(animation=True)
