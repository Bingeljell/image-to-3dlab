"""Render a temporary studio preview of the attack without saving scene edits."""
import bpy
from mathutils import Vector
from pathlib import Path
s=bpy.context.scene
folder=Path(bpy.data.filepath).parent/'attack_review_preview'
folder.mkdir(exist_ok=True)
s.render.engine='BLENDER_WORKBENCH'
s.render.resolution_x=960; s.render.resolution_y=720; s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'
s.render.film_transparent=False
sh=s.display.shading
sh.light='STUDIO'; sh.studio_light='paint.sl'; sh.color_type='TEXTURE'
sh.show_shadows=True; sh.show_cavity=True; sh.cavity_type='BOTH'
sh.background_type='WORLD'; s.world.color=(.055,.055,.055)
sh.show_specular_highlight=False
s.view_settings.view_transform='Standard'; s.view_settings.look='None'
s.camera.location=(1.4,-2.3,.95)
target=Vector((0,-.13,.055))
s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO'; s.camera.data.ortho_scale=1.65
s.render.filepath=str(folder/'frame_')
bpy.ops.render.render(animation=True)
