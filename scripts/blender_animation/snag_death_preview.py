"""Render a temporary studio preview; never save changes into the scene file."""
import bpy
from mathutils import Vector
from pathlib import Path

s=bpy.context.scene
folder=Path(bpy.data.filepath).parent/'death_review_preview'
folder.mkdir(exist_ok=True)
s.render.engine='BLENDER_WORKBENCH'
s.render.resolution_x=960
s.render.resolution_y=720
s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'
s.render.film_transparent=False
s.display.shading.light='STUDIO'
s.display.shading.studio_light='paint.sl'
s.display.shading.color_type='TEXTURE'
s.display.shading.show_shadows=True
s.display.shading.show_cavity=True
s.display.shading.cavity_type='BOTH'
s.display.shading.background_type='WORLD'
s.world.color=(.055,.055,.055)
s.display.shading.show_specular_highlight=False
s.view_settings.view_transform='Standard'
s.view_settings.look='None'
s.camera.location=(1.3,-2.4,1.05)
target=Vector((-.09,-.1,-.025))
s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.type='ORTHO'
s.camera.data.ortho_scale=1.65
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.215))
floor=bpy.context.object
floor.name='Preview ground (temporary)'
mat=bpy.data.materials.new('Preview ground')
mat.diffuse_color=(.12,.14,.16,1)
floor.data.materials.append(mat)
s.render.filepath=str(folder/'frame_')
bpy.ops.render.render(animation=True)
