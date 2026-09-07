import bpy
from mathutils import Vector
import runpy
from pathlib import Path
setup=runpy.run_path(str(Path(__file__).with_name('forest_flicker_preview.py')),run_name='preview_setup')
s=setup['s']
s.frame_set(31)
s.camera.location=(-3,-4,2.25)
s.camera.rotation_euler=(Vector((0,0,.8))-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.render.filepath='/private/tmp/forest-paw-v7-reverse31.png'
bpy.ops.render.render(write_still=True)
s.frame_set(44)
s.render.filepath='/private/tmp/forest-paw-v7-reverse44.png'
bpy.ops.render.render(write_still=True)
