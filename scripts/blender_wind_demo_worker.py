"""Bend labelled foliage and render a sequence in a private scene.

Run (internal): bundled and sent to Blender by blender_wind_demo.py.
"""

import math


def run(arguments: dict) -> dict:
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    scene_name = "IMG3D wind"
    previous_scene = bpy.context.window.scene
    scene = bpy.data.scenes.get(scene_name) or bpy.data.scenes.new(scene_name)
    bpy.context.window.scene = scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new(scene.name + " World")
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for prior in list(scene.collection.children):
        bpy.data.collections.remove(prior)
    result = render_wind(bpy, np, Vector, KDTree, scene, arguments)
    bpy.context.window.scene = previous_scene
    return result


def load_mesh(bpy, render_scene, path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [o for o in render_scene.objects if o not in before]
    meshes = [o for o in added if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path}")
    for o in render_scene.objects:
        o.select_set(False)
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def moving_labels(np, label_mesh):
    if not label_mesh.color_attributes:
        raise RuntimeError("label mesh has no color labels")
    # Read the label colours. glTF may land them on points or on face corners.
    attribute = label_mesh.color_attributes[0]
    count = len(attribute.data)
    raw = np.zeros(count * 4, dtype=np.float32)
    attribute.data.foreach_get("color", raw)
    raw = raw.reshape(-1, 4)[:, :3]
    if attribute.domain == "CORNER":
        per_vertex = np.zeros((len(label_mesh.vertices), 3), dtype=np.float32)
        loop_vertex = np.zeros(len(label_mesh.loops), dtype=np.int32)
        label_mesh.loops.foreach_get("vertex_index", loop_vertex)
        per_vertex[loop_vertex] = raw
        raw = per_vertex

    # glTF stores colour linearly; the palette was authored as sRGB. Only the winning
    # channel matters here, so compare channels rather than converting.
    is_foliage = (raw[:, 1] > raw[:, 0]) & (raw[:, 1] >= raw[:, 2])
    is_flower = (raw[:, 2] > raw[:, 0]) & (raw[:, 2] > raw[:, 1])
    moves = is_foliage | is_flower
    return moves


def foliage_attachments(np, Vector, KDTree, label_mesh, body_mesh, moves):
    label_points = np.zeros(len(label_mesh.vertices) * 3, dtype=np.float32)
    label_mesh.vertices.foreach_get("co", label_points)
    label_points = label_points.reshape(-1, 3)

    # Match the textured mesh to the labelled one by position: exporting through trimesh
    # does not preserve vertex order, so indices cannot be trusted between the two.
    if len(label_points) == 0:
        raise RuntimeError("label mesh has no vertices")
    lookup = KDTree(len(label_points))
    for index, point in enumerate(label_points):
        lookup.insert(Vector(point), index)
    lookup.balance()

    base = np.zeros(len(body_mesh.vertices) * 3, dtype=np.float32)
    body_mesh.vertices.foreach_get("co", base)
    base = base.reshape(-1, 3)
    if len(base) == 0:
        raise RuntimeError("render mesh has no vertices")

    moving = np.zeros(len(base), dtype=bool)
    for index, point in enumerate(base):
        _, nearest, _ = lookup.find(Vector(point))
        moving[index] = moves[nearest]

    # Stiffness: distance to the nearest rigid (body) vertex, normalised. This is what
    # makes the tail bend instead of sliding -- it is anchored where it meets the hip and
    # free at the tips, and it costs nothing to derive because the labels already say
    # which vertices are rigid.
    rigid_points = label_points[~moves]
    if len(rigid_points) == 0:
        raise RuntimeError("label mesh has no rigid anchor vertices")
    anchors = KDTree(len(rigid_points))
    for index, point in enumerate(rigid_points):
        anchors.insert(Vector(point), index)
    anchors.balance()

    stiffness = np.zeros(len(base), dtype=np.float32)
    # Keep the anchor position too, not just the distance: bending has to pivot about the
    # point where the foliage attaches, and each frond has its own attachment point.
    anchor_of = base.copy()
    for index in np.nonzero(moving)[0]:
        position, _, distance = anchors.find(Vector(base[index]))
        stiffness[index] = distance
        anchor_of[index] = position
    if stiffness.max() > 0:
        stiffness /= stiffness.max()
    stiffness = stiffness * stiffness * (3.0 - 2.0 * stiffness)  # ease the falloff
    return base, moving, stiffness, anchor_of


def set_up_render(bpy, Vector, scene, base, arguments):
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = arguments["resolution"]
    scene.render.resolution_y = arguments["resolution"]
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "Medium High Contrast"
    scene.world.color = arguments["world_color"]
    try:
        scene.eevee.use_raytracing = arguments["raytracing"]
    except AttributeError:
        pass

    lo = base.min(axis=0)
    hi = base.max(axis=0)
    centre = Vector(((lo + hi) * 0.5).tolist())
    size = float(max(hi - lo))

    camera_data = bpy.data.cameras.new("CAM")
    camera = bpy.data.objects.new("CAM", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 60
    scene.camera = camera

    def aim(obj, target):
        obj.rotation_euler = (
            (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
        )

    for suffix, location, energy, radius in (
        ("KEY", (-2.8, -3.8, 4.5), 700, 4.0),
        ("FILL", (3.2, -1.8, 2.7), 420, 3.5),
        ("RIM", (1.0, 3.5, 4.0), 600, 3.0),
    ):
        light_data = bpy.data.lights.new(suffix, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = radius
        light = bpy.data.objects.new(suffix, light_data)
        scene.collection.objects.link(light)
        light.location = Vector(location) * size
        aim(light, centre)

    angle = math.radians(arguments["azimuth"])
    distance = size * 1.9 / arguments["zoom"]
    camera.location = centre + Vector(
        (-math.sin(angle) * distance, -math.cos(angle) * distance, size * 0.10)
    )
    aim(camera, centre)


def render_wind(bpy, np, Vector, KDTree, scene, arguments):
    body_obj = load_mesh(bpy, scene, arguments["mesh_path"])
    body_obj.name = arguments["body_name"]
    label_obj = load_mesh(bpy, scene, arguments["labels_path"])
    label_obj.name = arguments["labels_name"]

    moves = moving_labels(np, label_obj.data)
    body_mesh = body_obj.data
    base, moving, stiffness, anchor_of = foliage_attachments(
        np, Vector, KDTree, label_obj.data, body_mesh, moves
    )

    print(
        f"WIND:: moving {int(moving.sum())} of {len(base)} vertices, max stiffness {stiffness.max():.3f}"
    )

    extent = float(np.ptp(base, axis=0).max())
    # Phase varies with position so the wind travels across the tail as a wave rather than
    # moving every leaf in lockstep.
    phase = (base[:, 0] + base[:, 2]) / max(extent, 1e-6) * arguments["wave"]

    # Smoothly varying phase alone still moves neighbouring fronds almost together, which
    # reads as one sheet rather than a bushy mass. Give each small clump its own offset by
    # hashing a coarse grid of positions, so clumps flutter independently while staying
    # coherent within themselves.
    cell = np.floor(base / max(extent * arguments["cluster_size"], 1e-6)).astype(
        np.int64
    )
    clump = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    phase = phase + (clump % 997) / 997.0 * (2.0 * math.pi) * arguments["jitter"]

    # Bending, not sliding. Translating every vertex along one direction is a shear: the
    # tail slides bodily, its tip traces no arc, and the volume never turns to show its
    # depth -- which is what makes a mesh with real thickness read as a flat cutout.
    # Rotating about the attachment point instead preserves length and sweeps the mass
    # through an arc.
    offset_from_anchor = base - anchor_of
    axis = np.array([0.0, arguments["axis_y"], arguments["axis_z"]], dtype=np.float32)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
    axis_dot = offset_from_anchor @ axis
    axis_cross = np.cross(
        np.broadcast_to(axis, offset_from_anchor.shape), offset_from_anchor
    )
    max_angle = math.radians(arguments["bend"])

    set_up_render(bpy, Vector, scene, base, arguments)

    label_obj.hide_render = True

    flat = np.zeros(base.size, dtype=np.float32)
    written = []
    frames = arguments["frames"]
    for index in range(frames):
        t = index / frames
        # Two waves at different rates so the motion does not read as a single metronome,
        # plus a slow gust that swells and fades across the loop.
        swing = (
            np.sin(2.0 * math.pi * (arguments["frequency"] * t) + phase) * 0.65
            + np.sin(2.0 * math.pi * (arguments["frequency"] * 1.7 * t) + phase * 1.4)
            * 0.35
        )
        strength = 1.0 + arguments["gust"] * math.sin(2.0 * math.pi * t)
        angle = (swing * strength * max_angle * stiffness).astype(np.float32)

        # Rodrigues rotation of each vertex about its own anchor, by its own angle.
        cos_a = np.cos(angle)[:, None]
        sin_a = np.sin(angle)[:, None]
        rotated = (
            offset_from_anchor * cos_a
            + axis_cross * sin_a
            + np.broadcast_to(axis, offset_from_anchor.shape)
            * axis_dot[:, None]
            * (1.0 - cos_a)
        )
        moved = anchor_of + rotated

        # A little high-frequency flutter on the tips, on top of the bend, so the surface
        # shimmers instead of moving as one rigid fan.
        flutter = np.sin(
            2.0 * math.pi * (arguments["frequency"] * 3.1 * t) + phase * 2.3
        )
        moved[:, 1] += flutter * stiffness * extent * arguments["flutter"] * strength

        flat[:] = moved.reshape(-1)
        body_mesh.vertices.foreach_set("co", flat)
        body_mesh.update()

        path = f"{arguments['frame_dir']}/{arguments['label']}_{index:04d}.png"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)

    return {"frames": len(written)}
