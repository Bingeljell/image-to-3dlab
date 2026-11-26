"""Return the Blender text that renders in a private scene.

Run (internal): imported by live Blender scripts. Not a command.

A long-lived Blender session holds the user's scenes. Wiping every object, which
these helpers once did, emptied unsaved work. The text below switches to a scene
this repo owns, clears only that scene, and the caller puts the window back.
"""

from __future__ import annotations

from inspect import getsource


def clear_private_scene(bpy, scene):
    """Detach the private scene's links and remove only unused data."""
    scene.camera = None
    objects = set()

    def detach_links(collection):
        for obj in list(collection.objects):
            collection.objects.unlink(obj)
            objects.add(obj)
        children = list(collection.children)
        for child in children:
            collection.children.unlink(child)
        return children

    pending = detach_links(scene.collection)
    if pending or objects:
        # Flush the view layer's references to the links just detached.
        bpy.context.view_layer.update()
    while pending:
        child = pending.pop()
        if child.users == 0 and not any(bpy.data.user_map(subset=[child]).values()):
            pending.extend(detach_links(child))
            bpy.data.collections.remove(child, do_unlink=False)

    # Blender's user count excludes some links, including parent pointers.
    while objects:
        references = bpy.data.user_map(subset=objects)
        unused = {obj for obj in objects if obj.users == 0 and not references.get(obj)}
        if not unused:
            break
        objects.difference_update(unused)
        for obj in unused:
            bpy.data.objects.remove(obj, do_unlink=False)


def isolated_scene_prelude(scene_name: str) -> str:
    """Blender statements that enter `scene_name` and clear only that scene."""
    quoted = repr(scene_name)
    return (
        getsource(clear_private_scene)
        + "\n"
        "previous_scene = bpy.context.window.scene\n"
        f"render_scene = bpy.data.scenes.get({quoted}) or bpy.data.scenes.new({quoted})\n"
        "bpy.context.window.scene = render_scene\n"
        "if render_scene.world is None:\n"
        '    render_scene.world = bpy.data.worlds.new(render_scene.name + " World")\n'
        "clear_private_scene(bpy, render_scene)\n"
    )


def restore_previous_scene() -> str:
    """Blender statement that hands the window back to the scene we found."""
    return "bpy.context.window.scene = previous_scene\n"


def private_scene_collections(bpy, scene):
    """Find collections that can be edited without changing another scene."""
    private = {scene.collection, *scene.collection.children_recursive}
    for other in bpy.data.scenes:
        if other != scene:
            private.difference_update((other.collection, *other.collection.children_recursive))
    # An instance can make a collection visible in another scene without linking
    # it into that scene's collection tree.
    collection_users = bpy.data.user_map(subset=private)
    for collection in list(private - {scene.collection}):
        if any(user not in private and user != scene for user in collection_users.get(collection, ())):
            private.difference_update((collection, *collection.children_recursive))

    return private


def remove_scene_objects(bpy, objects):
    """Unlink selected objects from the active scene's private collections."""
    if not objects:
        return []
    scene = bpy.context.scene
    private = private_scene_collections(bpy, scene)
    removed = []
    object_users = bpy.data.user_map(subset=set(objects))
    for obj in objects:
        local_links = set(obj.users_collection) & private
        if not local_links:
            continue
        name = obj.name
        external_users = any(
            user not in local_links and user != scene for user in object_users.get(obj, ())
        )
        for collection in local_links:
            collection.objects.unlink(obj)
        if obj.users == 0 and not external_users:
            bpy.data.objects.remove(obj)
        if name not in scene.objects:
            removed.append(name)
    return removed


def import_gltf_objects(bpy, path, *, require_mesh=False):
    """Return new glTF objects, discarding partial imports on failure."""
    before = set(bpy.data.objects)
    try:
        result = bpy.ops.import_scene.gltf(filepath=path)
        if "FINISHED" not in result:
            raise RuntimeError(f"could not import {path}")
        imported = [obj for obj in bpy.data.objects if obj not in before]
        if require_mesh and not any(obj.type == "MESH" for obj in imported):
            raise RuntimeError("no mesh imported")
        return imported
    except Exception:
        for obj in list(bpy.data.objects):
            if obj not in before:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise



def scene_marker_objects(scene, prefix):
    """Find markers by their logical names even when another scene owns the exact object name."""
    markers = {}
    staged_names = {}
    for obj in scene.objects:
        if obj.type == "EMPTY" and obj.name.startswith(prefix):
            name = obj.get("assetfurnace_marker_name")
            if name and name.startswith(prefix):
                staged_names[name[len(prefix):]] = obj
            else:
                markers[obj.name[len(prefix):]] = obj
    return markers | staged_names
