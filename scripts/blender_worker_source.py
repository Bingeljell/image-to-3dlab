"""Pack a Python worker and its arguments for a live Blender connection.

Run (internal): imported by live Blender scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

_SCENE_WORKERS = frozenset({
    "blender_agent_worker.py",
    "blender_bake_normals_worker.py",
    "blender_render_asset_worker.py",
    "blender_rig_weights_worker.py",
    "blender_scene_query_worker.py",
    "blender_scene_stage_worker.py",
    "blender_solidify_bake_worker.py",
    "blender_split_regions_worker.py",
    "blender_turntable_worker.py",
    "blender_wind_demo_worker.py",
})


def build_worker_code(worker_path: Path, arguments: dict[str, object]) -> str:
    """Send the source bytes so the remote Blender needs no checkout."""
    source = worker_path.read_bytes()
    argument_json = json.dumps(arguments, allow_nan=False)
    setup = ""
    if worker_path.name in _SCENE_WORKERS:
        scene_path = Path(__file__).with_name("blender_scene.py")
        scene_source = scene_path.read_text(encoding="utf-8")
        setup = f"    exec(compile({scene_source!r}, {scene_path.name!r}, 'exec'), worker)\n"
    return f'''def _assetfurnace_run_worker():
    import json

    worker = {{"__name__": "_assetfurnace_worker", "__package__": ""}}
{setup}    exec(compile({source!r}, {worker_path.name!r}, "exec", dont_inherit=True), worker)
    return worker["run"](json.loads({argument_json!r}))


result = _assetfurnace_run_worker()
'''
