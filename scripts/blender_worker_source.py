"""Pack a Python worker and its arguments for a live Blender connection.

Run (internal): imported by live Blender scripts.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_worker_code(worker_path: Path, arguments: dict[str, object]) -> str:
    """Send the source bytes so the remote Blender needs no checkout."""
    source = worker_path.read_bytes()
    argument_json = json.dumps(arguments, allow_nan=False)
    return f'''def _assetfurnace_run_worker():
    import json

    worker = {{"__name__": "_assetfurnace_worker"}}
    exec(compile({source!r}, {worker_path.name!r}, "exec", dont_inherit=True), worker)
    return worker["run"](json.loads({argument_json!r}))


result = _assetfurnace_run_worker()
'''
