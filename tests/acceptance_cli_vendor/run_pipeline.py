"""Invoke pipeline.py under the inherited offline guard and reject ComfyUI startup."""

import runpy
import sys
from pathlib import Path


def forbid_comfyui_import(event, args):
    if event == "import" and args[0] == "image_to_3dlab.comfyui_backend":
        raise AssertionError("Hunyuan policy should refuse before importing its adapter")


pipeline = Path(__file__).resolve().parents[2] / "pipeline.py"
sys.path.insert(0, str(pipeline.parent))
sys.addaudithook(forbid_comfyui_import)
runpy.run_path(str(pipeline), run_name="__main__")
