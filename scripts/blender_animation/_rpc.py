"""Reuse the repository's Blender RPC client when recipes run as scripts.

Importing this module does not contact Blender. Recipe modules may execute edits
at import time: run them deliberately, never import them for discovery/testing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blender_inspect import send  # noqa: E402,F401
