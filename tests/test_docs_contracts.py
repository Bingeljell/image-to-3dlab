"""The setup docs say what the tools actually do.

Read the files as text. Sizes come from the downloader, not from a second copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _weights():
    path = REPO / "hunyuan_mlx" / "download_weights.py"
    spec = importlib.util.spec_from_file_location("download_weights_docs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_weights_docs"] = module
    spec.loader.exec_module(module)
    return module


def test_readme_does_not_promise_a_silent_download():
    readme = (REPO / "README.md").read_text()
    assert "auto-downloaded" not in readme
    assert "downloads on first use" not in readme
    assert "weights automatically" not in readme
    assert "--allow-rembg" not in readme
    assert "about 16.5 GB" in readme
    assert "asks before it fetches" in readme


def test_agents_states_both_bria_refusals():
    agents = (REPO / "AGENTS.md").read_text()
    assert "refuse an unpatched tree" in agents
    assert "rembg_model = None" in agents


def test_mlx_recipe_matches_the_default_download():
    dw = _weights()
    recipe = (REPO / "docs" / "hunyuan-mlx-recipes.md").read_text()
    total = dw.APPROX_GB["2.0"] + dw.APPROX_GB["paint"]
    assert f"{total:.1f} GB" in recipe
    assert "the EU, the UK or South Korea" in recipe
    assert "asks before it fetches" in recipe
    assert "`--all`" in recipe
    assert "2.1, 2.0, 2.0-turbo" not in recipe


def test_downloader_docstring_defaults_to_model_2_0():
    doc = _weights().__doc__
    assert "omit for" not in doc
    assert "all three" not in doc
    assert "2.0" in doc
    assert "--all" in doc


def test_prop_sheets_keep_picture_rights_off_the_mesh():
    text = (REPO / "docs" / "prop-sheets.md").read_text()
    assert "The pictures Qwen-Image makes are yours" in text
    assert "and so are the props" not in text
    assert "Pixal3D" in text
    assert "DINOv3" in text
    assert "commercial-conditional" in text


def test_shape_readme_commands_carry_yes():
    text = (REPO / "hunyuan_mlx" / "shape" / "README.md").read_text()
    commands = [
        line for line in text.splitlines()
        if "dl_modelscope.py" in line or "dl_any.py" in line
        if not line.strip().startswith("#")
    ]
    assert commands
    assert all("--yes" in line for line in commands)


def test_colour_check_does_not_fetch_tinyclip():
    text = (REPO / "docs" / "trellis2-flat-illustration-colour-drift.md").read_text()
    assert "TinyCLIP" in text
    assert "does not fetch" in text


def test_paint_readme_does_not_promise_every_model():
    text = (REPO / "hunyuan_mlx" / "paint" / "README.md").read_text()
    assert "auto-downloaded" not in text
    assert "omit for" not in text
