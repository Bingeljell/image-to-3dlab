"""Picture zoom in the viewer (viewer/studio/zoom.js), run under Node against the shipped file.

What matters: the spot under the pointer stays put while zooming, zoom stays between 1x and the
cap, and panning never shows past the picture's edge.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "zoom.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@needs_node
def test_the_point_under_the_pointer_stays_put():
    out = _run("""const s = m.zoomAt(m.RESET, 2, 100, 50, 400, 400);
      console.log(JSON.stringify([s, (100 - s.x) / s.scale, (50 - s.y) / s.scale]));""")
    assert out[0]["scale"] == 2 and out[1] == 100 and out[2] == 50


@needs_node
def test_zoom_stays_between_one_and_the_cap():
    out = _run("""console.log(JSON.stringify([m.zoomAt(m.RESET, 0.5, 10, 10, 400, 400).scale,
      m.zoomAt({scale: 7, x: 0, y: 0}, 3, 0, 0, 400, 400).scale, m.MAX_ZOOM]));""")
    assert out[0] == 1 and out[1] == out[2]


@needs_node
def test_panning_never_shows_past_the_edge():
    out = _run("""const s = {scale: 2, x: -100, y: -100};
      console.log(JSON.stringify([m.panBy(s, 500, 500, 400, 400), m.panBy(s, -900, -900, 400, 400)]));""")
    assert out[0] == {"scale": 2, "x": 0, "y": 0}
    assert out[1] == {"scale": 2, "x": -400, "y": -400}


@needs_node
def test_a_trackpad_flick_zooms_gently_and_a_wheel_notch_is_capped():
    out = _run("""console.log(JSON.stringify([m.wheelFactor(-4), m.wheelFactor(-120), m.wheelFactor(120), m.wheelFactor(0)]));""")
    tiny, notch, back, none = out
    assert 1 < tiny < 1.01 and notch == pytest.approx(1.06) and back == pytest.approx(1 / 1.06) and none == 1


@needs_node
def test_a_zoom_button_keeps_the_middle_in_the_middle():
    out = _run("""const s = m.zoomTo(m.RESET, 2, 400, 300);
      console.log(JSON.stringify([s.scale, (200 - s.x) / s.scale, (150 - s.y) / s.scale, m.STEPS]));""")
    assert out[:3] == [2, 200, 150] and out[3] == [1, 1.25, 1.5, 2, 3, 4]
