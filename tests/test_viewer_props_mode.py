"""Static wiring checks for the Props mode.

`props.js` touches the DOM at import time, so like `finish.js` it is checked by reading
it beside `index.html`: a `getElementById` that matches no id yields null, and the tab
breaks the moment someone uses it. The endpoints are checked against the server's own
routes for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
INDEX = (VIEWER / "index.html").read_text()
APP = (VIEWER / "app.js").read_text()
PROPS = (VIEWER / "modes" / "props.js").read_text()
SERVER = (VIEWER / "generate_api.py").read_text()


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', markup))


def test_every_id_the_props_mode_looks_up_exists_in_the_page():
    referenced = set(re.findall(r"f\('([^']+)'\)", PROPS))
    assert referenced, "the module should look up some elements"
    missing = sorted(referenced - _element_ids(INDEX))
    assert not missing, f"props.js references ids absent from index.html: {missing}"


def test_the_props_mode_is_imported_registered_and_has_a_button():
    assert "import './modes/props.js';" in APP
    assert "props: byId('props-view')" in APP
    assert "modes.props.hidden = activeMode !== 'props';" in APP
    assert {"mode-props", "props-view"} <= _element_ids(INDEX)


def test_its_stylesheet_is_linked():
    assert (VIEWER / "styles" / "props.css").is_file()
    assert '<link rel="stylesheet" href="./styles/props.css">' in INDEX


def test_the_props_mode_uses_the_routes_the_server_has():
    for url in ("'/api/props'", "'/api/props/runs'", "/api/props/${jobId}/status",
                "/api/props/${jobId}/cancel", "/turn`"):
        assert url in PROPS, url
    assert 'parts == ["api", "props"]' in SERVER
    assert 'parts == ["api", "props", "runs"]' in SERVER
    assert 'parts[:3] == ["api", "props", "runs"] and parts[4] == "turn"' in SERVER
    assert 'parts[:2] == ["api", "props"] and parts[3] == "cancel"' in SERVER


def test_the_progress_track_shares_the_styled_markup():
    assert '<div class="progress-track"><div id="props-overall-bar"></div></div>' in INDEX


def test_a_missing_gltfpack_is_said_rather_than_silently_skipped():
    assert "gltfpack not found" in PROPS
    assert "vendor/gltfpack/gltfpack" in PROPS


def test_a_near_45_degree_turn_offers_the_quarter_turn_fix():
    assert "prop.yaw_tie" in PROPS
    assert "Turn 90°" in PROPS
    assert "degrees: 90" in PROPS


def test_the_preview_is_the_restricted_compare_page():
    assert "/viewer/index.html?a=${encodeURIComponent(src)}" in PROPS
    assert "restricted=1" in PROPS


def test_a_prop_bake_and_the_other_heavy_jobs_refuse_to_share_the_machine():
    """They hold gigabytes in the same unified memory, so none starts over another."""
    assert "def _props_busy(self)" in SERVER
    assert "a finishing job is running; wait for it to finish" in SERVER
    # Generation, a new finish, a finish resume and a rig rebind each check for a bake.
    assert SERVER.count("if _props_baking():") == 4


def test_the_tab_sits_under_the_menu_bar_like_generate():
    """A plain section scrolls its heading under the fixed 42px menu."""
    base = (VIEWER / "styles" / "base.css").read_text()
    rule = base[:base.index("position: fixed; inset: 42px 0 0 0")]
    assert "#props-view" in rule.rsplit("}", 1)[-1]


def test_every_prop_gets_a_chip_and_one_detail_panel():
    assert "chip.onclick = () => showProp(run, prop);" in PROPS
    assert {"props-list", "props-detail", "props-frame", "props-turn", "props-lods-body"} <= _element_ids(INDEX)


def test_the_result_says_which_licence_the_props_inherit():
    assert "props-licence" in _element_ids(INDEX)
    assert "The props inherit it." in PROPS
    assert "check it before shipping the props" in PROPS


def test_a_job_the_server_forgot_does_not_leave_the_tab_running():
    """After a restart the status route 404s forever; polling on would never stop."""
    assert "if (response.status === 404) {" in PROPS


def test_the_lod_table_shows_what_each_file_holds():
    """The count read from the file, falling back to the target only when it can't be read."""
    assert "lod.triangles ?? targets[lod.index]" in PROPS


def test_the_note_names_every_extension_the_web_files_need():
    for extension in ("KHR_mesh_quantization", "EXT_meshopt_compression", "EXT_texture_webp"):
        assert extension in INDEX


def test_a_generated_model_that_lost_its_record_says_so():
    """Not "Uploaded model": the user picked it from what this machine generated."""
    assert "licence.source === 'generated'" in PROPS
    assert "no licence record was kept" in PROPS


def test_cancel_is_wired_when_the_job_is_asked_for_not_when_it_answers():
    begin = PROPS.split("function begin(", 1)[1].split("\n}\n", 1)[0]
    watch = PROPS.split("function watch(", 1)[1].split("\n}\n", 1)[0]
    assert "f('props-cancel').onclick" in begin
    assert "earlyCancel(" in begin
    assert "onclick" not in watch
    assert "state.cancel.started(payload.job_id)" in watch


def test_a_preview_iframe_does_not_list_the_runs():
    """Every preview loads the app with ?restricted=1, and listing reads every LOD.

    The runs are listed on arriving at the tab, which a preview never does, and on
    nothing at load time.
    """
    assert "if (event.detail?.mode === 'props' && !state.running) loadRuns();" in PROPS
    assert not re.search(r"^(?:if \(.*\) )?loadRuns\(\);$", PROPS, re.M)
    assert "import './modes/props.js';" in APP
    assert APP.index("import './modes/props.js';") < APP.index("setMode(landingMode(")


def test_a_missing_blender_is_explained_in_the_servers_words():
    assert "tools.blender_problem" in PROPS
    assert "I2L_BLENDER" not in PROPS

