"""Static wiring checks for the Finish mode and the Pixal3D backend option.

`finish.js` touches the DOM at import time, so it cannot be executed under Node the way
`job-progress.js` is. The failure mode worth catching is cheaper than that anyway: a
`getElementById` in the module that no longer matches an id in `index.html` silently yields
null, and the panel breaks at the moment someone tries to use it. These tests read both
files and compare.
"""

from __future__ import annotations

import re
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
INDEX = (VIEWER / "index.html").read_text()
APP = (VIEWER / "app.js").read_text()
FINISH = (VIEWER / "modes" / "finish.js").read_text()
GENERATE = (VIEWER / "modes" / "generate.js").read_text()


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', markup))


def test_every_id_the_finish_mode_looks_up_exists_in_the_page():
    referenced = set(re.findall(r"f\('([^']+)'\)", FINISH))
    assert referenced, "the module should look up some elements"
    missing = sorted(referenced - _element_ids(INDEX))
    assert not missing, f"finish.js references ids absent from index.html: {missing}"


def test_the_finish_mode_is_imported_and_registered():
    assert "import './modes/finish.js';" in APP
    assert "finish: byId('finish-view')" in APP
    assert "modes.finish.hidden = activeMode !== 'finish';" in APP


def test_the_page_has_a_finish_button_and_section():
    ids = _element_ids(INDEX)
    assert "mode-finish" in ids
    assert "finish-view" in ids


def test_the_finish_mode_posts_to_the_documented_endpoints():
    assert "'/api/finish'" in FINISH
    assert "/api/finish/${jobId}/status" in FINISH
    assert "/api/finish/${payload.job_id}/cancel" in FINISH


def test_pixal3d_is_offered_as_a_backend_with_its_settings():
    assert '<option value="pixal3d">' in INDEX
    assert 'data-backend="pixal3d"' in INDEX
    ids = _element_ids(INDEX)
    for field in ("pixal3d-res", "pixal3d-steps", "pixal3d-seed", "pixal3d-fov"):
        assert field in ids


def test_the_generate_mode_reads_the_pixal3d_fields_that_exist():
    block = GENERATE[GENERATE.index("pixal3d: () => ({"):]
    block = block[: block.index("}),")]
    referenced = set(re.findall(r"g\('([^']+)'\)", block))
    assert referenced == {"pixal3d-res", "pixal3d-steps", "pixal3d-seed", "pixal3d-fov"}
    assert not referenced - _element_ids(INDEX)


def test_the_mlx_attention_choice_is_still_offered():
    """The three routes this viewer exposes: MLX attention, finishing, and Pixal3D."""
    assert 'id="generate-attention"' in INDEX
    assert 'value="mlx"' in INDEX and 'value="mlx-fp16"' in INDEX


def test_skipping_the_repaint_hides_only_the_paint_fields():
    assert "finish-paint-fields" in _element_ids(INDEX)
    assert "f('finish-paint-fields').hidden = !f('finish-repaint').checked;" in FINISH


def test_every_progress_track_has_a_styled_fill():
    """The bar inside a .progress-track must be reachable by a rule that paints it.

    `#generate-overall-bar` was the only selector, so the Finish panel's identically
    structured bar rendered as an empty groove for the whole run (found 2026-09-21).
    """
    css = (VIEWER / "styles" / "generate.css").read_text()
    assert ".progress-track > div" in css, "the fill is styled per-id, so a new track shows nothing"

    tracks = re.findall(r'<div class="progress-track"><div id="([^"]+)"></div></div>', INDEX)
    assert len(tracks) >= 2, "expected the Generate and Finish tracks to share this markup"


def test_the_finish_mode_can_list_and_resume_runs_on_disk():
    # A job registry lives in the server's memory; the run directories outlive it, and
    # this list is the only way back to a run whose browser tab was closed.
    assert "/api/finish/runs" in FINISH
    assert "/resume" in FINISH


SETUP = (VIEWER / "modes" / "setup.js").read_text()


def test_every_id_the_setup_mode_looks_up_exists_in_the_page():
    # The lookup helper is a single `s`, so the pattern has to reject any call that merely
    # *ends* in one: `.includes('huggingface')` matched a bare `s\('...'\)` and reported
    # "huggingface" as a missing element id.
    referenced = set(re.findall(r"(?<![A-Za-z0-9_.])s\('([^']+)'\)", SETUP))
    assert referenced, "the module should look up some elements"
    missing = sorted(referenced - _element_ids(INDEX))
    assert not missing, f"setup.js references ids absent from index.html: {missing}"


def test_the_setup_mode_is_registered_and_lands_first():
    assert "import { skipRequested } from './modes/setup.js';" in APP
    assert "setup: byId('setup-view')" in APP
    # First run lands on Setup & Status; ticking skip makes Generate the default. The page
    # is never forced back, because overriding that preference would ignore the user. The
    # choice itself lives in core/embed.js landingMode (tested in test_viewer_js_modules).
    assert "setMode(landingMode({ embedded, linked, skipSetup: skipRequested() }));" in APP


def test_generate_no_longer_carries_the_full_setup_card():
    # The checks moved to Setup & Status; Generate keeps one health line pointing at it.
    ids = _element_ids(INDEX)
    assert "health-dot" in ids and "health-text" in ids and "health-goto" in ids
    assert "setup-checks" not in ids, "the old Setup card is still in the page"
    assert "setHealth(" in GENERATE


def test_the_setup_page_states_cost_and_licence_before_downloading():
    # AGENTS.md requires naming the backend, the route, the size and asking first.
    assert "window.confirm(" in SETUP
    assert "will be downloaded from Hugging Face" in SETUP
    assert "Licence:" in SETUP
    assert "action = 'download'" in SETUP and "/cancel" in SETUP
    assert "startDownload(backend, button, 'rebuild')" in SETUP


def test_removing_weights_needs_the_backend_id_typed_out():
    """A yes/no dialog is not enough for this one.

    Three Remove buttons sit in one column, each deleting gigabytes that take minutes to
    hours to replace. The typed id proves both that it was meant and which one was meant.
    """
    assert "window.prompt(" in SETUP
    assert "Type  ${backend.id}  to confirm" in SETUP
    assert "!== backend.id.toLowerCase()" in SETUP
    # And it must never fall through to the request on a mistyped or cancelled prompt.
    assert "if (typed === null) return;" in SETUP


def test_absent_shape_models_are_disabled_rather_than_left_to_fail():
    """Only the default Hunyuan route is downloaded now, so 2.1 and 2.0-turbo are usually
    absent. Offering them anyway lets a run die minutes in, which is the failure the mlx
    attention options were already disabled to avoid."""
    assert "s.model_availability" in GENERATE
    assert "option.disabled = !present" in GENERATE
    assert "not downloaded" in GENERATE
    # And the selection must move off an option it just disabled.
    assert "select.selectedOptions[0]?.disabled" in GENERATE


def test_the_finish_tab_rereads_its_runs_on_arrival():
    """A run made while the tab was hidden (CLI, another tab) must show up on return."""
    arrival = FINISH.split("if (event.detail?.mode === 'finish') {", 1)[1].split("}", 1)[0]
    assert "loadRuns();" in arrival


def test_the_finish_page_reads_what_the_machine_can_do():
    assert "/api/finish/capabilities" in FINISH
    assert "f('finish-repaint').disabled = true;" in FINISH
    assert "!state.ready" in FINISH
    assert 'id="finish-machine-note"' in INDEX


def test_setup_page_reattaches_to_a_run_already_in_progress():
    # The catalog names the running setup; load() must watch it, or a refresh mid-setup
    # loses the progress bar for the rest of a 30-60 minute build.
    assert "state.catalog.running_setup" in SETUP
    assert "resume(state.catalog.running_setup)" in SETUP


def test_setup_confirmation_mentions_the_build_before_the_download():
    assert "This builds the code for this machine first, then:" in SETUP


def test_setup_title_says_setting_up_until_the_build_exists():
    # NVIDIA TRELLIS.2 compiles for half an hour before downloading; "Downloading" misled.
    assert "'Setting up'" in SETUP


def test_setup_page_has_machine_tabs():
    # One page for every machine read as a contradiction on NVIDIA ("Mac port here",
    # "use the NVIDIA one instead"). Tabs: this machine's live cards, others read-only.
    assert 'id="setup-tabs"' in (VIEWER / "index.html").read_text()
    assert "function renderTabs(" in SETUP and "readOnly" in SETUP


def test_a_build_without_a_percentage_shows_the_bar_working():
    assert "classList.toggle('busy'" in SETUP
    css = (VIEWER / "styles" / "generate.css").read_text()
    assert ".progress-track.busy" in css


def test_another_machines_tab_says_whose_it_is():
    assert "Showing what runs on" in SETUP


def test_overlapping_setup_renders_cannot_double_a_card():
    # Seen live: first load drew the page twice at once (startup + tab open). Each cleared
    # the list, awaited the Blender check, then appended, so Blender showed twice. Only
    # the newest render may write, and it writes after its await, never across one.
    assert "state.renderToken" in SETUP
    body = SETUP[SETUP.index("async function renderBackends"):]
    body = body[:body.index("\n}\n")]
    assert body.index("await blenderCard()") < body.index("host.innerHTML = ''")


def test_the_finished_line_shows_the_pixel_match_note():
    finish_js = (VIEWER / "modes" / "finish.js").read_text()
    assert "event.note" in finish_js


def test_remove_button_counts_only_what_it_would_free():
    assert "backend.bytes_removable" in SETUP


def test_generate_hands_a_model_straight_to_finish():
    # On a remote machine, finishing a fresh model meant downloading it to a laptop and
    # uploading it again. Generate's result has a button that hands both files over.
    index = (VIEWER / "index.html").read_text()
    generate = (VIEWER / "modes" / "generate.js").read_text()
    finish_js = (VIEWER / "modes" / "finish.js").read_text()
    assert 'id="generate-finish"' in index
    assert "'viewer:finish-this'" in generate and "mode: 'finish'" in generate
    listener = finish_js[finish_js.index("addEventListener('viewer:finish-this'"):]
    assert "state.asset" in listener and "state.image" in listener and "updateSubmit()" in listener


def test_a_route_needing_a_top_up_is_not_called_nothing_installed():
    # An upgrader whose TRELLIS worked yesterday saw "No backend installed yet" because one
    # new small file (the background remover) was missing. Partial routes are named.
    body = SETUP[SETUP.index("function summarise"):]
    body = body[:body.index("\n}\n")]
    assert "state === 'partial'" in body and "Resume download" in body
    assert "NVIDIA support is on the way" not in SETUP


def test_setup_has_a_hugging_face_sign_in_card():
    # `hf auth login` was the one terminal step left between the install command and a
    # working TRELLIS. The Setup page takes the token itself.
    assert "async function hfCard" in SETUP and "/api/hf/sign-in" in SETUP
    assert "type=\"password\"" in SETUP or "type = 'password'" in SETUP
    body = SETUP[SETUP.index("async function renderBackends"):]
    assert "await hfCard()" in body[:body.index("\n}\n")]


def test_blender_card_offers_an_install_on_linux():
    assert "/api/blender/install" in SETUP and "blender_installable" in SETUP
