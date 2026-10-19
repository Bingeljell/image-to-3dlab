"""The server-side chain runner and the Activity history, driven by a fake job server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viewer import activity_api as aa


class FakeJobs:
    """Answers the job endpoints the way the real ones do; each job is done on its first status poll."""

    def __init__(self, output: Path, fail: str | None = None, refuse: str | None = None, four_legged: bool = False):
        self.output, self.fail, self.refuse, self.four_legged = output, fail, refuse, four_legged
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.jobs: dict[str, str] = {}

    def __call__(self, method, path, body, content_type):
        self.calls.append((method, path, body))
        if method == "POST" and path.endswith("/cancel"):
            return 200, {"ok": True}
        if method == "POST":
            kind = {"/api/image": "picture", "/api/image/back": "back", "/api/generate": "model", "/api/finish": "finished",
                    "/api/animate/rig": "rigged", "/api/animate/play": "animated", "/api/props": "split"}[path]
            if kind == self.refuse:
                return 409, {"error": "another job is running; wait for it to finish"}
            job_id = f"job-{kind}"
            self.jobs[job_id] = kind
            api = path.split("/")[2]
            return 202, {"job_id": job_id, "status_url": f"/api/{api}/{job_id}/status"}
        kind = self.jobs[path.split("/")[3]]
        if kind == self.fail:
            return 200, {"status": "error", "error": "MemoryError: out of memory", "log_tail": "ggml_noise\nKilled"}
        made = {"picture": "images/p/pic.png", "back": "images/p/pic__back.png", "model": "pixal3d/m/model.glb",
                "finished": "finish/f/finished.glb", "rigged": "animate/r/rigged.glb", "animated": "animate/r/rigged_walk.glb"}
        for rel in made.values():
            (self.output / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.output / rel).write_bytes(b"x")
        event = {"phase": "done", "message": f"{kind} done", "overall_pct": 100}
        if kind == "rigged" and self.four_legged:
            event.update(humanoid=False, message="Rigged. Preset moves are for humanoids only for now.")
        if kind in ("finished", "rigged", "animated"):
            event["path"] = made[kind]
        return 200, {"status": "done", "last_event": event, "log_tail": f"{kind} ok",
                     "picture": made[kind] if kind in ("picture", "back") else None,
                     "model": made["model"] if kind == "model" else None}


def _runner(tmp_path, **fake):
    jobs = FakeJobs(tmp_path, **fake)
    return aa.ChainRunner(tmp_path, jobs, poll=0, sleep=lambda s: None), jobs


def test_a_whole_character_runs_and_each_step_gets_the_last_ones_file(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="sensei", prompt="a sensei", engine="pixal3d", first_move="walk",
                     steps=["picture", "model", "finished", "rigged", "animated"])
    runner.submit(chain, background=False)
    assert chain.status == "done" and chain.done == chain.steps
    posts = [(p, b) for m, p, b in jobs.calls if m == "POST"]
    assert [p for p, _ in posts] == ["/api/image", "/api/generate", "/api/finish", "/api/animate/rig", "/api/animate/play"]
    assert b'name="model"\r\n\r\nfinish/f/finished.glb' in posts[3][1]          # rig takes the finished model
    assert json.loads(posts[4][1]) == {"model": "animate/r/rigged.glb", "preset": "walk"}
    history = aa.read_history(tmp_path)
    assert [h["step"] for h in history] == ["animated", "rigged", "finished", "model", "picture"]  # newest first
    assert all(h["result"] == "done" for h in history)
    assert {h["prompt"] for h in history} == {"a sensei"}   # Activity shows what was asked for
    assert chain.made["animated"] == "animate/r/rigged_walk.glb"   # the move's own file, for agents
    assert history[0]["made"] == "animate/r/rigged_walk.glb"


def test_a_failure_stops_the_chain_in_plain_words_and_is_kept_in_history(tmp_path):
    runner, _ = _runner(tmp_path, fail="finished")
    chain = aa.Chain(title="knight", steps=["picture", "model", "finished", "rigged"], prompt="a knight")
    runner.submit(chain, background=False)
    assert chain.status == "error" and chain.done == ["picture", "model"]
    assert "ran out of memory" in chain.error
    assert "ggml" not in chain.log  # engine chatter dropped
    last = aa.read_history(tmp_path)[0]
    assert last["step"] == "finished" and last["result"] == "error" and "Killed" in last["log"]


def test_a_refused_start_is_reported_not_raised(tmp_path):
    runner, _ = _runner(tmp_path, refuse="model")
    chain = aa.Chain(title="x", steps=["picture", "model"], prompt="x")
    runner.submit(chain, background=False)
    assert chain.status == "error" and "Another job is running" in chain.error


def test_a_rig_from_an_uploaded_model_sends_the_file(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="mine", steps=["rigged", "animated"], first_move="idle", upload=("mine.glb", b"GLB!"))
    runner.submit(chain, background=False)
    rig_body = next(b for m, p, b in jobs.calls if p == "/api/animate/rig")
    assert b'filename="mine.glb"' in rig_body and b"GLB!" in rig_body
    assert chain.status == "done"


def test_only_one_chain_at_a_time_and_unknown_steps_are_refused(tmp_path):
    runner, _ = _runner(tmp_path)
    runner.chains["busy"] = aa.Chain(title="busy", steps=["picture"], id="busy")
    with pytest.raises(RuntimeError):
        runner.submit(aa.Chain(title="b", steps=["picture"]), background=False)
    runner.chains.clear()
    with pytest.raises(ValueError):
        runner.submit(aa.Chain(title="c", steps=["teleport"]), background=False)


def test_cancel_asks_the_running_job_to_stop(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="x", steps=["picture"], id="c1")
    runner.chains["c1"] = chain
    chain.job = ("image", "job-picture")
    assert runner.cancel("c1") and chain.cancel_requested
    assert ("POST", "/api/image/job-picture/cancel", None) in jobs.calls


def test_files_outside_output_are_never_read(tmp_path):
    runner, _ = _runner(tmp_path)
    with pytest.raises(ValueError):
        runner._file("../secrets.txt")


def test_history_skips_damaged_lines_and_survives_a_missing_file(tmp_path):
    assert aa.read_history(tmp_path) == []
    (tmp_path / aa.HISTORY_NAME).write_text('{"step": "a"}\nnot json\n{"step": "b"}\n')
    assert [h["step"] for h in aa.read_history(tmp_path)] == ["b", "a"]


def test_a_quiet_step_is_flagged_as_stalled():
    chain = aa.Chain(title="x", steps=["picture"])
    chain.last_change = 0
    assert chain.to_dict(now=aa.STALL_SECONDS + 1)["stalled"]
    assert not chain.to_dict(now=10)["stalled"]


def test_a_plan_sent_as_a_form_is_read_back_by_the_real_parser():
    from viewer.generate_api import parse_multipart
    body, ctype = aa.multipart(
        {"title": "sensei", "steps": json.dumps(["rigged", "animated"]), "first_move": "walk",
         "made": json.dumps({"finished": "finish/f/x.glb", "evil": "/etc/passwd"})},
        {"file": ("mine.glb", b"\x00GLB")})
    chain = aa.chain_from_form(parse_multipart(ctype, body))
    assert chain.title == "sensei" and chain.steps == ["rigged", "animated"] and chain.first_move == "walk"
    assert chain.made == {"finished": "finish/f/x.glb"}           # only the known kinds are taken
    assert chain.upload == ("mine.glb", b"\x00GLB")


def test_a_rig_that_does_not_fit_the_moves_ends_the_run_calmly_before_animate(tmp_path):
    runner, jobs = _runner(tmp_path, four_legged=True)
    chain = aa.Chain(title="griffin", steps=["picture", "model", "finished", "rigged", "animated"], prompt="g", first_move="walk")
    runner.submit(chain, background=False)
    assert chain.status == "done" and chain.done[-1] == "rigged"
    assert "humanoids only" in chain.to_dict()["note"]
    assert not any(p == "/api/animate/play" for _, p, _ in jobs.calls)


def test_the_picture_step_sends_the_chosen_size_steps_and_seed(tmp_path):
    from viewer.generate_api import parse_multipart
    body, ctype = aa.multipart({"title": "x", "steps": json.dumps(["picture"]), "prompt": "a crate",
                                "picture_settings": json.dumps({"width": 512, "height": 512, "steps": 8, "seed": 99, "evil": 1})}, {})
    chain = aa.chain_from_form(parse_multipart(ctype, body))
    assert chain.picture_settings == {"width": 512, "height": 512, "steps": 8, "seed": 99}
    runner, jobs = _runner(tmp_path)
    runner.submit(chain, background=False)
    sent = json.loads(next(b for m, p, b in jobs.calls if p == "/api/image"))
    assert sent == {"prompt": "a crate", "settings": {"width": 512, "height": 512, "steps": 8, "seed": 99}}


def test_the_model_step_sends_the_fresh_seed_create_chose(tmp_path):
    from viewer.generate_api import parse_multipart
    body, ctype = aa.multipart({"title": "x", "steps": json.dumps(["model"]), "engine": "pixal3d", "model_seed": "12345"},
                               {"file": ("boy.png", b"png")})
    chain = aa.chain_from_form(parse_multipart(ctype, body))
    assert chain.model_seed == 12345
    runner, jobs = _runner(tmp_path)
    runner.submit(chain, background=False)
    sent = next(b for m, p, b in jobs.calls if p == "/api/generate")
    assert b'{"backend": "pixal3d", "seed": 12345}' in sent


def test_a_missing_or_bad_model_seed_leaves_the_engine_default():
    from viewer.generate_api import parse_multipart
    for seed in (None, "", "lots"):
        fields = {"title": "x", "steps": json.dumps(["model"])}
        if seed is not None:
            fields["model_seed"] = seed
        body, ctype = aa.multipart(fields, {})
        assert aa.chain_from_form(parse_multipart(ctype, body)).model_seed is None


def test_keep_everything_reaches_the_3d_step_as_debug(tmp_path):
    """Create's "Keep everything" box: the 3D job keeps its textures, meshes and caches.
    /api/generate already honours `debug`; the chain just has to send it."""
    from viewer.generate_api import parse_multipart
    body, ctype = aa.multipart({"title": "x", "steps": json.dumps(["model", "finished"]), "engine": "pixal3d",
                                "keep_everything": "1"}, {"file": ("boy.png", b"png")})
    chain = aa.chain_from_form(parse_multipart(ctype, body))
    assert chain.keep_everything is True
    runner, jobs = _runner(tmp_path)
    runner.submit(chain, background=False)
    sent = {p: b for m, p, b in jobs.calls}
    assert b'"debug": true' in sent["/api/generate"]
    assert b"debug" not in sent["/api/finish"]       # only the 3D step has anything to keep


def test_keep_everything_is_off_unless_asked(tmp_path):
    from viewer.generate_api import parse_multipart
    for value in (None, "", "0", "no"):
        fields = {"title": "x", "steps": json.dumps(["model"])}
        if value is not None:
            fields["keep_everything"] = value
        body, ctype = aa.multipart(fields, {"file": ("boy.png", b"png")})
        chain = aa.chain_from_form(parse_multipart(ctype, body))
        assert chain.keep_everything is False
    runner, jobs = _runner(tmp_path)
    runner.submit(chain, background=False)
    assert b"debug" not in next(b for m, p, b in jobs.calls if p == "/api/generate")



def test_a_front_and_back_character_draws_the_back_then_builds_from_both(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="viking", prompt="a viking", engine="pixal3d", first_move="walk",
                     steps=["picture", "back", "model", "finished", "rigged", "animated"])
    runner.submit(chain, background=False)
    assert chain.status == "done" and chain.made["back"] == "images/p/pic__back.png"
    posts = [(p, b) for m, p, b in jobs.calls if m == "POST"]
    assert [p for p, _ in posts][:3] == ["/api/image", "/api/image/back", "/api/generate"]
    assert json.loads(posts[1][1])["picture"] == "images/p/pic.png"
    assert b'name="back"; filename="pic__back.png"' in posts[2][1]


def test_an_uploaded_picture_is_kept_in_output_so_its_back_can_be_drawn(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="mine", engine="pixal3d", steps=["back", "model"],
                     upload=("my orc.png", b"\x89PNG-bytes"))
    runner.submit(chain, background=False)
    assert chain.status == "done"
    sent = json.loads([b for m, p, b in jobs.calls if p == "/api/image/back"][0])["picture"]
    assert sent.startswith("images/uploads/") and (tmp_path / sent).read_bytes() == b"\x89PNG-bytes"


def test_the_back_only_goes_to_pixal3d(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="x", prompt="x", engine="trellis", steps=["picture", "back", "model"])
    runner.submit(chain, background=False)
    model_post = [b for m, p, b in jobs.calls if p == "/api/generate"][0]
    assert b'name="back"' not in model_post


def test_no_named_engine_means_no_back_since_the_default_engine_is_not_pixal3d(tmp_path):
    runner, jobs = _runner(tmp_path)
    chain = aa.Chain(title="x", prompt="x", steps=["picture", "back", "model"])
    runner.submit(chain, background=False)
    assert b'name="back"' not in [b for m, p, b in jobs.calls if p == "/api/generate"][0]


def test_each_finished_step_reports_how_long_it_took(tmp_path):
    clock = iter(range(0, 1000, 5))
    jobs = FakeJobs(tmp_path)
    runner = aa.ChainRunner(tmp_path, jobs, poll=0, sleep=lambda s: None, now=lambda: float(next(clock)))
    chain = aa.Chain(title="x", prompt="x", engine="pixal3d", steps=["picture", "back", "model"])
    runner.submit(chain, background=False)
    seconds = chain.to_dict()["step_seconds"]
    assert set(seconds) == {"picture", "back", "model"} and all(s > 0 for s in seconds.values())
    assert [h["made"] for h in aa.read_history(tmp_path) if h["step"] == "back"] == ["images/p/pic__back.png"]


def test_a_plan_carries_its_finish_choices_to_the_finish_job(tmp_path):
    runner, jobs = _runner(tmp_path)
    form = {"title": {"value": "k"}, "steps": {"value": json.dumps(["finished"])},
            "made": {"value": json.dumps({"model": "pixal3d/m/model.glb", "picture": "images/p/pic.png"})},
            "finish_settings": {"value": json.dumps({"keep_mesh": True, "skip_photo": False, "voxel": 9})}}
    for rel in ("pixal3d/m/model.glb", "images/p/pic.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    chain = aa.chain_from_form(form)
    assert chain.finish_settings == {"keep_mesh": True, "skip_photo": False}   # unknown keys never reach Finish
    runner.submit(chain, background=False)
    body = next(b for m, p, b in jobs.calls if p == "/api/finish")
    assert b'{"keep_mesh": true, "skip_photo": false}' in body


def test_a_plan_without_finish_choices_gets_finishs_defaults(tmp_path):
    chain = aa.chain_from_form({"title": {"value": "k"}, "steps": {"value": json.dumps(["finished"])}})
    assert chain.finish_settings == {}


def test_a_plan_carries_edge_fade_to_the_finish_job():
    form = {"title": {"value": "k"}, "steps": {"value": json.dumps(["finished"])},
            "finish_settings": {"value": json.dumps({"keep_mesh": True, "skip_edge_fade": True, "skip_nudge": True})}}
    assert aa.chain_from_form(form).finish_settings == {"keep_mesh": True, "skip_edge_fade": True, "skip_nudge": True}


def test_a_plan_carries_move_speed_and_arm_spread_to_the_animate_job(tmp_path):
    runner, jobs = _runner(tmp_path)
    form = {"title": {"value": "k"}, "steps": {"value": json.dumps(["animated"])}, "first_move": {"value": "wave"},
            "made": {"value": json.dumps({"rigged": "animate/k/k_rigged.glb"})},
            "move_settings": {"value": json.dumps({"speed": 1.5, "arm_spread": 10, "colour": "red"})}}
    (tmp_path / "animate/k").mkdir(parents=True)
    (tmp_path / "animate/k/k_rigged.glb").write_bytes(b"x")
    chain = aa.chain_from_form(form)
    assert chain.move_settings == {"speed": 1.5, "arm_spread": 10}     # unknown keys never reach Animate
    runner.submit(chain, background=False)
    body = next(b for m, p, b in jobs.calls if p == "/api/animate/play")
    assert json.loads(body) == {"model": "animate/k/k_rigged.glb", "preset": "wave", "speed": 1.5, "arm_spread": 10}


def test_a_plan_without_move_settings_lets_the_move_use_its_own_pace():
    chain = aa.chain_from_form({"title": {"value": "k"}, "steps": {"value": json.dumps(["animated"])}})
    assert chain.move_settings == {}


def test_install_help_for_blender_is_passed_on_as_it_is():
    """2026-10-19, the first site report: "Blender stopped with an error, see the log" for a Mac
    with no Blender, and no log, because nothing had run."""
    message = "Finish needs Blender 4.2 or newer, and none was found. Install Blender from blender.org."
    assert aa.plain_error(message) == message
    assert aa.plain_error("Blender: Error: something broke").startswith("Blender stopped with an error")


def test_a_plan_with_finish_is_refused_up_front_when_blender_is_missing(tmp_path, monkeypatch):
    """Checked before the picture and the 3D model, not ten minutes in at Finish."""
    import http.server
    import threading
    import urllib.error
    import urllib.request

    from viewer import generate_api as api

    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(api, "find_blender", lambda: None)
    started = []
    monkeypatch.setattr(api.ChainRunner, "submit", lambda self, chain, **k: started.append(chain))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(steps):
        body, ctype = aa.multipart({"title": "knight", "steps": json.dumps(steps)}, {})
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/chains", data=body,
                                         headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())
    try:
        status, payload = post(["picture", "model", "finished"])
        assert status == 409 and "Install Blender" in payload["error"] and payload["needs_blender"]
        assert "stop after the 3D model" in payload["error"] and started == []
        assert post(["picture", "model"])[0] == 202 and len(started) == 1   # no Blender needed
    finally:
        server.shutdown()
        server.server_close()
