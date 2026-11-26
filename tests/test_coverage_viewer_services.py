"""Local viewer settings and external request adapters."""

import io
import json
import signal
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import bug_report_api
import hf_api
import notices_api
import pytest
import serve
import update_api


@pytest.mark.parametrize("scenario", ["valid", "expired", "offline"])
def test_hugging_face_adapters_handle_access_and_never_return_token(monkeypatch, scenario):
    import huggingface_hub
    from huggingface_hub import utils

    token = "hf_test-token-with-no-access"
    current = Mock(return_value=token)
    identity = Mock(return_value={"name": "sample-user"})
    denied = utils.GatedRepoError("terms must be accepted", response=SimpleNamespace(headers={}, request=None))
    access = Mock(side_effect=[None, denied])
    save = Mock()
    if scenario == "expired":
        identity.side_effect = RuntimeError("expired token")
    if scenario == "offline":
        access.side_effect = OSError("Hugging Face offline")
    for name, function in (("get_token", current), ("whoami", identity), ("auth_check", access), ("login", save)):
        monkeypatch.setattr(huggingface_hub, name, function)
    result = hf_api.status()
    current.assert_called_once_with()
    identity.assert_called_once_with(token=token)
    if scenario == "expired":
        assert result["signed_in"] is False
        access.assert_not_called()
    else:
        assert result["signed_in"] is True
        assert result["user"] == "sample-user"
    assert [repo["access"] for repo in result["repos"]] == (
        ["yes", "no"] if scenario == "valid" else ["unknown", "unknown"]
    )
    assert token not in json.dumps(result)
    if scenario == "valid":
        access.side_effect = None
        signed = hf_api.sign_in(" " + token + " ")
        assert signed["user"] == "sample-user"
        assert token not in json.dumps(signed)
        save.assert_called_once_with(token=token, add_to_git_credential=False)
    else:
        save.assert_not_called()


@pytest.mark.parametrize(("module", "fetcher", "url", "body"), [
    (notices_api, notices_api.fetch_notices, notices_api.NOTICES_URL,
     {"notices": [{"id": "notice", "title": "New tools"}]}),
    (update_api, update_api.fetch_latest, update_api.LATEST_URL, {"tag_name": "v9.9.9"}),
])
def test_viewer_fetchers_read_expected_url_without_real_http(monkeypatch, module, fetcher, url, body):
    def respond(request, timeout):
        assert request.full_url == url
        assert request.get_header("User-agent")
        assert timeout == 5
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    assert fetcher() == body


def test_notice_feed_uses_release_even_with_bad_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.write_text("[]", encoding="utf-8")
    read = tmp_path / "read"
    blocked_read = read / "state.json"
    read.write_text("not a directory", encoding="utf-8")
    notices_api.mark_read(["release-9.0.0"], read_path=blocked_read)
    result = notices_api.notices("1.0.0", {"newer": True, "latest": "9.0.0"}, cache=cache,
                                 read_path=blocked_read, fetch=lambda: {"notices": {}}, now=12, env={})
    assert result["unread"] == 1
    assert result["notices"][0]["id"] == "release-9.0.0"
    assert read.read_text(encoding="utf-8") == "not a directory"


def test_long_report_keeps_first_message_words_if_details_cannot_fit():
    message = "First line explains. " + "A meaningful report. " * 1000
    links = bug_report_api.links(message, "detail" * 1000, home="")
    assert len(links["github"]) <= bug_report_api.GITHUB_URL_MAX
    captured = parse_qs(urlsplit(links["github"]).query)
    assert captured["body"][0].startswith("First line explains.")
    assert "[truncated]" in captured["body"][0]
    assert links["text"].startswith(message.strip())


def test_server_refuses_malformed_key_before_opening_listener(monkeypatch, capsys):
    monkeypatch.setattr(serve.sys, "argv", ["serve", "--host", "0.0.0.0"])
    monkeypatch.setenv(serve.studio_auth.TOKEN_ENV, "too-short")
    server = Mock()
    monkeypatch.setattr(serve, "ThreadingHTTPServer", server)
    assert serve.main() == 2
    assert "must be 16 to 128" in capsys.readouterr().err
    server.assert_not_called()


def test_server_reconciles_runs_and_removes_token_when_stopped(tmp_path, monkeypatch, capsys):
    key = "this-is-a-fixture-token"
    store = tmp_path / ".token"
    monkeypatch.setattr(serve.sys, "argv", ["serve", "--auto", "--port", "8877"])
    monkeypatch.setenv("RUNPOD_POD_ID", "example")
    monkeypatch.setenv("I3D_ALLOWED_HOSTS", "studio.example")
    monkeypatch.setenv(serve.studio_auth.TOKEN_ENV, key)
    monkeypatch.setattr(serve, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(serve.studio_auth, "save_token", partial(serve.studio_auth.save_token, path=store))
    monkeypatch.setattr(serve.studio_auth, "forget_token", partial(serve.studio_auth.forget_token, path=store))
    monkeypatch.setattr(serve, "_reconcile_orphaned_jobs", Mock(return_value=["stale job"]))
    terminate = Mock()
    monkeypatch.setattr(serve, "_terminate_active_job", terminate)
    monkeypatch.setattr(serve.processes, "kill_children_with_this_process", Mock())
    monkeypatch.setattr(serve.usage_api, "sharing", lambda: True)
    send_usage = Mock()
    monkeypatch.setattr(serve.usage_api, "send_soon", send_usage)
    installed = {}
    monkeypatch.setattr(serve.signal, "signal", lambda sig, callback: installed.update({sig: callback}))

    class Server:
        def __init__(self, address, serve_handler):
            assert address == ("0.0.0.0", 8877)
            assert serve_handler.func is handler

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert not store.exists()

        def serve_forever(self):
            assert store.read_text(encoding="utf-8") == key
            installed[signal.SIGINT](signal.SIGINT, None)

    def request_handler(*args, **kwargs):
        raise AssertionError("test does not accept requests")

    request_handler.studio_token = None
    handler = request_handler
    monkeypatch.setattr(serve, "Handler", handler)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", Server)
    assert serve.main() == 0
    assert handler.studio_token == key
    terminate.assert_called_once_with()
    serve._reconcile_orphaned_jobs.assert_called_once_with(tmp_path)
    send_usage.assert_called_once_with()
    output = capsys.readouterr().out
    assert f"https://example-8877.proxy.runpod.net/viewer/studio.html?token={key}" in output
    assert "reconciled 1 orphaned job" in output
    assert "stopped" in output


def test_usage_opt_out_during_send_preserves_new_choice(tmp_path, monkeypatch):
    import usage_api

    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"noticed": True, "id": "fixture-id", "finished": {"picture": 1}}), encoding="utf-8")
    seen = []

    def post(request, timeout):
        body = json.loads(request.data)
        assert request.full_url == usage_api.USAGE_URL
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert timeout == 5
        seen.append(body)
        assert usage_api.set_sharing(False, state=path, env={})["enabled"] is False
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(usage_api.urllib.request, "urlopen", post)
    assert not usage_api.send_daily(state=path, env={}, today="2026-01-02", clock=lambda: 4000.0)
    assert seen[0]["finished"] == {"picture": 1}
    assert seen[0]["id"] == "fixture-id"
    assert not usage_api.sharing(state=path, env={})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["opted_out"]
    assert "id" not in record and "sent_day" not in record


def test_usage_state_retry_does_not_change_choice_when_replace_never_succeeds(tmp_path, monkeypatch):
    import usage_api

    path = tmp_path / "usage.json"
    path.write_text('{"noticed":true,"opted_out":false}', encoding="utf-8")
    sleeps = []

    def blocked(source, destination):
        assert source.name == "usage.tmp" and destination == path
        raise PermissionError("temporarily busy")

    monkeypatch.setattr(type(path), "replace", blocked)
    monkeypatch.setattr(usage_api, "time", SimpleNamespace(sleep=sleeps.append))
    result = usage_api.set_sharing(False, state=path, env={})
    assert result["enabled"] is True
    assert sleeps == pytest.approx([0.05, 0.1, 0.15])
    assert (tmp_path / "usage.tmp").is_file()


def test_preview_falls_back_to_bundled_font_and_validates_asset_stage(monkeypatch, tmp_path):
    import preview_api
    from PIL import ImageFont

    load_font = ImageFont.truetype

    def no_system_fonts(font, *args, **kwargs):
        if isinstance(font, str):
            raise OSError("system font unavailable")
        return load_font(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", no_system_fonts)
    font = preview_api.label_font()
    assert font.getbbox("Front")[2] > 0
    with pytest.raises(ValueError, match="has no latest yet"):
        preview_api.preview(tmp_path, {"name": "Blank"})
