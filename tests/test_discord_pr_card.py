import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import discord_pr_card as card

URL = "https://github.com/Bingeljell/image-to-3dlab/pull/114"


def pr(**over):
    base = {"html_url": URL, "number": 114, "title": "Studio QoL", "draft": False, "merged": False,
            "body": "## What\n- Fixes #110, thanks @dlm21.\n", "updated_at": "2026-10-12T14:02:38Z",
            "user": {"login": "Bingeljell", "html_url": "https://github.com/Bingeljell", "avatar_url": "https://a/1"}}
    return base | over


def test_an_opened_pr_becomes_an_ember_card_with_its_author():
    embed = card.build_payload(pr(), "opened")["embeds"][0]
    assert embed["title"] == "#114 Studio QoL" and embed["url"] == URL
    assert embed["color"] == card.EMBER
    assert embed["author"]["name"] == "Bingeljell" and embed["author"]["icon_url"] == "https://a/1"
    assert embed["description"].startswith("**What**")
    assert "[#110](https://github.com/Bingeljell/image-to-3dlab/issues/110)" in embed["description"]
    assert embed["footer"]["text"] == "Bingeljell/image-to-3dlab · opened"


@pytest.mark.parametrize(("action", "over", "said", "colour"), [
    ("closed", {"merged": True, "merged_at": "t"}, "merged", card.MERGED),
    ("closed", {}, "closed", card.CLOSED),
    ("ready_for_review", {}, "ready for review", card.EMBER),
])
def test_merges_and_closes_say_so_in_their_own_colour(action, over, said, colour):
    embed = card.build_payload(pr(**over), action)["embeds"][0]
    assert embed["footer"]["text"].endswith(f"· {said}") and embed["color"] == colour


@pytest.mark.parametrize(("action", "over"), [("opened", {"draft": True}), ("edited", {}), ("labeled", {})])
def test_drafts_and_small_events_stay_quiet(action, over):
    assert card.build_payload(pr(**over), action) is None


def test_a_long_description_is_cut_with_a_link_to_the_pr():
    body = "\n\n".join(f"Paragraph {i}: " + "words " * 60 for i in range(20))
    text = card.build_payload(pr(body=body))["embeds"][0]["description"]
    assert len(text) <= card.SUMMARY_LIMIT and text.endswith(f"({URL})")


def test_an_empty_description_still_links_the_pr():
    assert card.build_payload(pr(body=None))["embeds"][0]["description"] == f"[Open the pull request]({URL})"


def test_the_event_file_github_hands_the_workflow_is_read_as_is(tmp_path, capsys):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"action": "closed", "pull_request": pr(merged=True)}))
    assert card.main([str(event), "--dry-run"]) == 0
    assert "· merged" in capsys.readouterr().out


def test_a_quiet_event_sends_nothing_even_without_a_webhook(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_PR_WEBHOOK", raising=False)
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"action": "edited", "pull_request": pr()}))
    assert card.main([str(event)]) == 0


def test_the_workflow_never_runs_the_prs_own_code():
    workflow = (REPO / ".github" / "workflows" / "discord-pr.yml").read_text()
    assert "pull_request_target" in workflow
    assert "ref:" not in workflow and "head.sha" not in workflow   # checks out main, not the PR
    assert "scripts/discord_pr_card.py" in workflow
