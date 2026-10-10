import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import discord_release_card as card  # noqa: E402

URL = "https://github.com/Bingeljell/image-to-3dlab/releases/tag/v0.4.3"
BODY = """# AssetFurnace 0.4.3

Thanks to @MylesLandais and @dlm21.

## New
- **Keep everything** in Create.
- A docs page: [docs/pixel-match.md](https://github.com/x/y/blob/main/docs/pixel-match.md) (#101).

## Update
```bash
curl -fsSL https://assetfurnace.com/install.sh | bash
# not a heading
```
"""


def release(**over):
    base = {"html_url": URL, "tag_name": "v0.4.3", "name": "AssetFurnace 0.4.3",
            "body": BODY, "published_at": "2026-10-10T12:00:00Z", "prerelease": False}
    return base | over


def test_payload_shape():
    payload = card.build_payload(release())
    embed = payload["embeds"][0]
    assert embed["title"] == "AssetFurnace 0.4.3"
    assert embed["url"] == URL
    assert embed["color"] == card.EMBER
    assert embed["image"]["url"].endswith("/Bingeljell/image-to-3dlab/v0.4.3/docs/images/social-preview.jpg")
    assert embed["timestamp"] == "2026-10-10T12:00:00Z"
    assert payload["allowed_mentions"] == {"parse": []}
    json.dumps(payload)


def test_notes_drop_title_and_bold_headings():
    text = card.format_notes(BODY, "Bingeljell/image-to-3dlab")
    assert not text.startswith("# ")
    assert "**New**" in text and "## New" not in text


def test_code_block_untouched():
    text = card.format_notes(BODY, "o/r")
    assert "# not a heading" in text
    assert "```bash" in text


def test_linkify_issues_and_users_but_not_existing_links():
    text = card.format_notes(BODY, "o/r")
    assert "[#101](https://github.com/o/r/issues/101)" in text
    assert "[@dlm21](https://github.com/dlm21)" in text
    assert "[docs/pixel-match.md](https://github.com/x/y/blob/main/docs/pixel-match.md)" in text


def test_fit_cuts_long_notes_at_paragraph_with_link():
    long = "\n\n".join(f"para {i} " + "x" * 200 for i in range(40))
    out = card.fit(long, URL)
    assert len(out) <= card.DESCRIPTION_LIMIT
    assert out.endswith(f"[Read the full notes]({URL})")


def test_fit_never_leaves_an_open_code_block():
    long = "intro\n\n```bash\n" + ("line\n\n" * 1500) + "```"
    out = card.fit(long, URL, limit=500)
    assert out.count("```") % 2 == 0


def test_prerelease_and_empty_body():
    embed = card.build_payload(release(prerelease=True, body=None, name=""))["embeds"][0]
    assert embed["title"] == "v0.4.3 (pre-release)"
    assert URL in embed["description"]


def test_cli_needs_webhook(tmp_path, monkeypatch):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(release()))
    monkeypatch.delenv("DISCORD_RELEASE_WEBHOOK", raising=False)
    assert card.main([str(path)]) == 1
    assert card.main([str(path), "--dry-run"]) == 0


def test_github_suffix_stripped(tmp_path, monkeypatch):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(release()))
    sent = {}
    monkeypatch.setattr(card, "send", lambda payload, hook: sent.update(hook=hook))
    monkeypatch.setenv("DISCORD_RELEASE_WEBHOOK", "https://discord.com/api/webhooks/1/abc/github")
    assert card.main([str(path)]) == 0
    assert sent["hook"] == "https://discord.com/api/webhooks/1/abc"
