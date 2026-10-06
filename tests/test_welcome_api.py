"""The welcome card: brand, machine, and what changed since the version last seen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import welcome_api

import image_to_3dlab

REPO = Path(__file__).resolve().parents[1]

CHANGELOG = """# Changelog

Preamble that is not a release.

## [Unreleased]

### Added
- **Not shipped yet.** Should never be announced.

## [0.3.0] - 2026-09-30

### Added
- **Text to image to 3D on NVIDIA.** Long explanation that follows the bold lead
  and wraps onto a second line.
- `scripts/bootstrap_sf3d.py` installs SF3D. More words here.

### Fixed
- Generate Image read output in blocks. It now reads what is there.

## [0.2.0] - 2026-09-23

### Added
- **A Generate Image tab.** Type a prompt.

## [0.1.0] - 2026-09-01

### Added
- **First release.**
"""


def test_releases_are_parsed_newest_first_without_unreleased():
    releases = welcome_api.parse_changelog(CHANGELOG)
    assert [r["version"] for r in releases] == ["0.3.0", "0.2.0", "0.1.0"]
    assert releases[0]["date"] == "2026-09-30"


def test_each_item_is_reduced_to_its_headline():
    """The card has room for a line per change, not the paragraph behind it."""
    first = welcome_api.parse_changelog(CHANGELOG)[0]
    assert first["sections"]["Added"] == [
        "Text to image to 3D on NVIDIA.",
        "`scripts/bootstrap_sf3d.py` installs SF3D.",
    ]
    assert first["sections"]["Fixed"] == ["Generate Image read output in blocks."]


@pytest.mark.parametrize("since,expected", [
    (None, ["0.3.0"]),            # first visit: just the current release
    ("0.2.0", ["0.3.0"]),
    ("0.1.0", ["0.3.0", "0.2.0"]),
    ("0.3.0", []),                # up to date: nothing new
    ("not-a-version", ["0.3.0"]),  # junk from storage is treated as a first visit
])
def test_news_since_the_last_seen_version(since, expected):
    releases = welcome_api.parse_changelog(CHANGELOG)
    assert [r["version"] for r in welcome_api.news_since(releases, since)] == expected


def test_version_matches_the_newest_changelog_release():
    """The welcome card decides what is new from this number. It sat at 0.1.0 through the
    0.2.0 release, which would have announced nothing to anyone."""
    releases = welcome_api.parse_changelog((REPO / "CHANGELOG.md").read_text())
    assert image_to_3dlab.__version__ == releases[0]["version"]


def test_brand_lives_in_one_file():
    brand = json.loads((REPO / "viewer" / "brand.json").read_text())
    assert brand["name"] == "AssetFurnace"
    assert welcome_api.brand() == brand


def test_payload_names_this_machine_and_its_routes(monkeypatch):
    import backend_catalog
    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.NVIDIA)
    monkeypatch.setattr(welcome_api, "read_changelog", lambda: CHANGELOG)
    data = welcome_api.payload(since="0.2.0")
    assert data["version"] == image_to_3dlab.__version__
    assert data["brand"]["name"] == "AssetFurnace"
    assert data["host"]["id"] == "nvidia"
    runs_here = {r["id"] for r in data["routes"]}
    assert {"pixal3d", "qwen-image"} <= runs_here
    assert "sf3d" not in runs_here  # Mac only: its pins broke the shared environment
    assert "hunyuan_xiong" not in runs_here  # Mac-only for now
    # TRELLIS.2 runs on NVIDIA under Linux, not yet under Windows.
    linux_ok = backend_catalog.BY_ID["trellis"].excluded_os(backend_catalog.NVIDIA) is None
    assert ("trellis" in runs_here) is linux_ok
    assert [r["version"] for r in data["news"]] == ["0.3.0"]


def test_repeated_section_headings_are_merged_in_order():
    """0.2.0's entry repeats `### Added` a dozen times; keeping only the last one dropped
    the release's headline feature."""
    text = """## [0.2.0] - 2026-09-23

### Added
- **First.**

### Fixed
- **A fix.**

### Added
- **Second.**
"""
    sections = welcome_api.parse_changelog(text)[0]["sections"]
    assert sections["Added"] == ["First.", "Second."]
    assert sections["Fixed"] == ["A fix."]


def test_a_bold_lead_that_is_only_a_name_keeps_its_sentence():
    item = ("**`scripts/bootstrap_qwen_image.py`** — installs that route: a prebuilt "
            "binary and 13.4 GB of weights. It names the backend.")
    assert welcome_api.headline(item) == (
        "`scripts/bootstrap_qwen_image.py` — installs that route: a prebuilt binary and "
        "13.4 GB of weights.")


def test_headlines_are_capped_at_one_line():
    assert len(welcome_api.headline("word " * 100)) <= welcome_api.HEADLINE_MAX


def test_the_viewer_serves_it():
    """Through the real handler, not the function: the route and the query string too."""
    import functools
    import http.server
    import threading
    import urllib.request

    import generate_api

    handler = functools.partial(generate_api.Handler, directory=str(REPO))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/welcome?since=0.0.1"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
    finally:
        server.shutdown()
    assert data["brand"]["name"] == "AssetFurnace"
    assert data["news"] and data["news"][0]["version"] == image_to_3dlab.__version__
