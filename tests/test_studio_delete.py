"""The Library's delete: what the row offers, what the confirmation says, and that both
name the same asset."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


KNIGHT = {"id": "a" * 12, "name": "a chunky knight", "kind": "character", "stage": "animated",
          "picture": None, "model": "a/k.glb", "finished": "finish/k/k_40k.glb",
          "rigged": "animate/k/k_rigged.glb", "clips": [{"name": "dance"}], "props": [],
          "files": 9, "bytes": 240 * 1000 * 1000}   # decimal, so "240 MB" is what a reader expects


def test_the_bin_gets_the_corner_to_itself_and_never_sits_over_text():
    """It is an absolutely positioned overlay, so nothing keeps it off the row's words unless
    the row leaves room. Two things must hold: the right edge of the name and of the status
    line is reserved, and the status line stays on one line."""
    css = (REPO / "viewer" / "studio" / "studio.css").read_text()
    for rule in re.findall(r"\.asset \.nm \{[^}]*\}|\.asset \.st[^{]*\{[^}]*\}", css):
        assert "padding-right" in rule or "white-space: nowrap" in rule, rule
    assert "white-space: nowrap" in css, "a long status line must shorten, not wrap"
    # and the row keeps the bin out of the way until it is wanted
    wipe = re.search(r"\.wipe \{[^}]*\}", css).group()
    assert "opacity: 0" in wipe and "pointer-events: none" in wipe, "the bin is armed only on hover"
    assert ".asset-row:hover .wipe" in css and ":focus-visible" in css, "hover alone must not be the only way"


def test_the_dialog_keeps_the_keyboard_while_it_is_open():
    """Tab must not wander out of the question into the page behind it: there are exactly two
    buttons, and they are where the tab order goes until it closes."""
    app = (REPO / "viewer" / "studio" / "app.js").read_text()
    ask = app[app.index("function askDelete("):app.index("async function deleteAsset(")]
    assert "event.key !== 'Tab'" in ask and "preventDefault()" in ask
    assert "event.key === 'Escape'" in ask, "Escape must back out without deleting"
    # and it opens on the safe answer
    assert "querySelector('[data-keep]').focus()" in ask, "the question must open on Keep it"


@needs_node
def test_a_size_is_read_the_way_a_person_would_say_it():
    out = _run("library.js", """
      console.log(JSON.stringify([0, 1, 999, 1024, 1536, 1000 * 1024, 240 * 1000 * 1000,
                                  2.5 * 1000 ** 3, 1.5 * 1000 ** 4, undefined].map(m.humanBytes)));""")
    assert out == ["0 B", "1 B", "999 B", "1 kB", "1.5 kB", "1 MB", "240 MB", "2.5 GB", "1.5 TB", "0 B"]


@needs_node
def test_the_confirmation_says_what_goes_and_where_it_lands():
    out = _run("library.js", f"""
      console.log(JSON.stringify(m.deletePrompt({json.dumps(KNIGHT)})));""")
    assert out["name"] == "a chunky knight"
    assert "240 MB" in out["size"] and "9 files" in out["size"]
    assert "a chunky knight" in out["body"] and "output/.trash" in out["body"]
    assert "cannot be undone" not in out["body"]   # it can: the trash is there to be used
    assert out["cancel"] and out["confirm"]


@needs_node
def test_one_file_is_not_said_as_files():
    out = _run("library.js", """
      const one = { id: 'b'.repeat(12), name: 'a lone fox', kind: 'picture', files: 1, bytes: 2048, props: [] };
      console.log(JSON.stringify({ subject: m.deleteSubject(one), body: m.deletePrompt(one).body,
                                   propSet: m.deleteSubject({ ...one, kind: 'prop set', props: [1, 2, 3] }) }));""")
    assert out["subject"] == "1 file"
    assert "1 file (2 kB)" in out["body"]
    assert "9 files" not in out["body"]
    assert out["propSet"] == "3 props"


@needs_node
def test_the_step_is_named_so_nobody_deletes_the_wrong_thing():
    """A rig with two moves deletes all of them, so the confirmation says what that is."""
    out = _run("library.js", f"""
      const knight = {json.dumps(KNIGHT)};
      const props = {{ ...knight, kind: 'prop set', props: [1, 2, 3, 4, 5, 6, 7, 8, 9] }};
      console.log(JSON.stringify({{ one: m.deletePrompt(knight).body, props: m.deletePrompt(props).body,
                                   files: m.deletePrompt(knight).size }}));""")
    assert "a chunky knight" in out["one"] and "output/.trash" in out["one"]
    assert "9 files · 240 MB" == out["files"]
    assert "9 props" in out["props"] and "9 files" not in out["props"]


def test_every_row_offers_a_delete_that_only_its_own_asset_can_reach():
    """The trash must be a sibling of the row's own button, never inside it: a button inside
    a button is not a button, and its click would also open the asset behind it."""
    app = (REPO / "viewer" / "studio" / "app.js").read_text()
    row = app[app.index("function renderLibrary()"):app.index("function engineOf(")]
    assert "data-wipe" in row, "the Library row has no delete button"
    # the row's own button must close before the bin opens: a button inside a button is not a button
    # the per-asset row, not the Empty trash control above it or the "Show more" row after it
    row = row[row.index('class="asset-row"'):row.index("+ (more ?")]
    buttons = [m.start() for m in re.finditer(r"<button\b", row)]
    closes = [m.start() for m in re.finditer(r"</button>", row)]
    assert len(buttons) == len(closes) == 2, "the row should hold exactly two buttons"
    assert buttons[0] < closes[0] < buttons[1] < closes[1], "the bin must be a sibling, not nested"
    # the click handler tells the two apart by dataset, and checks the trash first
    handler = app[app.index("$('assetList').addEventListener"):app.index("$('search')")]
    assert handler.index("dataset.wipe") < handler.index("closest('.asset')")
    assert "deleteAsset" in app and "askDelete" in app


@needs_node
def test_emptying_the_trash_says_how_much_space_it_would_free():
    out = _run("library.js", """
      const say = (t) => m.emptyTrashPrompt(t);
      console.log(JSON.stringify({
        nothing: say(null),
        two: say({ folds: 2, files: 41, bytes: 340_000_000 }),
        one: say({ folds: 1, files: 1, bytes: 2048 }),
        props: say({ folds: 1, files: 300, bytes: 2_500_000_000 }),
      }));""")
    assert out["nothing"]["show"] is False, "no trash, no button"
    assert "Free" in out["two"]["body"] and "340 MB" in out["two"]["body"]
    assert "2 deletes" in out["two"]["body"] and "41 files" in out["two"]["body"]
    assert out["two"]["show"] is True and out["two"]["confirm"] == "Empty it"
    assert "permanently" in out["two"]["body"].lower()
    assert "1 delete" in out["one"]["body"] and "1 file" in out["one"]["body"]
    assert "2.5 GB" in out["props"]["body"]


@needs_node
def test_a_trash_button_only_appears_when_there_is_something_in_it():
    out = _run("library.js", """
      console.log(JSON.stringify([
        m.trashLabel(null), m.trashLabel({ folds: 1, files: 3, bytes: 5_000_000 }),
      ]));""")
    assert out[0] == ""
    assert "Empty trash" in out[1] and "5 MB" in out[1]


def test_the_trash_button_sits_above_the_first_asset_and_asks_before_emptying():
    app = (REPO / "viewer" / "studio" / "app.js").read_text()
    row = app[app.index("function renderLibrary()"):app.index("function engineOf(")]
    assert "data-empty-trash" in row, "the Library has no Empty trash button"
    assert row.index("data-empty-trash") < row.index("makingRow()"), \
        "the button belongs above the rows, not among them"
    assert "askEmptyTrash" in app and "emptyTrashAsset" in app
    handler = app[app.index("async function emptyTrashAsset("):app.index("/** One line at the bottom")]
    assert "confirm: 'empty'" in handler, "the request must be confirmed, not merely sent"


def test_the_how_to_page_explains_the_trash():
    out = _run("pages.js", "console.log(JSON.stringify(m.GUIDE.map(([t, x]) => t + ' ' + x).join(' ')));")
    assert "Delete" in out and "Empty trash" in out and ".trash" in out