"""Text files are read and written as UTF-8, whatever the machine's locale.

Without `encoding=`, Python uses the locale's: cp1252 on a Windows machine, which cannot
decode an em dash or a middle dot written as UTF-8. The studio then could not read back
JSON with such characters, and the test suite failed on Windows unless PYTHONUTF8=1 was
set. This scan keeps the studio's code and these tests from growing new ones.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKED = sorted((REPO / "viewer").glob("*.py")) + [
    REPO / "tests" / "test_scripts_registry.py", REPO / "tests" / "test_studio_modules.py"]
# `.open(...)` on these is not a file opened as text (an image, a browser tab)
NOT_FILES = {"Image", "webbrowser"}
TEMP_FILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}


def _text_mode(mode) -> bool:
    """Whether a mode argument opens a file as text (None: the default, "r")."""
    if mode is None:
        return True
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" not in mode.value


def _mode(call: ast.Call, position: int):
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == "mode"), None)


def unencoded(source: str) -> list[int]:
    """Line numbers of text reads and writes that do not say `encoding=`."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        if _reads_or_writes_text(node):
            lines.append(node.lineno)
    return sorted(lines)


def _reads_or_writes_text(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):  # open(path, mode)
        return func.id == "open" and _text_mode(_mode(call, 1))
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in ("read_text", "write_text"):
        return True
    if func.attr in TEMP_FILES:  # binary unless a text mode is given
        mode = _mode(call, 0)
        return mode is not None and _text_mode(mode)
    on_a_file = not (isinstance(func.value, ast.Name) and func.value.id in NOT_FILES)
    return func.attr == "open" and on_a_file and _text_mode(_mode(call, 0))  # Path.open(mode)


SAMPLE = """p.read_text()
p.read_text(encoding="utf-8")
open(f)
open(f, "rb")
p.open("a")
p.open("wb")
p.write_text(s, encoding="utf-8")
Image.open(src)
p.open()
webbrowser.open(url)
tempfile.NamedTemporaryFile("w", suffix=".py")
tempfile.NamedTemporaryFile(suffix=".png")
tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8")
tempfile.TemporaryFile("wb")
"""


def test_the_scan_finds_what_it_should():
    assert unencoded(SAMPLE) == [1, 3, 5, 9, 11]


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: p.name)
def test_text_files_are_read_and_written_as_utf8(path):
    assert unencoded(path.read_text(encoding="utf-8")) == [], f"{path.name}: say encoding='utf-8'"


# ---- pipes between the studio and its jobs ------------------------------------------------

# calls that start a child and may read its output as text
PIPED = {"Popen", "runner", "run", "check_output"}


def _keywords(call: ast.Call) -> dict:
    return {k.arg: k.value for k in call.keywords if k.arg}


def test_every_text_pipe_the_studio_reads_names_its_encoding():
    """`text=True` alone decodes with the locale: cp1252 on Windows, where a Chinese prompt
    or a name outside it breaks the reading loop or the child that prints it. A job's pipe,
    a `subprocess.run`, or any child given the studio's UTF-8 environment, says how to read
    it."""
    loose = []
    for path in sorted((REPO / "viewer").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            given = _keywords(node)
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name in PIPED or "env" in given:
                text = given.get("text")
                if isinstance(text, ast.Constant) and text.value is True and "encoding" not in given:
                    loose.append(f"{path.name}:{node.lineno}")
    assert loose == []


def test_a_child_prints_any_character_to_the_studio():
    import subprocess
    import sys

    from image_to_3dlab import processes

    child = subprocess.run([sys.executable, "-c", "print('\u2192 \u4e2d\u6587 \u00b7 caf\u00e9')"],
                           env=processes.child_env(), capture_output=True, check=True, **processes.TEXT_PIPE)
    assert child.stdout.strip() == "\u2192 \u4e2d\u6587 \u00b7 caf\u00e9"
    assert processes.child_env(HF_HUB_OFFLINE="1")["HF_HUB_OFFLINE"] == "1"
