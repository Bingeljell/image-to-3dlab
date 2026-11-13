"""Shared pytest setup: make the repo's own modules importable the ordinary way.

`viewer/` and `scripts/` are not packages and are not on Python's search path, so test
files had to hand-load the modules they exercise with `importlib`. That works, but it
quietly breaks Python's one-module-one-object rule: **every hand-load creates a new copy**.
When production code imports the same module normally, the test and the code under test
end up holding two different objects with the same name, and a monkeypatch applied to one
is invisible to the other.

That is not hypothetical. `tests/test_download_api.py` patched `backend_catalog` while
`download_api` held a different copy, so the "an unsupported machine is refused before
anything downloads" guard passed without refusing anything. It failed only in the file
order the alphabetical suite never takes, so it sat green for weeks.

Putting the two directories on the path here lets a test write `import download_api` and
get the same object everyone else has. Neither directory shadows a standard-library name
(checked: 7 and 116 modules, no clashes), so this is additive.

Existing files that hand-load are not wrong to do so where nothing else imports the same
module -- a private copy of a module no one shares cannot diverge from anyone. Migrate
them when convenient; migrate immediately if production code imports the same name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for directory in ("viewer", "scripts"):
    path = str(REPO / directory)
    if path not in sys.path:
        sys.path.insert(0, path)

# The suite never counts or sends usage from a real install (tests of it pass their own env).
os.environ["I3D_NO_USAGE"] = "1"


import pytest

_SYMLINKS: list[bool] = []


def symlinks_allowed() -> bool:
    """Whether this machine lets the user running the tests make a symlink.

    Windows refuses without admin rights or Developer Mode (WinError 1314). The product does
    not need them there (it falls back to hard links or copies), but some fixtures are built
    from symlinks. Asked once, by trying, so an admin Windows runner still runs those tests.
    """
    if not _SYMLINKS:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "target"
            target.write_text("x", encoding="utf-8")
            try:
                (Path(folder) / "link").symlink_to(target)
                _SYMLINKS.append(True)
            except OSError:
                _SYMLINKS.append(False)
    return _SYMLINKS[0]


NO_SYMLINKS = "this machine refuses symlinks (Windows without admin rights or Developer Mode)"


def pytest_collection_modifyitems(config, items):
    """Skip `@pytest.mark.symlinks` where links are refused and `@pytest.mark.posix` on Windows."""
    for item in items:
        if "symlinks" in item.keywords and not symlinks_allowed():
            item.add_marker(pytest.mark.skip(reason=NO_SYMLINKS))
        if "posix" in item.keywords and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="POSIX-only behaviour, not run on Windows"))


@pytest.fixture
def needs_symlinks():
    """For a fixture built from symlinks: skips the test where they are refused."""
    if not symlinks_allowed():
        pytest.skip(NO_SYMLINKS)


@pytest.fixture(autouse=True)
def _private_model_folder(tmp_path_factory, monkeypatch):
    """Every test sees an empty background-remover folder, never the real ~/.u2net.

    Two installer tests used to pass only because this machine already had the remover in
    ~/.u2net, so their "install it if missing" step quietly skipped a real download. On a
    clean machine, or after the remover file changed, they reached the internet instead.
    """
    monkeypatch.setenv("U2NET_HOME", str(tmp_path_factory.mktemp("u2net")))
