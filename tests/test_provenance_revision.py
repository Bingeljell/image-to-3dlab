"""A sidecar must record which code produced the asset, not only which parameters.

Provenance already captured `backend_revision` -- the vendored TRELLIS checkout. But
the patches that change what TRELLIS does (`scripts/patch_trellis_*.py`) live in *this*
repository, so two runs with byte-identical parameters could behave differently and the
sidecars would be indistinguishable.

That is not hypothetical. On 2026-08-10 a four-subject comparison silently mixed three
code states: the pangolin predated `40aaf9f`, which is the commit that made
`bake_target_faces` take effect on Metal at all -- so its recorded budget of 200,000 was
never applied. The parameters said one thing and the binary did another, and nothing in
the sidecar could reveal it.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.provenance import _git_revision, _pipeline_revision

REPO = Path(__file__).resolve().parents[1]


def test_pipeline_revision_reports_this_repository():
    revision = _pipeline_revision()
    assert revision is not None, "the pipeline repo is a git checkout; this must resolve"

    expected = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert revision["commit"] == expected


def test_pipeline_revision_flags_uncommitted_changes():
    """A dirty tree means the commit alone does not identify the code that ran."""
    revision = _pipeline_revision()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )
    assert revision["dirty"] is dirty


def test_pipeline_revision_is_not_the_backend_revision():
    """The two must be distinct fields -- conflating them reintroduces the blind spot."""
    backend = _git_revision(REPO / "vendor" / "trellis-mac")
    pipeline = _pipeline_revision()
    if backend is not None:
        assert backend != pipeline["commit"]


def test_git_revision_returns_none_outside_a_repository(tmp_path):
    assert _git_revision(tmp_path) is None


def test_pipeline_revision_survives_a_missing_git(monkeypatch):
    """Provenance must degrade, never crash, when git is unavailable."""

    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _pipeline_revision() is None
