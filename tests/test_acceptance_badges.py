"""Execute the workflow's publisher against a local bare Git repository."""

import os
import shlex
import subprocess
import sys

import pytest
import yaml
from test_merge_gate import WORKFLOW

pytestmark = pytest.mark.posix
BADGES = {
    "python-lines.svg",
    "python-branches.svg",
    "javascript-lines.svg",
    "javascript-branches.svg",
}


def _executable(path, body):
    path.write_text(f"#!/bin/bash\n{body}", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def publisher(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    artifact = tmp_path / "badge-artifact"
    artifact.mkdir()
    env = {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ALLOW_PROTOCOL": "file",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "AF_TEST_FIXTURE_ROOT": str(tmp_path),
    }
    if sys.platform == "darwin":
        # Ubuntu uses GNU install. macOS BSD install needs the destination
        # check spelled out to give -T semantics.
        tools = tmp_path / "bin"
        tools.mkdir()
        _executable(tools / "install", """
set -eu
[[ "$#" == 5 && "$1" == -T && "$2" == -m && "$3" == 0644 ]]
[[ ! -d "$5" ]]
shift
exec /usr/bin/install "$@"
""")
        env["PATH"] = f"{tools}{os.pathsep}{env['PATH']}"

    def git(cwd, *args):
        result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    actor = tmp_path / "main-actor"
    git(tmp_path, "clone", remote.as_uri(), str(actor))
    git(actor, "config", "user.name", "Fixture")
    git(actor, "config", "user.email", "fixture@example.invalid")
    (actor / "main.txt").write_text("main\n", encoding="utf-8")
    git(actor, "add", "main.txt")
    git(actor, "commit", "-m", "Initial main")
    git(actor, "push", "origin", "HEAD:refs/heads/main")
    main = git(remote, "rev-parse", "refs/heads/main")

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    step = next(step for step in workflow["jobs"]["publish-coverage-badges"]["steps"] if "run" in step)
    assert step["shell"] == "bash"
    script = tmp_path / "publish"
    _executable(script, "set -eo pipefail\n" + step["run"])
    checkout = tmp_path / step["working-directory"]

    def publish(*, sha=main, advance=False):
        if checkout.exists():
            # Each Actions job receives a new checkout of its source revision.
            checkout.rename(tmp_path / f"finished-{len(list(tmp_path.glob('finished-*')))}")
        git(tmp_path, "clone", remote.as_uri(), str(checkout))
        git(checkout, "switch", "--detach", sha)
        if advance:
            # A separate actor advances main after the publisher commits and
            # before it checks the remote a second time.
            hook = checkout / ".git/hooks/post-commit"
            _executable(hook, f"exec git -C {shlex.quote(str(actor))} push origin HEAD:refs/heads/main\n")
        return subprocess.run(
            [str(script)], cwd=checkout, env={**env, "GITHUB_SHA": sha},
            capture_output=True, text=True, timeout=15,
        )

    return git, remote, actor, artifact, main, publish


def _write_artifact(artifact, value):
    for name in BADGES | {"unexpected.svg"}:
        (artifact / name).write_text(f"<svg><title>{value}</title></svg>\n", encoding="utf-8")


def _publish_first(git, remote, artifact, main, publish):
    _write_artifact(artifact, "first")
    result = publish()
    assert result.returncode == 0, result.stdout + result.stderr
    tip = git(remote, "rev-parse", "refs/heads/coverage-badges")
    assert git(remote, "rev-list", "--parents", "-n", "1", tip) == tip
    assert set(git(remote, "ls-tree", "-r", "--name-only", tip).splitlines()) == BADGES
    assert git(remote, "show", "-s", "--format=%B", tip).splitlines()[-1] == f"Source: {main}"
    for name in BADGES:
        assert git(remote, "show", f"{tip}:{name}") == (artifact / name).read_text(encoding="utf-8").strip()
    assert git(remote, "rev-parse", "refs/heads/main") == main
    return tip


def test_publisher_creates_updates_and_skips_identical_badges(publisher):
    git, remote, actor, artifact, main, publish = publisher
    first = _publish_first(git, remote, artifact, main, publish)

    git(actor, "fetch", "origin", "refs/heads/coverage-badges")
    git(actor, "switch", "--detach", first)
    (actor / "keep.txt").write_text("Keep me\n", encoding="utf-8")
    git(actor, "add", "keep.txt")
    git(actor, "commit", "-m", "Add unrelated file")
    git(actor, "push", "origin", "HEAD:refs/heads/coverage-badges")
    previous = git(remote, "rev-parse", "refs/heads/coverage-badges")
    _write_artifact(artifact, "changed")

    result = publish()
    assert result.returncode == 0, result.stdout + result.stderr
    updated = git(remote, "rev-parse", "refs/heads/coverage-badges")
    assert git(remote, "rev-parse", f"{updated}^") == previous
    assert set(git(remote, "diff", "--name-only", previous, updated).splitlines()) == BADGES
    assert git(remote, "show", f"{updated}:keep.txt") == "Keep me"
    for name in BADGES:
        assert git(remote, "show", f"{updated}:{name}") == (artifact / name).read_text(encoding="utf-8").strip()
    assert git(remote, "show", "-s", "--format=%B", updated).splitlines()[-1] == f"Source: {main}"

    result = publish()
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(remote, "rev-parse", "refs/heads/coverage-badges") == updated
    assert git(remote, "rev-parse", "refs/heads/main") == main


@pytest.mark.parametrize("case", ["stale-at-entry", "main-advances", "missing-svg"])
def test_publisher_keeps_badge_tip_on_stale_or_missing_input(publisher, case):
    git, remote, actor, artifact, main, publish = publisher
    previous = _publish_first(git, remote, artifact, main, publish)
    _write_artifact(artifact, "would change")
    latest = main
    if case != "missing-svg":
        git(actor, "commit", "--allow-empty", "-m", "Advance main independently")
        latest = git(actor, "rev-parse", "HEAD")
    if case == "stale-at-entry":
        git(actor, "push", "origin", "HEAD:refs/heads/main")
        result = publish()
        assert result.returncode == 0
        assert "Main has advanced." in result.stdout
    elif case == "main-advances":
        result = publish(advance=True)
        assert result.returncode == 0
        assert "Main advanced while preparing badges." in result.stdout
    else:
        (artifact / "javascript-branches.svg").unlink()
        result = publish()
        assert result.returncode != 0
        assert "javascript-branches.svg" in result.stderr
    assert git(remote, "rev-parse", "refs/heads/coverage-badges") == previous
    assert git(remote, "rev-parse", "refs/heads/main") == latest
