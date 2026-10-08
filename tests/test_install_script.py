"""install.sh, run for real against a throwaway git repo and a fake `uv`.

The fake records what it was asked to do, so these tests cover the whole script (machine
check, clone, update, version choice, refusals) in seconds, without installing Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "install.sh"
BASH = shutil.which("bash")
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(not (BASH and GIT), reason="needs bash and git")


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run([GIT, *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A tiny stand-in for the GitHub repo: two releases and a non-release tag."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "requirements.txt").write_text("Pillow\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    _git(repo, "tag", "v0.1.0")
    (repo / "VERSION").write_text("0.2.0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "two")
    _git(repo, "tag", "v0.2.0")
    # A later tag that is not a release must never be picked as "newest release".
    (repo / "VERSION").write_text("backup\n")
    _git(repo, "commit", "-q", "-am", "three")
    _git(repo, "tag", "backup-before-rebase-2026-08-19")
    return repo


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """A PATH holding git, the basics and a `uv` that logs its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("git", "uname", "sed", "ls", "head", "grep", "cat", "env", "mkdir", "dirname"):
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/uv.log"\n')
    uv.chmod(0o755)
    return bin_dir


def run(tmp_path: Path, path: Path, *args: str, machine=("Darwin", "arm64"), env=None):
    env = {"PATH": str(path), "HOME": str(tmp_path / "home"),
           "I3D_UNAME_S": machine[0], "I3D_UNAME_M": machine[1],
           # The test machine has no card of any kind, whatever the host running the suite
           # has: I3D_KFD_NODES points the AMD check somewhere empty by default.
           "I3D_KFD_NODES": str(tmp_path / "no-kfd-nodes"), **(env or {})}
    # A new session has no controlling terminal, like a CI job or an agent.
    return subprocess.run([BASH, str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, start_new_session=True,
                          check=False)


def test_script_parses():
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)


def test_fresh_install_picks_the_newest_release_not_a_backup_tag(tmp_path, upstream,
                                                                 fake_bin):
    target = tmp_path / "lab"
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    assert done.returncode == 0, done.stderr
    assert (target / "VERSION").read_text() == "0.2.0\n"
    assert "Version: v0.2.0" in done.stdout
    uv_calls = (tmp_path / "uv.log").read_text()
    assert "venv" in uv_calls and "--python 3.11" in uv_calls
    assert f"-r {target}/requirements.txt" in uv_calls


def test_rerunning_updates_to_a_new_release(tmp_path, upstream, fake_bin):
    target = tmp_path / "lab"
    run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    (upstream / "VERSION").write_text("0.3.0\n")
    _git(upstream, "commit", "-q", "-am", "four")
    _git(upstream, "tag", "v0.3.0")
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    assert done.returncode == 0, done.stderr
    assert "Updating" in done.stdout
    assert (target / "VERSION").read_text() == "0.3.0\n"


def test_update_refuses_to_overwrite_local_edits(tmp_path, upstream, fake_bin):
    target = tmp_path / "lab"
    run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    (target / "VERSION").write_text("my own edit\n")
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    assert done.returncode != 0
    assert "local edits" in done.stderr
    assert (target / "VERSION").read_text() == "my own edit\n"


def test_a_branch_can_be_installed_for_testing(tmp_path, upstream, fake_bin):
    _git(upstream, "checkout", "-q", "-b", "feat/x")
    (upstream / "VERSION").write_text("branch\n")
    _git(upstream, "commit", "-q", "-am", "branch")
    target = tmp_path / "lab"
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target),
               "--ref", "feat/x")
    assert done.returncode == 0, done.stderr
    assert (target / "VERSION").read_text() == "branch\n"


def test_an_unrelated_folder_is_never_touched(tmp_path, upstream, fake_bin):
    target = tmp_path / "lab"
    target.mkdir()
    (target / "thesis.docx").write_text("precious")
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    assert done.returncode != 0 and "not an image-to-3dlab install" in done.stderr
    assert (target / "thesis.docx").read_text() == "precious"


def test_dry_run_changes_nothing(tmp_path, upstream, fake_bin):
    target = tmp_path / "lab"
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target),
               "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "would run: git clone" in done.stdout
    assert not target.exists()
    assert not (tmp_path / "uv.log").exists()


def test_without_uv_and_without_a_terminal_it_stops_instead_of_guessing(
        tmp_path, upstream, fake_bin):
    (fake_bin / "uv").unlink()
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"))
    assert done.returncode != 0
    assert "--yes" in done.stderr
    assert not (tmp_path / "lab").exists()


@pytest.mark.parametrize("machine,message", [
    (("Darwin", "x86_64"), "Intel"),
    (("MINGW64_NT-10.0", "x86_64"), "install.ps1"),
    (("Linux", "aarch64"), "Unsupported machine"),
])
def test_unsupported_machines_are_refused_by_name(tmp_path, upstream, fake_bin,
                                                  machine, message):
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               machine=machine)
    assert done.returncode != 0 and message in done.stderr


def test_linux_without_a_gpu_installs_but_says_so(tmp_path, upstream, fake_bin):
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               machine=("Linux", "x86_64"), env={"I3D_KFD_NODES": str(tmp_path / "none")})
    assert done.returncode == 0, done.stderr
    assert "No NVIDIA or AMD GPU found" in done.stdout


def test_linux_with_an_amd_card_is_named_as_one(tmp_path, upstream, fake_bin):
    """The installer has to recognise the machine, or it warns a user who has a perfectly
    good card that no route will run."""
    nodes = tmp_path / "nodes"
    # One directory per bound card is what the kernel driver leaves behind, so an empty
    # directory means no card.
    (nodes / "0").mkdir(parents=True)
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               machine=("Linux", "x86_64"), env={"I3D_KFD_NODES": str(nodes)})
    assert done.returncode == 0, done.stderr
    assert "Machine: Linux with an AMD GPU" in done.stdout
    assert "No NVIDIA or AMD GPU found" not in done.stdout


def test_it_never_mentions_downloading_weights_itself():
    """Weights are chosen in Setup & Status, with sizes and consent. The installer's job
    ends at code and Python packages."""
    text = SCRIPT.read_text()
    assert "hf_hub_download" not in text and "snapshot_download" not in text
    assert "bootstrap_" not in text


def test_windows_installer_keeps_step_with_the_shell_one():
    """install.ps1 cannot run here, so check what can be checked: it picks releases by the
    same rule, never fetches weights, and says it is untested."""
    ps1 = (REPO / "install.ps1").read_text()
    assert "v[0-9]*.[0-9]*.[0-9]*" in ps1 and "v[0-9]*.[0-9]*.[0-9]*" in SCRIPT.read_text()
    assert "--untracked-files=no" in ps1
    assert "hf_hub_download" not in ps1 and "bootstrap_" not in ps1
    assert "LIMITED TESTING" in ps1


def test_windows_installer_gets_cuda_pytorch_before_the_requirements():
    """PyPI only has CPU-only torch for Windows. The CUDA build must come from PyTorch's
    index, chosen by the tested helper, and before requirements.txt (which would otherwise
    settle for the CPU wheel and never replace it)."""
    ps1 = (REPO / "install.ps1").read_text()
    assert "image_to_3dlab.host torch-index" in ps1
    assert "image_to_3dlab.host torch-has-cuda" in ps1
    assert "--index-url $TorchIndex" in ps1
    assert ps1.index("torch-index") < ps1.index('"requirements.txt"')
    # Only on a machine with a GPU, and only when the installed torch lacks CUDA, so an
    # update does not re-download a few GB every time.
    assert "if ($HasGpu)" in ps1 and "-not $TorchHasCuda" in ps1


def test_both_installers_look_for_blender_with_finishs_own_finder_and_never_install_it():
    """Finish needs Blender; the installers say so when it is missing, using the same lookup
    Finish uses, and leave installing a 400 MB application to the user."""
    for path in (SCRIPT, REPO / "install.ps1"):
        text = path.read_text()
        assert "from image_to_3dlab.blender import find_blender" in text
        assert "blender.org/download" in text
        assert "apt install blender" not in text and "winget install" not in text


def test_update_moves_learned_timings_aside_instead_of_refusing(tmp_path, upstream, fake_bin):
    # Seen on a real pod: viewer releases up to 0.3.6 rewrote the tracked
    # viewer/generate_baseline.json after every generation, so every user who had generated
    # once was refused an update. Their timings move to output/ and the update goes on.
    (upstream / "viewer").mkdir()
    (upstream / "viewer" / "generate_baseline.json").write_text('{"seconds": {}}\n')
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "baseline")
    _git(upstream, "tag", "v0.2.1")
    target = tmp_path / "lab"
    run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    learned = '{"seconds": {"decode": 42.0}}\n'
    (target / "viewer" / "generate_baseline.json").write_text(learned)
    (upstream / "VERSION").write_text("0.3.0\n")
    _git(upstream, "commit", "-q", "-am", "four")
    _git(upstream, "tag", "v0.3.0")
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(target))
    assert done.returncode == 0, done.stderr
    assert (target / "VERSION").read_text() == "0.3.0\n"
    assert (target / "output" / ".generate_baseline.json").read_text() == learned
    assert (target / "viewer" / "generate_baseline.json").read_text() == '{"seconds": {}}\n'


def _with_lab(upstream):
    lab = upstream / "lab"
    lab.write_text('#!/bin/sh\necho "LAB STARTED $*"\n')
    lab.chmod(0o755)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lab")
    _git(upstream, "tag", "v0.2.2")


def test_start_opens_the_lab_when_asked(tmp_path, upstream, fake_bin):
    # The goal: run the curl command, end up in the viewer. Nothing to type in between.
    _with_lab(upstream)
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               "--start")
    assert done.returncode == 0, done.stderr
    assert "LAB STARTED" in done.stdout


def test_without_a_terminal_it_says_how_to_start_instead(tmp_path, upstream, fake_bin):
    # Scripts and agents (no terminal, or --yes) must not be left holding a running server.
    _with_lab(upstream)
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"))
    assert done.returncode == 0, done.stderr
    assert "LAB STARTED" not in done.stdout and "./lab" in done.stdout


def test_the_real_launcher_is_executable_and_uses_auto():
    lab = SCRIPT.parent / "lab"
    assert lab.is_file() and lab.stat().st_mode & 0o111
    assert "serve.py --auto" in lab.read_text()


def test_installs_from_the_lock_file_when_there_is_one(tmp_path, upstream, fake_bin):
    # Unpinned requirements meant every fresh install got whatever PyPI had that day: on
    # 2026-10-01 huggingface-hub jumped to 2.x between two installs.
    (upstream / "requirements.lock").write_text("Pillow==10.4.0\n")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.3")
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"))
    assert done.returncode == 0, done.stderr
    uv_calls = (tmp_path / "uv.log").read_text()
    assert "requirements.lock" in uv_calls and "requirements.txt" not in uv_calls


def test_the_shipped_lock_pins_what_broke_before():
    lock = (SCRIPT.parent / "requirements.lock").read_text()
    for line in ("numpy==1.", "rembg==2.0.69", "huggingface-hub=="):
        assert line in lock, line


def test_the_lock_covers_every_requirement():
    # Regenerate with: uv pip compile --universal --python-version 3.11 requirements.txt
    #                  -o requirements.lock
    import re

    lock = (SCRIPT.parent / "requirements.lock").read_text().lower().replace("_", "-")
    for raw in (SCRIPT.parent / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-")
        assert f"\n{name}==" in lock, f"{name} is in requirements.txt but not the lock"
