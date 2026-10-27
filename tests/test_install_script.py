"""install.sh, run for real against a throwaway git repo and a fake `uv`.

The fake records what it was asked to do, so these tests cover the whole script (machine
check, clone, update, version choice, refusals) in seconds, without installing Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    """A PATH holding git, the basics, a fake `uv`, and a fake `mise`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("git", "uname", "sed", "ls", "head", "grep", "cat", "env", "mkdir", "dirname",
                 "cp", "chmod", "rm", "mv", "tr", "printf", "curl", "sh"):
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/uv.log"\n')
    uv.chmod(0o755)
    mise = bin_dir / "mise"
    mise.write_text(f'#!{sys.executable}\n'
                    f'import sys, os\n'
                    f'with open("{tmp_path}/mise.log", "a") as f:\n'
                    f'    f.write(" ".join(sys.argv[1:]) + "\\n")\n'
                    f'if "exec" in sys.argv and "--" in sys.argv:\n'
                    f'    idx = sys.argv.index("--") + 1\n'
                    f'    os.execvp(sys.argv[idx], sys.argv[idx:])\n')
    mise.chmod(0o755)
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
    assert "sync" in uv_calls and "--frozen" in uv_calls and "--no-dev" in uv_calls
    assert "requirements.lock" not in uv_calls
    assert "install" in (tmp_path / "mise.log").read_text()


def test_install_syncs_without_the_requirements_lock(tmp_path, fake_bin, upstream):
    result = run(tmp_path, fake_bin, "--dir", str(tmp_path / "lab"), "--repo", str(upstream), "--yes")
    assert result.returncode == 0, result.stderr
    uv_log = (tmp_path / "uv.log").read_text()
    assert "sync" in uv_log and "--frozen" in uv_log and "--no-dev" in uv_log
    assert "requirements.lock" not in uv_log
    assert "install" in (tmp_path / "mise.log").read_text()


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


def test_without_mise_and_without_a_terminal_it_stops_instead_of_guessing(
        tmp_path, upstream, fake_bin):
    (fake_bin / "mise").unlink()
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"))
    assert done.returncode != 0
    assert "--yes" in done.stderr
    assert not (tmp_path / "lab").exists()


def test_mise_is_installed_by_the_script_when_it_is_missing(
        tmp_path, upstream, fake_bin):
    """The installer bootstraps mise via curl|sh when it is not already on PATH."""
    template = tmp_path / "mise-template"
    template.write_text((fake_bin / "mise").read_text())
    (fake_bin / "mise").unlink()
    curl = fake_bin / "curl"
    curl.unlink()
    curl.write_text(f'''#!/bin/sh
if [ "$1" = "-sSL" ] && [ "$2" = "https://mise.run" ]; then
  mkdir -p "$HOME/.local/bin"
  cat > "$HOME/.local/bin/mise" <<'EOF'
{template.read_text()}
EOF
  chmod +x "$HOME/.local/bin/mise"
else
  echo "unexpected curl: $@" >&2
  exit 1
fi
''')
    curl.chmod(0o755)
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               "--yes")
    assert done.returncode == 0, done.stderr
    assert "mise: command not found" not in done.stderr
    assert "install" in (tmp_path / "mise.log").read_text()
    assert (tmp_path / "home" / ".local" / "bin" / "mise").exists()


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


def test_it_prints_the_line_that_connects_an_agent(tmp_path, upstream, fake_bin):
    _with_lab(upstream)
    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"))
    assert done.returncode == 0, done.stderr
    assert f"claude mcp add assetfurnace -- {tmp_path / 'lab'}/lab mcp" in done.stdout


def test_the_real_launcher_is_executable_and_uses_auto():
    lab = SCRIPT.parent / "lab"
    assert lab.is_file() and lab.stat().st_mode & 0o111
    assert "serve.py --auto" in lab.read_text()


def test_the_uv_lock_pins_rembg():
    lock = (SCRIPT.parent / "uv.lock").read_text()
    assert 'name = "rembg"' in lock
    assert 'version = "2.0.86"' in lock


# --- AMD: the installer must not hand an AMD card a CUDA torch ----------------------------

# A real slice of the shipped lock. PyPI's Linux torch IS a CUDA build and brings about 6 GB
# of NVIDIA wheels with it, none of which an AMD card can load.
LOCK_WITH_CUDA = """\
# Generated by uv pip compile
accelerate==1.12.0 ; sys_platform == 'linux'
cuda-toolkit==13.0.3.0 ; sys_platform == 'linux'
nvidia-cudnn-cu13==9.24.0.43 ; sys_platform == 'linux'
torch==2.14.1
torchvision==0.29.1
triton==3.8.0 ; sys_platform == 'linux'
numpy==1.26.4
"""


@pytest.fixture
def amd_venv(tmp_path: Path):
    """A fake `uv venv` result: an interpreter that answers host.py's questions.

    It delegates the lock filtering to the *real* image_to_3dlab.host, so the test covers
    the shipped filter rather than a re-implementation of it, and answers the two GPU
    questions from parameters so the result does not depend on the card running the suite.
    """
    real_python = sys.executable
    bin_dir = tmp_path / "venv-bin"

    def build(rocm_already: bool, index: str):
        bin_dir.mkdir(exist_ok=True)
        python = bin_dir / "python"
        python.write_text(f"""#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "image_to_3dlab.host" ]; then
  # Like real Python: `-m` only finds the package from inside the install folder.
  if [ "$PWD" != "{tmp_path / 'lab'}" ]; then
    echo "No module named image_to_3dlab" >&2; exit 1
  fi
  shift 2
  case "$1" in
    torch-has-rocm) exit {0 if rocm_already else 1} ;;
    torch-rocm-index) echo "{index}"; exit 0 ;;
    *) PYTHONPATH="{REPO}" exec "{real_python}" -m image_to_3dlab.host "$@" ;;
  esac
fi
echo "$@" >> "{tmp_path}/venv-python.log"
""")
        python.chmod(0o755)
        return python

    build.rocm_already = build
    return build


def _amd_run(tmp_path, fake_bin, upstream, amd_venv, rocm_already=False,
             index="https://download.pytorch.org/whl/rocm7.2"):
    """install.sh on an AMD machine, with the fake venv interpreter already in place."""
    python = amd_venv(rocm_already, index)
    uv = fake_bin / "uv"
    uv.write_text(f'''#!/bin/sh
echo "$@" >> "{tmp_path}/uv.log"
file=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output-file) file="$2"; shift 2 ;;
    venv)
      allow=0
      dir=""
      shift
      while [ $# -gt 0 ]; do
        case "$1" in
          --allow-existing) allow=1; shift ;;
          -*) shift ;;
          *) dir="$1"; shift ;;
        esac
      done
      if [ -d "$dir" ] && [ "$allow" != 1 ]; then
        echo "A virtual environment already exists at $dir" >&2
        exit 1
      fi
      mkdir -p "$dir/bin"
      cp "{python}" "$dir/bin/python"
      ;;
    *) shift ;;
  esac
done
if [ -n "$file" ]; then
  cat > "$file" <<'EOF'
{LOCK_WITH_CUDA}
EOF
fi
''')
    uv.chmod(0o755)
    nodes = tmp_path / "nodes"
    (nodes / "0").mkdir(parents=True, exist_ok=True)
    return run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               machine=("Linux", "x86_64"), env={"I3D_KFD_NODES": str(nodes)})


def test_an_amd_install_never_puts_cuda_wheels_on_the_card(tmp_path, upstream, fake_bin,
                                                           amd_venv):
    """PyPI's Linux torch is a CUDA build, so the shared lock installs one on an AMD box:
    a torch that reports no device, plus about 6 GB of NVIDIA libraries it cannot load."""
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.4")

    done = _amd_run(tmp_path, fake_bin, upstream, amd_venv)
    assert done.returncode == 0, done.stderr

    uv_calls = (tmp_path / "uv.log").read_text()
    assert "export" in uv_calls
    assert "pip install" in uv_calls and ".venv/requirements.rocm.txt" in uv_calls
    assert "sync" not in uv_calls
    assert "requirements.lock" not in uv_calls
    # And it really was filtered, by the real code rather than a sed in the script.
    filtered = (tmp_path / "lab" / ".venv" / "requirements.rocm.txt").read_text()
    assert "numpy==1.26.4" in filtered and "accelerate==1.12.0" in filtered
    assert "torch==" not in filtered
    assert "nvidia-cudnn-cu13" not in filtered and "cuda-toolkit" not in filtered


def test_an_amd_install_gets_the_rocm_build_of_torch(tmp_path, upstream, fake_bin, amd_venv):
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.5")

    done = _amd_run(tmp_path, fake_bin, upstream, amd_venv)
    assert done.returncode == 0, done.stderr
    uv_calls = (tmp_path / "uv.log").read_text()
    assert "https://download.pytorch.org/whl/rocm7.2" in uv_calls
    # --reinstall-package is what makes uv swap the PyPI build instead of considering
    # "torch" already satisfied.
    assert "--reinstall-package torch" in uv_calls


def test_an_amd_that_already_has_rocm_torch_is_left_alone(tmp_path, upstream, fake_bin,
                                                          amd_venv):
    """An update must not re-download a few GB of PyTorch every time."""
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.6")

    done = _amd_run(tmp_path, fake_bin, upstream, amd_venv, rocm_already=True)
    assert done.returncode == 0, done.stderr
    assert "--reinstall-package torch" not in (tmp_path / "uv.log").read_text()


def test_an_amd_with_too_old_a_rocm_is_told_to_update_and_gets_cpu_torch(
        tmp_path, upstream, fake_bin, amd_venv):
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.7")

    done = _amd_run(tmp_path, fake_bin, upstream, amd_venv, index="")
    assert done.returncode == 0, done.stderr
    assert "older than PyTorch's ROCm builds" in done.stdout
    uv_calls = (tmp_path / "uv.log").read_text()
    assert "--reinstall-package torch" not in uv_calls
    # The lock went in without torch, so the CPU build has to fill the gap.
    assert "https://download.pytorch.org/whl/cpu torch torchvision" in uv_calls


def test_an_amd_install_leaves_no_stray_file_in_the_checkout(tmp_path, upstream, fake_bin,
                                                             amd_venv):
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.9")

    done = _amd_run(tmp_path, fake_bin, upstream, amd_venv)
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "lab" / "requirements.rocm.txt").exists()


def test_an_amd_install_can_be_re_run_over_the_existing_venv(tmp_path, upstream, fake_bin,
                                                             amd_venv):
    """Re-running the installer on AMD must not fail because the venv already exists."""
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.10")

    first = _amd_run(tmp_path, fake_bin, upstream, amd_venv)
    assert first.returncode == 0, first.stderr
    second = _amd_run(tmp_path, fake_bin, upstream, amd_venv)
    assert second.returncode == 0, second.stderr


def test_an_nvidia_install_is_left_alone(tmp_path, upstream, fake_bin, amd_venv):
    """The ROCm branch is AMD's. An NVIDIA card's lock is already right, and the plain
    install must not have its pins stripped out from under it."""
    (upstream / "requirements.lock").write_text(LOCK_WITH_CUDA)
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "lock")
    _git(upstream, "tag", "v0.2.8")

    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 'GPU 0: Test'\n")
    nvidia_smi.chmod(0o755)
    env = {"I3D_KFD_NODES": str(tmp_path / "none")}

    done = run(tmp_path, fake_bin, "--repo", str(upstream), "--dir", str(tmp_path / "lab"),
               machine=("Linux", "x86_64"), env=env)
    assert done.returncode == 0, done.stderr
    assert "Machine: Linux with an NVIDIA GPU" in done.stdout
    uv_calls = (tmp_path / "uv.log").read_text()
    assert "sync" in uv_calls and "--frozen" in uv_calls and "--no-dev" in uv_calls
    assert "requirements.lock" not in uv_calls
    assert "requirements.rocm.lock" not in uv_calls
    assert "rocm" not in uv_calls


def test_the_uv_lock_covers_every_requirement():
    import re
    import tomllib

    lock = tomllib.loads((SCRIPT.parent / "uv.lock").read_text())
    locked_names = {pkg["name"].lower().replace("_", "-") for pkg in lock["package"]}
    pyproject = tomllib.loads((SCRIPT.parent / "pyproject.toml").read_text())
    for raw in pyproject["project"]["dependencies"]:
        name = re.split(r"[<>=!~; ]", raw.split("[", 1)[0], maxsplit=1)[0].lower().replace("_", "-")
        assert name in locked_names, f"{name} is in pyproject.toml but not uv.lock"


def test_windows_lab_starts_the_same_studio_as_the_shell_one():
    """lab.cmd cannot run here: check it runs what ./lab runs, from its own folder, with the
    Windows venv layout and Windows line endings (cmd misreads some lines without them)."""
    shell = (REPO / "lab").read_text()
    cmd = (REPO / "lab.cmd").read_bytes()
    assert b"\r\n" in cmd and b"\n" not in cmd.replace(b"\r\n", b"")
    text = cmd.decode()
    assert "viewer/serve.py --auto" in shell and "viewer\\serve.py --auto %*" in text
    assert "image_to_3dlab.mcp_server" in shell and "image_to_3dlab.mcp_server" in text
    assert '.venv\\Scripts\\python.exe' in text and 'cd /d "%~dp0"' in text
    assert ".\\lab" in (REPO / "install.ps1").read_text()
