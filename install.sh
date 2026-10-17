#!/usr/bin/env bash
# Install or update image-to-3dlab on a Mac (Apple Silicon) or Linux (NVIDIA or AMD).
#
#   curl -fsSL https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.sh | bash
#
# Re-running it is how you update: it moves to the newest release and refreshes the
# Python packages. It never downloads model weights. Those are chosen, sized and agreed
# to in the viewer's Setup & Status page.
#
# Options (after `bash -s --` when piping):
#   --dir PATH     where to install (default: ~/image-to-3dlab)
#   --ref REF      a release tag or branch (default: the newest vX.Y.Z release)
#   --repo URL     clone from here instead of GitHub (a fork, or a local copy for testing)
#   --yes          no questions; for scripts and agents
#   --start        start the lab when done (the default when run in a terminal)
#   --no-start     only install; start it later with ./lab
#   --dry-run      say what would happen, change nothing
set -euo pipefail

REPO_URL="https://github.com/Bingeljell/image-to-3dlab.git"
DIR="${HOME}/image-to-3dlab"
REF=""
YES=0
DRY=0
START=""

say()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then say "would run: $*"; else "$@"; fi; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --start) START=1; shift ;;
    --no-start) START=0; shift ;;
    -h|--help) sed -n '2,19p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' \
                 || echo "Options: --dir PATH --ref REF --repo URL --yes --start --no-start --dry-run"; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# Ask on the terminal, not stdin: under `curl | bash`, stdin is this script.
confirm() {
  [ "$YES" = 1 ] && return 0
  # Opening it is the only honest test: the file exists even with no terminal attached.
  if ! { : < /dev/tty; } 2>/dev/null; then
    die "$1 Nobody to ask here, so stopping. Re-run with --yes to go ahead."
  fi
  printf '%s [y/N] ' "$1" > /dev/tty
  read -r answer < /dev/tty || answer=""
  case "$answer" in y|Y|yes|YES) return 0 ;; *) die "Stopped. Nothing was changed." ;; esac
}

# --- 1. This machine --------------------------------------------------------------------
OS="${I3D_UNAME_S:-$(uname -s)}"      # the I3D_ overrides exist for the test suite
ARCH="${I3D_UNAME_M:-$(uname -m)}"
GPU="none"
case "$OS/$ARCH" in
  Darwin/arm64)
    MACHINE="Apple Silicon Mac" ;;
  Linux/x86_64)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
      MACHINE="Linux with an NVIDIA GPU"
      GPU="nvidia"
    elif [ -n "$(ls -A "${I3D_KFD_NODES:-/sys/class/kfd/kfd/topology/nodes}" 2>/dev/null)" ]; then
      # The same test image_to_3dlab/host.py uses: one directory per AMD card the kernel
      # driver has bound. I3D_KFD_NODES exists for the test suite, the way I3D_UNAME does.
      MACHINE="Linux with an AMD GPU"
      GPU="amd"
    else
      MACHINE="Linux without a supported GPU"
      say "No NVIDIA or AMD GPU found. The viewer will install, but no generation route"
      say "will run here until one is."
    fi ;;
  Darwin/*) die "This Mac has an Intel chip. The lab needs Apple Silicon (M1 or newer)." ;;
  *) die "Unsupported machine: $OS/$ARCH. The lab runs on Apple Silicon Macs and x86-64 Linux with an NVIDIA or AMD GPU. On Windows, use install.ps1." ;;
esac
say "Machine: $MACHINE"

# --- 2. Tools ---------------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  if [ "$OS" = Darwin ]; then die "git is missing. Install it with: xcode-select --install"
  else die "git is missing. Install it with your package manager, e.g. sudo apt install git"
  fi
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "uv (Astral's Python installer, ~40 MB) is needed to set up Python 3.11."
  confirm "Install uv from astral.sh?"
  if [ "$DRY" = 1 ]; then
    say "would run: curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi
UV="$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")"

# --- 3. Code: clone, or update an existing install ---------------------------------------
if [ -d "$DIR/.git" ]; then
  say "Updating the install in $DIR"
  # Viewers up to 0.3.6 wrote learned ETA timings into this tracked file, which then
  # blocked every update. Keep the timings where newer viewers look, and restore the file.
  BASELINE="viewer/generate_baseline.json"
  if [ "$DRY" != 1 ] && [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no -- "$BASELINE")" ]; then
    mkdir -p "$DIR/output"
    [ -e "$DIR/output/.generate_baseline.json" ] || cat "$DIR/$BASELINE" > "$DIR/output/.generate_baseline.json"
    git -C "$DIR" checkout --quiet -- "$BASELINE"
  fi
  # Tracked edits would be overwritten by a checkout; untracked files (outputs, weights,
  # vendor/) are git-ignored or left alone.
  if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]; then
    die "$DIR has local edits to tracked files. Commit or stash them, then run this again."
  fi
  run git -C "$DIR" fetch --tags --force --quiet "$REPO_URL" '+refs/heads/*:refs/remotes/origin/*'
elif [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
  die "$DIR exists and is not an image-to-3dlab install. Pick another place with --dir."
else
  say "Installing into $DIR"
  run git clone --quiet "$REPO_URL" "$DIR"
fi

newest_release() {
  git -C "$DIR" tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname 2>/dev/null | head -n 1
}
if [ -z "$REF" ]; then
  if [ "$DRY" = 1 ] && [ ! -d "$DIR/.git" ]; then REF="(newest release)"
  else REF="$(newest_release)"; [ -n "$REF" ] || REF="main"
  fi
fi
say "Version: $REF"
if [ "$DRY" = 1 ] && [ ! -d "$DIR/.git" ]; then
  say "would run: git -C $DIR checkout $REF"
elif git -C "$DIR" rev-parse --verify --quiet "refs/remotes/origin/$REF" >/dev/null; then
  run git -C "$DIR" checkout --quiet -B "$REF" "origin/$REF"
else
  run git -C "$DIR" checkout --quiet "$REF"
fi

# --- 4. Python ----------------------------------------------------------------------------
say "Setting up Python 3.11 and the viewer's packages (a few hundred MB, mostly PyTorch)"
run "$UV" venv --quiet --allow-existing --python 3.11 "$DIR/.venv"
PY="$DIR/.venv/bin/python"

# The lock pins every package to versions tested together; requirements.txt only bounds
# them, so installing from it gave each install whatever PyPI had that day.
REQS="$DIR/requirements.lock"
[ -f "$REQS" ] || REQS="$DIR/requirements.txt"

# PyPI's Linux PyTorch is a CUDA build, and the lock carries about 6 GB of NVIDIA wheels
# with it. On an AMD card none of that can load, and the box is left with a torch that
# reports no device. So the lock is installed with that family taken out and the ROCm build
# is put in its place. image_to_3dlab.host picks the ROCm build that matches the installed
# runtime (tested in tests/test_host.py); the CUDA and Apple paths keep the plain install.
if [ "$GPU" = amd ] && [ -f "$REQS" ] && [ "$DRY" != 1 ] && [ -x "$PY" ]; then
  # Inside the venv: it belongs to this install, and git already ignores it there.
  ROCM_LOCK="$DIR/.venv/requirements.rocm.lock"
  # From inside $DIR: `-m` finds the package in the current folder, and under
  # `curl | bash` that folder is wherever the user happened to be.
  (cd "$DIR" && "$PY" -m image_to_3dlab.host without-cuda "$REQS") >"$ROCM_LOCK"
  REQS="$ROCM_LOCK"
fi

run "$UV" pip install --quiet --python "$PY" -r "$REQS"

# After the requirements, so it wins over the PyPI wheel: --reinstall-package is what makes
# uv swap the build rather than consider the requirement satisfied.
if [ "$GPU" = amd ] && [ "$DRY" != 1 ] && [ -x "$PY" ]; then
  ROCM_INDEX="$(cd "$DIR" && "$PY" -m image_to_3dlab.host torch-rocm-index | tr -d '[:space:]')"
  if ! (cd "$DIR" && "$PY" -m image_to_3dlab.host torch-has-rocm); then
    if [ -n "$ROCM_INDEX" ]; then
      say "Installing PyTorch with ROCm from $ROCM_INDEX"
      run "$UV" pip install --quiet --python "$PY" --reinstall-package torch \
        --reinstall-package torchvision --index-url "$ROCM_INDEX" torch torchvision
    else
      # The lock went in without torch, so without this there would be no PyTorch at all.
      say "Your ROCm is older than PyTorch's ROCm builds (they need ROCm 6.2 or newer)."
      say "Installing the CPU build of PyTorch for now. Install a newer ROCm, then run this"
      say "again to get the GPU build."
      run "$UV" pip install --quiet --python "$PY" \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision
    fi
  fi
fi

# --- 5. Blender: Finish needs it, and it is the user's to install -----------------------
# Checked with the same finder Finish uses. Missing is a note, not a failure: generating
# models works without it.
if [ "$DRY" != 1 ] && [ -x "$PY" ] && ! (cd "$DIR" && .venv/bin/python -c \
    'import sys; from image_to_3dlab.blender import find_blender; sys.exit(0 if find_blender() else 1)') \
    2>/dev/null; then
  say "Blender was not found. Finish (the low-poly clean-up) needs Blender 4.2 or newer."
  if [ "$OS" = Linux ]; then
    say "  Press Install Blender in the viewer's Setup & Status (no admin rights needed)."
  else
    say "  https://www.blender.org/download/"
  fi
fi

# --- 6. Done: start the lab, so the next thing a new user sees is the viewer -------------
# In a terminal it just starts; scripts and agents (--yes, or no terminal) get the command,
# so they are never left holding a running server.
if [ -z "$START" ]; then
  if [ "$YES" != 1 ] && [ -t 1 ]; then START=1; else START=0; fi
fi
say "Done. Open Setup & Status in the viewer to choose what to install. Nothing large"
say "downloads until you say so there. To update later, run this installer again."
say "Use it from your agent too (Claude Code shown; Codex, Kimi and others: docs/agents.md):"
printf '\n    claude mcp add assetfurnace -- %s/lab mcp\n\n' "$DIR"
if [ "$START" = 1 ] && [ "$DRY" != 1 ] && [ -x "$DIR/lab" ]; then
  say "Starting the lab. Ctrl-C stops it; start it again any time with: cd $DIR && ./lab"
  exec "$DIR/lab"
fi
say "Start the lab with:"
printf '\n    cd %s && ./lab\n\n' "$DIR"
