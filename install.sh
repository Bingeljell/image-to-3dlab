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
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$1" >&2; install_event stopped "${2:-other}"; exit 1; }
run()  { if [ "$DRY" = 1 ]; then say "would run: $*"; else "$@"; fi; }

# --- Anonymous install events -------------------------------------------------------------
# One tiny POST, in the background, with a 3 second ceiling: it can never fail the install or
# hold it up. No id and no address: the site keeps daily counts by event, reason, os and arch.
# The same promise as the app's counts: no pictures, prompts, files or anything private.
# I3D_NO_USAGE and DO_NOT_TRACK turn it off, exactly as they turn the app's counts off.
EVENT_URL="${I3D_EVENT_URL:-https://assetfurnace.com/api/install-event}"
EVENT_OS=""
EVENT_ARCH=""

usage_off() {
  for name in I3D_NO_USAGE DO_NOT_TRACK; do
    case "$(printf '%s' "${!name:-}" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes) return 0 ;;
    esac
  done
  return 1
}

script_version() {
  # The release this install is on, when the folder is already a checkout. Empty means the
  # clone has not happened yet, and the site records it as unknown.
  if [ -f "$DIR/pyproject.toml" ]; then
    sed -n 's/^version = "\(.*\)"$/\1/p' "$DIR/pyproject.toml" 2>/dev/null | head -n 1
  fi
}

install_event() {
  event="$1"; reason="${2:-}"
  [ "$DRY" = 1 ] && return 0
  [ -z "$EVENT_OS" ] && return 0        # not a machine we name; nothing to report about it
  usage_off && return 0
  command -v curl >/dev/null 2>&1 || return 0
  version="$(script_version)"
  body="{\"event\":\"$event\",\"os\":\"$EVENT_OS\",\"arch\":\"$EVENT_ARCH\",\"version\":\"${version:-unknown}\""
  [ -n "$reason" ] && body="$body,\"reason\":\"$reason\""
  body="$body}"
  # "started" is fire and forget. The others are the last thing this script does, so they
  # wait: a backgrounded child is killed when the shell exits, and the count would be lost.
  if [ "$event" = started ]; then
    curl -fsS -m 3 -X POST -H 'Content-Type: application/json' --data "$body" "$EVENT_URL" \
      >/dev/null 2>&1 &
  else
    curl -fsS -m 3 -X POST -H 'Content-Type: application/json' --data "$body" "$EVENT_URL" \
      >/dev/null 2>&1 || true
  fi
}

say "We collect anonymous data to make AssetFurnace better: which steps you run, whether they work, and what computer you use."
say "Never your pictures, prompts, files or anything private. Turn it off any time on the About page."
say "See exactly what's sent: https://assetfurnace.com/privacy#usage-counts"

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
    die "$1 Nobody to ask here, so stopping. Re-run with --yes to go ahead." other
  fi
  printf '%s [y/N] ' "$1" > /dev/tty
  read -r answer < /dev/tty || answer=""
  case "$answer" in y|Y|yes|YES) return 0 ;; *) die "Stopped. Nothing was changed." user_said_no ;; esac
}

# --- 1. This machine --------------------------------------------------------------------
OS="${I3D_UNAME_S:-$(uname -s)}"      # the I3D_ overrides exist for the test suite
ARCH="${I3D_UNAME_M:-$(uname -m)}"
EVENT_ARCH="$(printf '%s' "$ARCH" | tr '[:upper:]' '[:lower:]')"
GPU="none"
# Named before the refusals below, so a machine we turn away can still say which one it was.
case "$OS" in
  Darwin) EVENT_OS="macos" ;;
  Linux) EVENT_OS="linux" ;;
esac
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
  Darwin/*) die "This Mac has an Intel chip. The lab needs Apple Silicon (M1 or newer)." intel_mac ;;
  *) die "Unsupported machine: $OS/$ARCH. The lab runs on Apple Silicon Macs and x86-64 Linux with an NVIDIA or AMD GPU. On Windows, use install.ps1." unsupported_os ;;
esac
say "Machine: $MACHINE"

install_event started

# The same check Rig makes before it loads its model, made before anything is downloaded: our
# NVIDIA PyTorch carries code for RTX 20-series and newer (sm_75+), and an older card dies on
# its first kernel. 2026-11-13: a Quadro P5000 (6.1) installed two gigabytes to find that out.
if [ "$GPU" = nvidia ]; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  OLD_CARD="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  MAJOR="${CAP%%.*}"; MINOR="${CAP#*.}"
  if [ -n "$MAJOR" ] && [ -n "$MINOR" ] && [ "$MAJOR" -eq "$MAJOR" ] 2>/dev/null \
     && [ "$MINOR" -eq "$MINOR" ] 2>/dev/null && [ "$MAJOR$MINOR" -lt 75 ]; then
    die "Your graphics card${OLD_CARD:+ ($OLD_CARD)} is older than AssetFurnace supports: it needs an NVIDIA RTX 20-series or newer. Nothing was installed." old_nvidia
  fi
fi

# --- 2. Tools ---------------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  if [ "$OS" = Darwin ]; then die "git is missing. Install it with: xcode-select --install" no_git
  else die "git is missing. Install it with your package manager, e.g. sudo apt install git" no_git
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
PY="$DIR/.venv/bin/python"

# uv.lock pins every package to versions tested together. On an AMD card PyPI's Linux
# PyTorch is a CUDA build and brings about 6 GB of NVIDIA wheels with it, none of which
# can load, so that path exports the lock, strips that family, and installs the ROCm
# build instead. image_to_3dlab.host picks the ROCm build that matches the installed
# runtime (tested in tests/test_host.py). The CUDA and Apple paths keep a plain uv sync.
#
# --inexact: add and update the lock's packages, but never delete one. Backends install
# into this same environment (the SF3D setup script pip-installs with this interpreter),
# so an exact sync would remove a working backend on the next update.
#
# This script is served from main but checks out a release, and releases up to 0.5.7
# ship requirements.lock instead of uv.lock. Those install from that file, as they did.
OLD_REQS=""
if [ "$DRY" != 1 ] && [ ! -f "$DIR/uv.lock" ]; then
  OLD_REQS="$DIR/requirements.lock"
  [ -f "$OLD_REQS" ] || OLD_REQS="$DIR/requirements.txt"
fi
if [ "$GPU" = amd ]; then
  run "$UV" venv --quiet --allow-existing --python 3.11 "$DIR/.venv"
  if [ "$DRY" != 1 ]; then
    [ -x "$PY" ] || die "The Python environment in $DIR/.venv is missing, so the AMD packages cannot be set up. Delete $DIR/.venv and run this again."
    # Inside the venv: it belongs to this install, and git already ignores it there.
    ROCM_FILE="$DIR/.venv/requirements.rocm.txt"
    if [ -n "$OLD_REQS" ]; then
      cp "$OLD_REQS" "$ROCM_FILE"
    else
      (cd "$DIR" && "$UV" export --frozen --no-dev --output-file "$ROCM_FILE")
    fi
    # From inside $DIR: `-m` finds the package in the current folder, and under
    # `curl | bash` that folder is wherever the user happened to be.
    (cd "$DIR" && "$PY" -m image_to_3dlab.host without-cuda "$ROCM_FILE") >"$ROCM_FILE.tmp"
    mv "$ROCM_FILE.tmp" "$ROCM_FILE"
    run "$UV" pip install --quiet --python "$PY" -r "$ROCM_FILE"
  fi
else
  if [ "$DRY" = 1 ]; then
    say "would run: uv sync --frozen --no-dev --inexact --python 3.11 (in $DIR)"
  elif [ -n "$OLD_REQS" ]; then
    "$UV" venv --quiet --allow-existing --python 3.11 "$DIR/.venv"
    "$UV" pip install --quiet --python "$PY" -r "$OLD_REQS"
  else
    (cd "$DIR" && "$UV" sync --frozen --no-dev --inexact --python 3.11)
  fi
fi

# After the packages, so it wins over the PyPI wheel: --reinstall-package is what makes
# uv swap the build rather than consider the requirement satisfied.
if [ "$GPU" = amd ] && [ "$DRY" != 1 ] && [ -x "$PY" ]; then
  ROCM_INDEX="$(cd "$DIR" && "$PY" -m image_to_3dlab.host torch-rocm-index | tr -d '[:space:]')"
  if ! (cd "$DIR" && "$PY" -m image_to_3dlab.host torch-has-rocm); then
    if [ -n "$ROCM_INDEX" ]; then
      say "Installing PyTorch with ROCm from $ROCM_INDEX"
      run "$UV" pip install --quiet --python "$PY" --reinstall-package torch \
        --reinstall-package torchvision --index-url "$ROCM_INDEX" torch torchvision
    else
      # The export went in without torch, so without this there would be no PyTorch at all.
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
install_event finished
say "Use it from your agent too (Claude Code shown; Codex, Kimi and others: docs/agents.md):"
printf '\n    claude mcp add assetfurnace -- %s/lab mcp\n\n' "$DIR"
if [ "$START" = 1 ] && [ "$DRY" != 1 ] && [ -x "$DIR/lab" ]; then
  say "Starting the lab. Ctrl-C stops it; start it again any time with: cd $DIR && ./lab"
  exec "$DIR/lab"
fi
say "Start the lab with:"
printf '\n    cd %s && ./lab\n\n' "$DIR"
