#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF3D_DIR="${PROJECT_DIR}/vendor/stable-fast-3d"
IMAGE3D_PYTHON="${IMAGE3D_PYTHON:-}"

if [[ -z "${IMAGE3D_PYTHON}" ]]; then
  for candidate in python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      IMAGE3D_PYTHON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${IMAGE3D_PYTHON}" ]]; then
  echo "Python 3.10 or 3.11 is required. Install one with: brew install python@3.11" >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required to install libomp: https://brew.sh" >&2
  exit 1
fi

brew install libomp
"${IMAGE3D_PYTHON}" -m venv "${PROJECT_DIR}/.venv"
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

if [[ ! -d "${SF3D_DIR}/.git" ]]; then
  mkdir -p "${PROJECT_DIR}/vendor"
  git clone https://github.com/Stability-AI/stable-fast-3d.git "${SF3D_DIR}"
fi

# SF3D's requirements install its Metal texture baker and macOS UV unwrapper
# from the checked-out local directories.
pushd "${SF3D_DIR}" >/dev/null
USE_CUDA=0 USE_METAL=1 python -m pip install --no-build-isolation -r requirements.txt
popd >/dev/null

echo "Bootstrap complete. Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
