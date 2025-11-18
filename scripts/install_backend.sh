#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

VENV_DIR="${PROJECT_ROOT}/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[backend] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[backend] Dependencies installed. Activate the env with: source .venv/bin/activate"
