#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required (install Node.js LTS)" >&2
  exit 1
fi

cd "$FRONTEND_DIR"

echo "[frontend] Installing npm dependencies..."
npm install

echo "[frontend] Done. Start the dev server with: npm run dev"
