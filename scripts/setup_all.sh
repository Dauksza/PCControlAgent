#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/install_backend.sh"
"${SCRIPT_DIR}/install_frontend.sh"

echo "All dependencies installed. Backend: source .venv/bin/activate && uvicorn app:app. Frontend: cd frontend && npm run dev."
