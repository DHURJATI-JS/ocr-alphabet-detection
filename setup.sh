#!/usr/bin/env bash
# One-time setup on macOS / Linux: virtualenv, dependencies, Tesseract model.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/download_models.py

echo
echo "Setup complete. Start the app with ./run.sh and open http://localhost:8000"
