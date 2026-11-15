#!/usr/bin/env bash
# Start the server. Uses port 8000, or the next free one if 8000 is taken.
# Settings come from the environment or ./.env (see .env.example). Extra args: --port, --reload
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv yet - run ./setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python -m app "$@"
