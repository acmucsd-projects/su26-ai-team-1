#!/bin/bash
# Launch the recogniser using the project's virtualenv, so torch and the rest
# resolve without activating anything first.
cd "$(dirname "$0")"
# The venv lives at the repo root, which is either our parent (app inside the
# repo) or a sibling directory (app beside it).
for VENV in ../.venv ../su26-ai-team-1/.venv; do
  [ -x "$VENV/bin/python3" ] && exec "$VENV/bin/python3" app.py "$@"
done
echo "no virtualenv found; falling back to python3 on PATH" >&2
exec python3 app.py "$@"
