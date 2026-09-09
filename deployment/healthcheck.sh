#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "=== Python ==="
python --version

echo "=== GPU ==="
agent gpu || true
echo "=== Model runtime ==="
agent model || true
echo "=== Package ==="
python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
echo "HEALTHCHECK_OK"
