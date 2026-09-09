#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/agent ]]; then
  echo "[SMOKE] .venv is missing. Run deployment/bootstrap.sh first."
  exit 1
fi

source .venv/bin/activate

fail() {
  echo "[SMOKE][FAIL] $1" >&2
  exit 1
}

echo "=== Python ==="
python --version

echo "=== NVIDIA T4 ==="
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_JSON="$(agent gpu)" || fail "agent gpu failed"
echo "$GPU_JSON"
echo "$GPU_JSON" | grep -q '"ok": true' || fail "GPU health is not OK"
echo "$GPU_JSON" | grep -qi 'Tesla T4' || fail "Expected a Tesla T4 runtime"

echo "=== Ollama ==="
MODEL_JSON="$(agent model)" || fail "agent model failed"
echo "$MODEL_JSON"
echo "$MODEL_JSON" | grep -q '"ok": true' || fail "Ollama is not reachable"
echo "$MODEL_JSON" | grep -q '"model_present": true' || fail "Configured model is not installed"

echo "=== Package tests ==="
python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
python -m pytest -q

echo "=== Real agent execution ==="
agent run "Create a file named .agent_state/colab_smoke_marker.txt containing exactly SMOKE_OK, then read it back and verify the contents."

test "$(cat .agent_state/colab_smoke_marker.txt)" = "SMOKE_OK" || fail "Agent did not create the expected marker"

echo "[SMOKE][PASS] Colab + T4 + Ollama + agent end-to-end path is working."
