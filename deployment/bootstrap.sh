#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

if ! command -v ollama >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Starting Ollama..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

MODEL_NAME="${MODEL_NAME:-qwen2.5-coder:7b}"
if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL_NAME"; then
  echo "[BOOTSTRAP] Pulling $MODEL_NAME..."
  ollama pull "$MODEL_NAME"
fi

python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
python -m pytest -q

echo ""
echo "[READY] AI-Agent-T4 is ready."
echo "[READY] Try: agent status"
echo "[READY] Try: agent gpu"
echo "[READY] Try: agent model"
echo "[READY] Try: agent run \"Inspect the repository and run its tests.\""
