#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Fresh Ubuntu images may ship Python without the venv/ensurepip module.
if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Installing Python venv support..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if ! apt-get install -y "python${PYTHON_VERSION}-venv"; then
      apt-get install -y python3-venv
    fi
  else
    echo "[BOOTSTRAP][FAIL] Python venv support is missing and apt-get is unavailable." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

if ! command -v ollama >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Installing Ollama..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y zstd curl
  fi
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[BOOTSTRAP][FAIL] Ollama installation did not provide the ollama command." >&2
  exit 1
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Starting Ollama..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  ready=0
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[BOOTSTRAP][FAIL] Ollama did not become ready within 60 seconds." >&2
    cat /tmp/ollama.log >&2 || true
    exit 1
  fi
fi

MODEL_NAME="${MODEL_NAME:-qwen3-coder:30b}"
if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL_NAME"; then
  echo "[BOOTSTRAP] Pulling $MODEL_NAME..."
  ollama pull "$MODEL_NAME"
fi

python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
python -m pytest -q

echo ""
echo "[RECOVERY] Checking for an interrupted/running execution..."
agent recover
echo ""
echo "[READY] AI-Agent-T4 is ready."
echo "[READY] Try: agent status"
echo "[READY] Try: agent health"
echo "[READY] Try: agent gpu"
echo "[READY] Try: agent model"
echo "[READY] Try: agent run \"Inspect the repository and run its tests.\""
