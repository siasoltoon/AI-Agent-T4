from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(float(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "qwen3-coder:30b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    workspace_root: Path = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()
    state_dir: Path = Path(os.getenv("STATE_DIR", ".agent_state")).resolve()
    max_agent_steps: int = _int("MAX_AGENT_STEPS", 64, 1, 128)
    max_recovery_attempts: int = _int("MAX_RECOVERY_ATTEMPTS", 8, 0, 12)
    max_command_seconds: int = _int("MAX_COMMAND_SECONDS", 600, 1, 600)
    model_timeout_seconds: int = _int("MODEL_TIMEOUT_SECONDS", 900, 10, 3600)
    model_temperature: float = _float("MODEL_TEMPERATURE", 0.1, 0.0, 1.0)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 48000, 4000, 120000)
    max_output_chars: int = _int("MAX_TOOL_OUTPUT_CHARS", 16000, 1000, 50000)


SETTINGS = Settings()
SETTINGS.state_dir.mkdir(parents=True, exist_ok=True)
