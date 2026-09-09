from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    workspace_root: Path = Path(os.getenv("WORKSPACE_ROOT", ".")).resolve()
    state_dir: Path = Path(os.getenv("STATE_DIR", ".agent_state")).resolve()
    max_agent_steps: int = max(1, min(_int("MAX_AGENT_STEPS", 32), 64))
    max_recovery_attempts: int = max(0, min(_int("MAX_RECOVERY_ATTEMPTS", 6), 6))
    max_command_seconds: int = max(1, min(_int("MAX_COMMAND_SECONDS", 600), 600))
    model_timeout_seconds: int = max(10, _int("MODEL_TIMEOUT_SECONDS", 180))
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    max_context_chars: int = max(4000, _int("MAX_CONTEXT_CHARS", 24000))


SETTINGS = Settings()
SETTINGS.state_dir.mkdir(parents=True, exist_ok=True)
