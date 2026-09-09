from __future__ import annotations

from typing import Any

import httpx


class ModelRuntimeError(RuntimeError):
    pass


class OllamaRuntime:
    def __init__(self, host: str, model: str, timeout: int = 180) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {"ok": True, "host": self.host, "model": self.model, "models": models, "model_present": self.model in models}
        except Exception as exc:
            return {"ok": False, "host": self.host, "model": self.model, "error": str(exc)}

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.host}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
            text = str(data.get("response", "")).strip()
            if not text:
                raise ModelRuntimeError("Ollama returned an empty response.")
            return text
        except Exception as exc:
            if isinstance(exc, ModelRuntimeError):
                raise
            raise ModelRuntimeError(f"Model runtime request failed: {exc}") from exc
