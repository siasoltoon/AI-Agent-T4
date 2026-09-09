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

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, f"{self.host}{path}", **kwargs)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            raise ModelRuntimeError(f"Ollama request failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        try:
            data = self._request("GET", "/api/tags")
            models = [m.get("name") for m in data.get("models", [])]
            return {"ok": True, "host": self.host, "model": self.model, "models": models, "model_present": self.model in models}
        except Exception as exc:
            return {"ok": False, "host": self.host, "model": self.model, "error": str(exc)}

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 0.1) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        return self._request("POST", "/api/chat", json=payload)

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.1) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self.chat(messages, temperature=temperature)
        text = str(data.get("message", {}).get("content", "")).strip()
        if not text:
            raise ModelRuntimeError("Ollama returned an empty response.")
        return text
