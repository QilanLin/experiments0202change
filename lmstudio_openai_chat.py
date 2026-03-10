"""
LM Studio OpenAI-compatible chat wrapper.

This adapter provides the minimal ``invoke()`` surface expected by the
experiment code while delegating generation to a running LM Studio local
server via its OpenAI-compatible API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import json

import requests


@dataclass
class _ChatResponse:
    content: str


class LMStudioOpenAIChat:
    """Minimal chat client compatible with the experiment agent."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 300,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _normalize_messages(self, inp: Any) -> List[Dict[str, str]]:
        if isinstance(inp, str):
            return [{"role": "user", "content": inp}]

        if isinstance(inp, dict) and "messages" in inp:
            inp = inp["messages"]

        if not isinstance(inp, list):
            raise TypeError(f"Unsupported input type for LMStudioOpenAIChat.invoke: {type(inp)}")

        messages: List[Dict[str, str]] = []
        for item in inp:
            if isinstance(item, dict):
                role = str(item.get("role", "user")).lower()
                content = str(item.get("content", ""))
            elif isinstance(item, tuple) and len(item) == 2:
                role = str(item[0]).lower()
                content = str(item[1])
            elif hasattr(item, "type") and hasattr(item, "content"):
                role = str(item.type).lower()
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = str(item.content)
            else:
                raise TypeError(f"Unsupported message item for LMStudioOpenAIChat.invoke: {type(item)}")

            messages.append({"role": role, "content": content})

        return messages

    def _available_models(self) -> List[str]:
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", []) if "id" in item]

    def _resolve_model(self) -> str:
        if self.model_name and self.model_name != "auto":
            return self.model_name

        models = self._available_models()
        if not models:
            raise RuntimeError(
                "LM Studio returned no loaded models. Start the local server and load Qwen 32B first."
            )
        return models[0]

    def invoke(self, inp: Any) -> _ChatResponse:
        messages = self._normalize_messages(inp)
        model = self._resolve_model()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                f"LM Studio chat request failed ({response.status_code}): {detail}"
            ) from exc

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _ChatResponse(content=content)
