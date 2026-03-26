"""
llm/client.py
=============
Thin HTTP wrapper around the Ollama local API.

Usage:
    client = OllamaClient()
    if client.is_available():
        text = client.generate("Write a haiku about food.")
"""

from __future__ import annotations

import os

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout


class OllamaClient:
    """
    Client for a locally-running Ollama instance.

    Parameters
    ----------
    base_url:
        Root URL for Ollama (default: env OLLAMA_BASE_URL or http://localhost:11434).
    model:
        Default model name (default: env OLLAMA_MODEL or "llama3.2").
    timeout:
        Per-request timeout in seconds.
    retry_on_timeout:
        If True, a single retry is attempted when the first call times out.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 30,
        retry_on_timeout: bool = True,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.timeout = timeout
        self.retry_on_timeout = retry_on_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.85,
        stop: list[str] | str | None = None,
    ) -> str:
        """
        Call Ollama's /api/generate endpoint and return the response text.

        Returns an empty string on any unrecoverable error so the caller can
        fall back gracefully without crashing the pipeline.
        """
        payload: dict = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]

        def _call() -> str:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

        try:
            return _call()
        except Timeout:
            if self.retry_on_timeout:
                try:
                    return _call()
                except Exception:
                    return ""
            return ""
        except (HTTPError, ConnectionError, Exception):
            return ""
