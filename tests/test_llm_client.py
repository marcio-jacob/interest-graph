"""
Unit tests for llm/client.py
==============================
All HTTP calls are mocked — no live Ollama instance required.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm.client import OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code=200, json_data=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# is_available
# ===========================================================================

class TestIsAvailable:
    def test_returns_true_on_200(self):
        with patch("requests.get", return_value=_make_response(200)):
            assert OllamaClient().is_available() is True

    def test_returns_false_on_non_200(self):
        with patch("requests.get", return_value=_make_response(503)):
            assert OllamaClient().is_available() is False

    def test_returns_false_on_connection_error(self):
        with patch("requests.get", side_effect=requests.ConnectionError):
            assert OllamaClient().is_available() is False

    def test_returns_false_on_timeout(self):
        with patch("requests.get", side_effect=requests.Timeout):
            assert OllamaClient().is_available() is False

    def test_pings_correct_endpoint(self):
        with patch("requests.get", return_value=_make_response(200)) as mock_get:
            OllamaClient(base_url="http://localhost:11434").is_available()
            called_url = mock_get.call_args[0][0]
            assert called_url.endswith("/api/tags")


# ===========================================================================
# generate
# ===========================================================================

class TestGenerate:
    def test_returns_response_text(self):
        resp = _make_response(200, {"response": "hello world"})
        with patch("requests.post", return_value=resp):
            result = OllamaClient().generate("test prompt")
        assert result == "hello world"

    def test_strips_whitespace(self):
        resp = _make_response(200, {"response": "  trimmed  "})
        with patch("requests.post", return_value=resp):
            result = OllamaClient().generate("prompt")
        assert result == "trimmed"

    def test_empty_response_key_returns_empty_string(self):
        resp = _make_response(200, {})
        with patch("requests.post", return_value=resp):
            result = OllamaClient().generate("prompt")
        assert result == ""

    def test_http_error_returns_empty_string(self):
        with patch("requests.post", return_value=_make_response(500)):
            result = OllamaClient().generate("prompt")
        assert result == ""

    def test_connection_error_returns_empty_string(self):
        with patch("requests.post", side_effect=requests.ConnectionError):
            result = OllamaClient().generate("prompt")
        assert result == ""

    def test_retries_once_on_timeout(self):
        ok_resp = _make_response(200, {"response": "retried"})
        with patch(
            "requests.post",
            side_effect=[requests.Timeout, ok_resp],
        ):
            result = OllamaClient(retry_on_timeout=True).generate("prompt")
        assert result == "retried"

    def test_no_retry_when_disabled(self):
        with patch("requests.post", side_effect=requests.Timeout):
            result = OllamaClient(retry_on_timeout=False).generate("prompt")
        assert result == ""

    def test_returns_empty_after_two_timeouts(self):
        with patch(
            "requests.post",
            side_effect=[requests.Timeout, requests.Timeout],
        ):
            result = OllamaClient(retry_on_timeout=True).generate("prompt")
        assert result == ""

    def test_posts_to_generate_endpoint(self):
        resp = _make_response(200, {"response": "ok"})
        with patch("requests.post", return_value=resp) as mock_post:
            OllamaClient(base_url="http://localhost:11434").generate("p")
            called_url = mock_post.call_args[0][0]
            assert called_url.endswith("/api/generate")

    def test_model_override_in_call(self):
        resp = _make_response(200, {"response": "ok"})
        with patch("requests.post", return_value=resp) as mock_post:
            OllamaClient().generate("p", model="llama3.1")
            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "llama3.1"

    def test_stop_list_included_in_options(self):
        resp = _make_response(200, {"response": "ok"})
        with patch("requests.post", return_value=resp) as mock_post:
            OllamaClient().generate("p", stop=["\\n", "END"])
            options = mock_post.call_args[1]["json"]["options"]
            assert options["stop"] == ["\\n", "END"]

    def test_stop_string_wrapped_in_list(self):
        resp = _make_response(200, {"response": "ok"})
        with patch("requests.post", return_value=resp) as mock_post:
            OllamaClient().generate("p", stop="END")
            options = mock_post.call_args[1]["json"]["options"]
            assert options["stop"] == ["END"]


# ===========================================================================
# Configuration
# ===========================================================================

class TestConfiguration:
    def test_default_model_is_llama32(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        client = OllamaClient()
        assert client.model == "llama3.2"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        client = OllamaClient()
        assert client.model == "mistral"

    def test_model_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        client = OllamaClient(model="phi3")
        assert client.model == "phi3"

    def test_base_url_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert not client.base_url.endswith("/")

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-server:11434")
        client = OllamaClient()
        assert "gpu-server" in client.base_url
