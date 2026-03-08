"""Tests for external provider model discovery helpers."""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from acestep.text_tasks.external_lm_model_discovery import (
    ExternalModelDiscoveryError,
    discover_external_models,
)


class _FakeHttpResponse:
    """Minimal response object for mocked urllib GET requests."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        """Return prepared response bytes."""
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit context manager without suppressing errors."""
        return False


class ExternalLmModelDiscoveryTests(unittest.TestCase):
    """Validate endpoint fallback and response parsing for model discovery."""

    @patch("acestep.text_tasks.external_lm_model_discovery.request.urlopen")
    def test_openai_style_models_endpoint_is_derived_and_parsed(self, urlopen_mock) -> None:
        """OpenAI-like /chat/completions URL should map to /models and parse ids."""
        urlopen_mock.return_value = _FakeHttpResponse(
            json.dumps({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4.1-mini"}]})
        )

        models = discover_external_models(
            provider="openai",
            protocol="openai_chat",
            base_url="https://api.openai.com/v1/chat/completions",
            api_key="sk-test",
        )

        self.assertEqual(["gpt-4o-mini", "gpt-4.1-mini"], models)
        req = urlopen_mock.call_args.args[0]
        self.assertEqual("https://api.openai.com/v1/models", req.full_url)

    @patch("acestep.text_tasks.external_lm_model_discovery.request.urlopen")
    def test_ollama_falls_back_to_api_tags_when_v1_models_fails(self, urlopen_mock) -> None:
        """Ollama should retry /api/tags when /v1/models is unavailable."""

        def _side_effect(req, timeout=20):
            if req.full_url.endswith("/v1/models"):
                raise HTTPError(req.full_url, 404, "Not Found", hdrs=None, fp=None)
            return _FakeHttpResponse(
                json.dumps({"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]})
            )

        urlopen_mock.side_effect = _side_effect

        models = discover_external_models(
            provider="ollama",
            protocol="openai_chat",
            base_url="http://127.0.0.1:11434/v1/chat/completions",
            api_key="",
        )

        self.assertEqual(["llama3.1:8b", "qwen2.5:7b"], models)

    @patch("acestep.text_tasks.external_lm_model_discovery.request.urlopen")
    def test_raises_error_when_all_candidates_fail(self, urlopen_mock) -> None:
        """Discovery should raise clear error after all endpoint attempts fail."""
        urlopen_mock.side_effect = HTTPError(
            url="https://example.invalid/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

        with self.assertRaises(ExternalModelDiscoveryError):
            discover_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://example.invalid/v1/chat/completions",
                api_key="bad",
            )


if __name__ == "__main__":
    unittest.main()
