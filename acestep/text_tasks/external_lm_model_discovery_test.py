"""Tests for external LM model discovery helpers."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_model_discovery import (
    _build_model_list_urls,
    discover_external_models,
)


class _FakeResponse:
    """Minimal context-manager HTTP response stub."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        """Return encoded payload bytes."""

        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        """Return self for context-manager use."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Do not suppress exceptions."""

        return False


class ExternalLmModelDiscoveryTests(unittest.TestCase):
    """Verify provider-specific discovery URL building and parsing."""

    def test_build_model_list_urls_supports_ollama_root_endpoint(self) -> None:
        """Ollama root URLs should expand to both tags and models endpoints."""

        self.assertEqual(
            _build_model_list_urls(
                provider="ollama",
                protocol="openai_chat",
                base_url="http://127.0.0.1:11434",
            ),
            [
                "http://127.0.0.1:11434/api/tags",
                "http://127.0.0.1:11434/v1/models",
            ],
        )

    @patch("acestep.text_tasks.external_lm_model_discovery.request.urlopen")
    def test_discover_external_models_reads_ollama_tags_payload(self, urlopen_mock) -> None:
        """Ollama `/api/tags` responses should yield model names."""

        urlopen_mock.return_value = _FakeResponse(
            {
                "models": [
                    {"name": "llama3.1:8b-instruct"},
                    {"name": "codellama:13b"},
                ]
            }
        )

        models = discover_external_models(
            provider="ollama",
            protocol="openai_chat",
            base_url="http://127.0.0.1:11434",
            api_key="",
        )

        self.assertEqual(models, ["llama3.1:8b-instruct", "codellama:13b"])


if __name__ == "__main__":
    unittest.main()
