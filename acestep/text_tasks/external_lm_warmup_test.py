"""Tests for external LM warm-up helpers."""

from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_warmup import warm_up_external_provider


class _FakeResponse:
    """Minimal context-manager HTTP response stub."""

    def read(self) -> bytes:
        """Return empty bytes to satisfy the warm-up read."""

        return b"{}"

    def __enter__(self):
        """Return self for context-manager use."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Do not suppress exceptions."""

        return False


class ExternalLmWarmupTests(unittest.TestCase):
    """Verify provider warm-up behavior stays narrow and deterministic."""

    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_provider")
    def test_warm_up_external_provider_skips_non_ollama(self, provider_mock) -> None:
        """Non-Ollama providers should not trigger warm-up requests."""

        provider_mock.return_value = "openai"

        self.assertIsNone(warm_up_external_provider())

    @patch("acestep.text_tasks.external_lm_warmup.external_base_url")
    @patch("acestep.text_tasks.external_lm_warmup.ollama_model_is_loaded")
    @patch("acestep.text_tasks.external_lm_warmup.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_model")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_provider")
    def test_warm_up_external_provider_short_circuits_loaded_model(
        self,
        provider_mock,
        model_mock,
        api_key_mock,
        loaded_mock,
        base_url_mock,
    ) -> None:
        """Already-loaded Ollama models should return a status without POSTing."""

        provider_mock.return_value = "ollama"
        model_mock.return_value = "qwen3:4b"
        api_key_mock.return_value = ""
        loaded_mock.return_value = True
        base_url_mock.return_value = "http://127.0.0.1:11434/v1/chat/completions"

        self.assertEqual(
            "Ollama model already loaded (qwen3:4b)",
            warm_up_external_provider(),
        )

    @patch("acestep.text_tasks.external_lm_warmup.request.urlopen")
    @patch("acestep.text_tasks.external_lm_warmup.external_base_url")
    @patch("acestep.text_tasks.external_lm_warmup.ollama_model_is_loaded")
    @patch("acestep.text_tasks.external_lm_warmup.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_model")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_provider")
    def test_warm_up_external_provider_treats_timeout_as_loaded_if_model_appears(
        self,
        provider_mock,
        model_mock,
        api_key_mock,
        loaded_mock,
        base_url_mock,
        urlopen_mock,
    ) -> None:
        """A timeout should still succeed if Ollama reports the model as loaded afterward."""

        provider_mock.return_value = "ollama"
        model_mock.return_value = "qwen3:4b"
        api_key_mock.return_value = ""
        loaded_mock.side_effect = [False, True]
        base_url_mock.return_value = "http://127.0.0.1:11434/v1/chat/completions"
        urlopen_mock.side_effect = socket.timeout()

        self.assertEqual(
            "Ollama model finished loading after timeout window (qwen3:4b)",
            warm_up_external_provider(timeout_sec=2),
        )

    @patch("acestep.text_tasks.external_lm_warmup.request.urlopen")
    @patch("acestep.text_tasks.external_lm_warmup.external_base_url")
    @patch("acestep.text_tasks.external_lm_warmup.ollama_model_is_loaded")
    @patch("acestep.text_tasks.external_lm_warmup.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_model")
    @patch("acestep.text_tasks.external_lm_warmup.get_active_external_lm_provider")
    def test_warm_up_external_provider_returns_completion_message(
        self,
        provider_mock,
        model_mock,
        api_key_mock,
        loaded_mock,
        base_url_mock,
        urlopen_mock,
    ) -> None:
        """Successful warm-up should return a clear completion status."""

        provider_mock.return_value = "ollama"
        model_mock.return_value = "qwen3:4b"
        api_key_mock.return_value = ""
        loaded_mock.return_value = False
        base_url_mock.return_value = "http://127.0.0.1:11434/v1/chat/completions"
        urlopen_mock.return_value = _FakeResponse()

        self.assertEqual(
            "Ollama warm-up complete (qwen3:4b)",
            warm_up_external_provider(timeout_sec=2),
        )


if __name__ == "__main__":
    unittest.main()
