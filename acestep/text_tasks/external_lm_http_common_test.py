"""Tests for external LM HTTP helpers."""

from __future__ import annotations

import json
import os
import unittest
from io import BytesIO
from urllib import error
from unittest.mock import patch

from acestep.text_tasks.external_ai_types import ExternalAIClientError
from acestep.text_tasks.external_lm_http_common import (
    coerce_keep_alive_value,
    external_base_url,
    ollama_model_is_loaded,
    ollama_native_api_url,
    post_json,
)


class _FakeResponse:
    """Minimal context-manager HTTP response stub."""

    def __init__(self, payload: dict[str, object]):
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


class ExternalLmHttpCommonTests(unittest.TestCase):
    """Verify external provider HTTP helpers."""

    def test_coerce_keep_alive_value_normalizes_blank_and_numeric(self) -> None:
        """Blank values should disable keep-alive and numeric strings stay numeric."""

        self.assertEqual(-1, coerce_keep_alive_value(""))
        self.assertEqual(30, coerce_keep_alive_value("30"))
        self.assertEqual("5m", coerce_keep_alive_value("5m"))

    def test_external_base_url_prefers_generic_override(self) -> None:
        """Generic external base URL should win over provider defaults."""

        with patch.dict(os.environ, {"ACESTEP_EXTERNAL_BASE_URL": "https://example.test/chat"}):
            self.assertEqual("https://example.test/chat", external_base_url("openai"))

    def test_ollama_native_api_url_rewrites_to_native_path(self) -> None:
        """Ollama native URL helper should preserve host while swapping path."""

        self.assertEqual(
            "http://127.0.0.1:11434/api/chat",
            ollama_native_api_url(
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                path="/api/chat",
            ),
        )

    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    def test_ollama_model_is_loaded_matches_active_name(self, urlopen_mock) -> None:
        """Loaded-model check should match names returned by Ollama `/api/ps`."""

        urlopen_mock.return_value = _FakeResponse({"models": [{"name": "qwen3:4b"}]})

        self.assertTrue(
            ollama_model_is_loaded(
                model="qwen3:4b",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
            )
        )

    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    def test_post_json_surfaces_http_guidance(self, urlopen_mock) -> None:
        """HTTP failures should include guidance from the supplied formatter."""

        urlopen_mock.side_effect = error.HTTPError(
            url="https://example.test/chat",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b'{"error": {"message": "quota"}}'),
        )

        with self.assertRaisesRegex(ExternalAIClientError, "HTTP 403"):
            post_json(
                url="https://example.test/chat",
                payload={"hello": "world"},
                headers={"Content-Type": "application/json"},
                timeout_sec=5,
                model="demo-model",
                provider_base_url="https://example.test/chat",
                build_http_error_guidance_fn=lambda **_: " | guidance",
            )


if __name__ == "__main__":
    unittest.main()
