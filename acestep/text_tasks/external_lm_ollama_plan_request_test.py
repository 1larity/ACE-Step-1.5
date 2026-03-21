"""Tests for Ollama-native external LM plan requests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.text_tasks.external_ai_types import ExternalAIClientError
from acestep.text_tasks.external_lm_ollama_plan_request import request_ollama_plan


class ExternalLmOllamaPlanRequestTests(unittest.TestCase):
    """Verify Ollama-native planning requests parse and fail safely."""

    @patch("acestep.text_tasks.external_lm_ollama_plan_request.post_json")
    def test_request_ollama_plan_parses_message_content(self, post_json_mock) -> None:
        """Ollama `/api/chat` wrappers should parse into an external plan."""

        post_json_mock.return_value = (
            '{"message": {"content": '
            '"{\\"caption\\": \\"Warm synthwave track\\", \\"lyrics\\": \\"\\", '
            '\\"bpm\\": 118, \\"duration\\": 95, \\"key_scale\\": \\"C minor\\", '
            '\\"time_signature\\": \\"4/4\\", \\"vocal_language\\": \\"english\\", '
            '\\"instrumental\\": true}"}}'
        )

        plan = request_ollama_plan(
            model="qwen3:4b",
            base_url="http://127.0.0.1:11434/v1/chat/completions",
            messages=[{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "x"}],
            timeout_sec=30,
            task_focus="format",
        )

        self.assertEqual("Warm synthwave track", plan.caption)
        self.assertTrue(plan.instrumental)

    @patch("acestep.text_tasks.external_lm_ollama_plan_request.post_json")
    def test_request_ollama_plan_rejects_missing_content(self, post_json_mock) -> None:
        """Missing final chat content should raise a user-facing client error."""

        post_json_mock.return_value = '{"message": {"content": ""}}'

        with self.assertRaisesRegex(ExternalAIClientError, "no final content"):
            request_ollama_plan(
                model="qwen3:4b",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                messages=[{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "x"}],
                timeout_sec=30,
                task_focus="format",
            )


if __name__ == "__main__":
    unittest.main()
