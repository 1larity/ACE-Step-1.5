"""Tests for external LM plan request orchestration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_plan_requests import request_external_plan


class ExternalLmPlanRequestsTests(unittest.TestCase):
    """Verify provider-specific request routing and parsing."""

    @patch("acestep.text_tasks.external_lm_plan_requests.request_ollama_plan")
    @patch("acestep.text_tasks.external_lm_plan_requests.external_base_url")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_protocol")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_model")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_provider")
    def test_request_external_plan_delegates_to_ollama_helper(
        self,
        provider_mock,
        model_mock,
        protocol_mock,
        resolve_key_mock,
        base_url_mock,
        request_ollama_mock,
    ) -> None:
        """Ollama provider mode should route through the native helper."""

        provider_mock.return_value = "ollama"
        model_mock.return_value = "qwen3:4b"
        protocol_mock.return_value = "openai_chat"
        resolve_key_mock.return_value = ""
        base_url_mock.return_value = "http://127.0.0.1:11434/v1/chat/completions"
        request_ollama_mock.return_value = "parsed-plan"

        plan, profile, model = request_external_plan(
            intent="synthwave instrumental",
            timeout_sec=30,
            task_focus="format",
        )

        self.assertEqual("parsed-plan", plan)
        self.assertEqual("ollama", profile.provider_id)
        self.assertEqual("qwen3:4b", model)

    @patch("acestep.text_tasks.external_lm_plan_requests.post_json")
    @patch("acestep.text_tasks.external_lm_plan_requests.external_base_url")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_protocol")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_model")
    @patch("acestep.text_tasks.external_lm_plan_requests.get_active_external_lm_provider")
    def test_request_external_plan_parses_non_ollama_response(
        self,
        provider_mock,
        model_mock,
        protocol_mock,
        resolve_key_mock,
        base_url_mock,
        post_json_mock,
    ) -> None:
        """Non-Ollama providers should parse structured chat-completions responses."""

        provider_mock.return_value = "openai"
        model_mock.return_value = "gpt-4o-mini"
        protocol_mock.return_value = "openai_chat"
        resolve_key_mock.return_value = "test-key"
        base_url_mock.return_value = "https://api.openai.com/v1/chat/completions"
        post_json_mock.return_value = (
            '{"choices": [{"message": {"content": '
            '"{\\"caption\\": \\"Dream pop anthem\\", \\"lyrics\\": \\"Verse 1\\", '
            '\\"bpm\\": 120, \\"duration\\": 96, \\"key_scale\\": \\"E major\\", '
            '\\"time_signature\\": \\"4/4\\", \\"vocal_language\\": \\"english\\", '
            '\\"instrumental\\": false}"}}]}'
        )

        plan, profile, model = request_external_plan(
            intent="dream pop with bright vocals",
            timeout_sec=30,
            task_focus="format",
        )

        self.assertEqual("Dream pop anthem", plan.caption)
        self.assertEqual("openai", profile.provider_id)
        self.assertEqual("gpt-4o-mini", model)


if __name__ == "__main__":
    unittest.main()
