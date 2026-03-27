"""Routing tests for external LM format flow."""

from __future__ import annotations

import unittest
from types import MethodType
from unittest.mock import patch

from acestep.llm_inference import LLMHandler


class ExternalFormatRoutingTests(unittest.TestCase):
    """Verify external backend routes through the external caption-enhancement path."""

    def test_format_sample_from_input_uses_external_path_without_local_init(self) -> None:
        """External format should not require local 5Hz initialization."""

        handler = LLMHandler.__new__(LLMHandler)
        handler.llm_backend = "external"
        handler.llm_initialized = False

        def fake_external(self, caption, lyrics, user_metadata=None):
            return ({"caption": f"external::{caption}", "lyrics": lyrics}, "external-ok")

        handler._format_sample_from_external_llm = MethodType(fake_external, handler)

        metadata, status = LLMHandler.format_sample_from_input(
            handler,
            caption="Dreamy synth-pop",
            lyrics="[Instrumental]",
        )

        self.assertEqual(status, "external-ok")
        self.assertEqual(metadata["caption"], "external::Dreamy synth-pop")

    def test_format_sample_from_input_keeps_local_guard_for_non_external_backend(self) -> None:
        """Non-external backends should still require local 5Hz initialization."""

        handler = LLMHandler.__new__(LLMHandler)
        handler.llm_backend = "vllm"
        handler.llm_initialized = False

        def fail_if_called(*args, **kwargs):
            raise AssertionError("external format path should not run for non-external backends")

        handler._format_sample_from_external_llm = fail_if_called

        metadata, status = LLMHandler.format_sample_from_input(
            handler,
            caption="Dreamy synth-pop",
            lyrics="[Instrumental]",
        )

        self.assertEqual(metadata, {})
        self.assertEqual(status, "❌ 5Hz LM not initialized. Please initialize it first.")

    @patch("acestep.llm_inference.request_external_format_plan")
    def test_external_format_uses_session_key_not_shared_runtime_config(self, request_plan_mock) -> None:
        """External format should resolve an inline session key without storing it in external_config."""

        handler = LLMHandler.__new__(LLMHandler)
        handler.llm_backend = "external"
        handler.llm_initialized = False
        handler.external_config = {
            "provider": "openai",
            "protocol": "openai_chat",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
        }
        handler._external_session_api_key = "sk-session"
        request_plan_mock.return_value = type(
            "Plan",
            (),
            {
                "to_dict": lambda self: {"caption": "expanded"},
                "vocal_language": "en",
                "key_scale": "C Minor",
                "time_signature": "4/4",
            },
        )()

        metadata, status = LLMHandler._format_sample_from_external_llm(
            handler,
            caption="Dreamy synth-pop",
            lyrics="[Instrumental]",
        )

        self.assertEqual(metadata["caption"], "expanded")
        self.assertIn("external", status.lower())
        self.assertNotIn("api_key", handler.external_config)
        self.assertEqual(request_plan_mock.call_args.kwargs["api_key"], "sk-session")


if __name__ == "__main__":
    unittest.main()
