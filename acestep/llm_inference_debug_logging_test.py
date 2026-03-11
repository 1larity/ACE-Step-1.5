"""Tests for global LM task prompt/response logging."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from acestep.llm_inference import LLMHandler


class LlmInferenceDebugLoggingTests(unittest.TestCase):
    """Validate prompt/response logging for local 5Hz LM tasks."""

    @patch("acestep.llm_inference.logger.debug")
    @patch("acestep.llm_inference.is_lm_task_debug_enabled", return_value=True)
    def test_unload_logs_safe_debug_message_without_missing_locals(
        self,
        _debug_enabled_mock,
        logger_debug_mock,
    ) -> None:
        """Unload should not reference generation-only locals when LM debug is enabled."""
        handler = LLMHandler.__new__(LLMHandler)
        handler.llm_backend = "pt"
        handler.llm_initialized = True
        handler.llm = object()
        handler.llm_tokenizer = object()
        handler.constrained_processor = None
        handler._hf_model_for_scoring = None
        handler._mlx_model = None
        handler._mlx_model_path = None

        with patch("acestep.llm_inference.torch.cuda.is_available", return_value=False),                 patch("acestep.llm_inference.torch.backends.mps.is_available", return_value=False, create=True),                 patch.object(LLMHandler, "_cleanup_torch_distributed_state"):
            LLMHandler.unload(handler)

        logger_debug_mock.assert_any_call(
            "LM unload backend={} initialized={}",
            "pt",
            True,
        )

    @patch("acestep.llm_inference.logger.debug")
    @patch("acestep.llm_inference.is_lm_task_debug_enabled", return_value=True)
    def test_generate_from_formatted_prompt_logs_prompt_and_output(
        self,
        _debug_enabled_mock,
        logger_debug_mock,
    ) -> None:
        """Global LM debug should log both the formatted prompt and raw output text."""
        handler = LLMHandler.__new__(LLMHandler)
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.llm = object()
        handler.llm_tokenizer = object()
        handler._run_pt = MagicMock(return_value="<think>\nbpm: 120\n</think>\n[Verse 1]\nhello")

        output_text, status = LLMHandler.generate_from_formatted_prompt(
            handler,
            formatted_prompt="# Instruction\nTest prompt",
            cfg={"generation_phase": "understand"},
        )

        self.assertIn("Generated successfully", status)
        self.assertIn("[Verse 1]", output_text)
        messages = [call.args[0] for call in logger_debug_mock.call_args_list]
        self.assertTrue(any("LM task prompt backend=" in message for message in messages))
        self.assertTrue(any("LM task output backend=" in message for message in messages))
