"""Unit tests for LLM token-budget safeguards in ``LLMHandler``."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

try:
    from acestep.llm_inference import (
        COT_PHASE_FALLBACK_MAX_TOKENS,
        LLMHandler,
        UNDERSTAND_PHASE_FALLBACK_MAX_TOKENS,
        UNDERSTAND_PHASE_MAX_TOKENS,
    )
except Exception as exc:  # pragma: no cover - import guard for constrained envs
    COT_PHASE_FALLBACK_MAX_TOKENS = None
    LLMHandler = None
    UNDERSTAND_PHASE_FALLBACK_MAX_TOKENS = None
    UNDERSTAND_PHASE_MAX_TOKENS = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmTokenBudgetTests(unittest.TestCase):
    """Validate max token budgeting for constrained LM phases."""

    def test_resolve_target_duration_prefers_explicit_input(self) -> None:
        """Explicit positive duration should override metadata-derived duration."""
        handler = LLMHandler()

        resolved = handler._resolve_target_duration(24, {"duration": 120})

        self.assertEqual(24.0, resolved)

    def test_resolve_target_duration_falls_back_to_metadata(self) -> None:
        """Metadata duration should be used when explicit duration is absent."""
        handler = LLMHandler()

        resolved = handler._resolve_target_duration(None, {"duration": "32"})

        self.assertEqual(32.0, resolved)

    def test_resolve_target_duration_rejects_non_positive_values(self) -> None:
        """Invalid or non-positive durations should resolve to ``None``."""
        handler = LLMHandler()

        resolved = handler._resolve_target_duration(None, {"duration": "N/A"})
        self.assertIsNone(resolved)
        resolved = handler._resolve_target_duration(0, {"duration": -1})
        self.assertIsNone(resolved)

    def test_cot_phase_without_duration_uses_fallback_cap(self) -> None:
        """CoT phase should not run with a near-4k fallback token budget."""
        handler = LLMHandler()
        handler.max_model_len = 4096

        max_new_tokens = handler._compute_max_new_tokens(
            target_duration=None,
            generation_phase="cot",
            fallback_max=handler.max_model_len - 64,
        )

        self.assertEqual(COT_PHASE_FALLBACK_MAX_TOKENS, max_new_tokens)

    def test_understand_phase_without_duration_uses_fallback_cap(self) -> None:
        """Understand phase should not use a full 4k fallback token budget."""
        handler = LLMHandler()
        handler.max_model_len = 4096

        max_new_tokens = handler._compute_max_new_tokens(
            target_duration=None,
            generation_phase="understand",
            fallback_max=handler.max_model_len - 64,
        )

        self.assertEqual(UNDERSTAND_PHASE_FALLBACK_MAX_TOKENS, max_new_tokens)

    def test_understand_phase_with_duration_is_capped(self) -> None:
        """Duration-driven understand budgets should still respect hard safety cap."""
        handler = LLMHandler()
        handler.max_model_len = 4096

        with patch(
            "acestep.llm_inference.get_global_gpu_config",
            return_value=types.SimpleNamespace(max_duration_with_lm=600),
        ):
            max_new_tokens = handler._compute_max_new_tokens(
                target_duration=600,
                generation_phase="understand",
                fallback_max=handler.max_model_len - 64,
            )

        self.assertEqual(UNDERSTAND_PHASE_MAX_TOKENS, max_new_tokens)

    def test_codes_phase_budget_is_unchanged(self) -> None:
        """Codes phase should keep its original duration + 10 token behavior."""
        handler = LLMHandler()
        handler.max_model_len = 4096

        with patch(
            "acestep.llm_inference.get_global_gpu_config",
            return_value=types.SimpleNamespace(max_duration_with_lm=240),
        ):
            max_new_tokens = handler._compute_max_new_tokens(
                target_duration=30,
                generation_phase="codes",
                fallback_max=handler.max_model_len - 64,
            )

        self.assertEqual(160, max_new_tokens)


if __name__ == "__main__":
    unittest.main()
