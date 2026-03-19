"""Focused tests for LM format-action debug defaults."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.generation.llm_format_action_helpers import (
    _effective_llm_debug_enabled,
)


class LlmFormatActionsTests(unittest.TestCase):
    """Verify format-action debug mode respects global defaults."""

    @patch("acestep.ui.gradio.events.generation.llm_format_action_helpers.DEBUG_LLM", "ON")
    def test_effective_llm_debug_enabled_uses_global_debug_default(self) -> None:
        """Global LLM debug should enable terminal prompt logging even if UI flag is false."""

        self.assertTrue(_effective_llm_debug_enabled(False))

    @patch("acestep.ui.gradio.events.generation.llm_format_action_helpers.DEBUG_LLM", "OFF")
    def test_effective_llm_debug_enabled_honors_explicit_ui_debug_flag(self) -> None:
        """Explicit UI debug should still enable logging even if the global default is off."""

        self.assertTrue(_effective_llm_debug_enabled(True))
        self.assertFalse(_effective_llm_debug_enabled(False))


if __name__ == "__main__":
    unittest.main()
