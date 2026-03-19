"""Focused tests for 5Hz LM format debug logging."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.llm_inference import LLMHandler


class LlmInferenceFormatDebugTests(unittest.TestCase):
    """Verify format debug mode logs the prompt and raw response."""

    @patch("acestep.llm_inference.debug_log_for")
    def test_format_sample_from_input_logs_prompt_and_raw_response_in_debug_mode(
        self,
        debug_log_mock,
    ) -> None:
        """Debug mode should emit the exact format prompt and raw response text."""

        handler = LLMHandler()
        handler.llm_initialized = True
        handler.build_formatted_prompt_for_format = lambda **_: "PROMPT BODY"
        handler.generate_from_formatted_prompt = lambda **_: (
            "<think>\ncaption: Tropical funk arrangement\n</think>\n[Instrumental]",
            "ok",
        )
        handler.parse_lm_output = lambda _text: (
            {
                "caption": "Tropical funk arrangement",
                "bpm": 112,
                "keyscale": "C major",
                "language": "unknown",
                "timesignature": "4/4",
            },
            "",
        )
        handler._extract_lyrics_from_output = lambda _text: "[Instrumental]"

        metadata, status = handler.format_sample_from_input(
            caption="Tropical funk",
            lyrics="",
            constrained_decoding_debug=True,
        )

        self.assertEqual(metadata["caption"], "Tropical funk arrangement")
        self.assertIn("Format completed successfully", status)

        logged_text = "\n".join(call.args[1] for call in debug_log_mock.call_args_list)
        self.assertIn("5Hz LM format prompt", logged_text)
        self.assertIn("PROMPT BODY", logged_text)
        self.assertIn("5Hz LM raw format response", logged_text)
        self.assertIn("caption: Tropical funk arrangement", logged_text)


if __name__ == "__main__":
    unittest.main()
