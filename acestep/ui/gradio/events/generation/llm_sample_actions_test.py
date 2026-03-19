"""Tests for Simple-mode sample creation with external LM support."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.ui.gradio.events.generation.llm_sample_actions import handle_create_sample


class LlmSampleActionsTests(unittest.TestCase):
    """Verify Simple-mode sampling honors external LM mode."""

    @patch(
        "acestep.ui.gradio.events.generation.llm_sample_actions.create_sample_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_sample_actions.gr.Info")
    def test_handle_create_sample_uses_external_provider_when_active(
        self,
        _info_mock,
        create_external_mock,
    ) -> None:
        """External LM mode should bypass local-LLM initialization checks."""

        create_external_mock.return_value = SimpleNamespace(
            success=True,
            caption="Bright indie pop",
            lyrics="[Verse]",
            bpm=118,
            duration=30,
            keyscale="C Major",
            language="en",
            timesignature="4/4",
            instrumental=False,
            status_message="External sample created",
        )
        llm_handler = SimpleNamespace(llm_initialized=False)

        with patch(
            "acestep.ui.gradio.events.generation.llm_sample_actions.is_external_lm_active",
            return_value=True,
        ):
            result = handle_create_sample(
                llm_handler=llm_handler,
                query="bright indie pop",
                instrumental=False,
                vocal_language="en",
                lm_temperature=0.8,
                lm_top_k=0,
                lm_top_p=0.9,
            )

        create_external_mock.assert_called_once()
        self.assertEqual(result[0], "Bright indie pop")
        self.assertEqual(result[13], "External sample created")


if __name__ == "__main__":
    unittest.main()
