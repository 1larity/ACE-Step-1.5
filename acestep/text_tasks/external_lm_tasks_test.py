"""Tests for external LM task adapter error handling behavior."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_tasks import (
    ExternalAIClientError,
    create_sample_with_external_provider,
    format_sample_with_external_provider,
    generate_lyrics_from_caption_with_external_provider,
)
from acestep.text_tasks.secure_secret_store import SecretStoreError


class ExternalLmTasksErrorTests(unittest.TestCase):
    """Verify secret-store failures are surfaced as External AI client errors."""

    @patch(
        "acestep.text_tasks.external_lm_tasks.resolve_external_api_key_for_runtime",
        side_effect=SecretStoreError("Missing credentials"),
    )
    def test_create_sample_wraps_secret_store_error(self, _resolve_key_mock) -> None:
        """Create-sample adapter should wrap secret-store errors as ExternalAIClientError."""
        with self.assertRaisesRegex(ExternalAIClientError, "Missing credentials"):
            create_sample_with_external_provider(
                query="test",
                instrumental=False,
                vocal_language="en",
            )

    @patch(
        "acestep.text_tasks.external_lm_tasks.resolve_external_api_key_for_runtime",
        side_effect=SecretStoreError("Missing credentials"),
    )
    def test_format_sample_wraps_secret_store_error(self, _resolve_key_mock) -> None:
        """Format adapter should wrap secret-store errors as ExternalAIClientError."""
        with self.assertRaisesRegex(ExternalAIClientError, "Missing credentials"):
            format_sample_with_external_provider(
                caption="caption",
                lyrics="lyrics",
                user_metadata={},
            )

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_generate_lyrics_from_caption_uses_lyrics_task_focus(
        self,
        request_plan_mock,
    ) -> None:
        """Dedicated lyrics adapter should use lyrics-focused external planning."""
        request_plan_mock.return_value = (
            SimpleNamespace(
                caption="arcade rush",
                lyrics="[Verse 1]\nPixel hearts in the midnight glow",
                bpm=142,
                duration=26.0,
                key_scale="E minor",
                vocal_language="en",
                time_signature="4/4",
            ),
            SimpleNamespace(label="Z.ai"),
            "glm-5",
        )

        result = generate_lyrics_from_caption_with_external_provider(
            caption="energetic chiptune track",
            bpm=142,
            audio_duration=26.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
        )

        self.assertEqual("[Verse 1]\nPixel hearts in the midnight glow", result.lyrics)
        self.assertEqual("External Z.ai lyrics generated (glm-5)", result.status_message)
        self.assertEqual("lyrics", request_plan_mock.call_args.kwargs["task_focus"])
        intent = request_plan_mock.call_args.kwargs["intent"]
        self.assertIn("Generate complete singable lyrics", intent)
        self.assertIn("Caption concept: energetic chiptune track", intent)
        self.assertIn("Return finished sung lyrics only", intent)
        self.assertIn("Use explicit [Verse 1], [Chorus], and [Verse 2]", intent)
        self.assertIn("same number of lines", intent)
        self.assertIn("Do not describe instrumentation", intent)
        self.assertEqual(120, request_plan_mock.call_args.kwargs["timeout_sec"])

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_generate_lyrics_from_caption_uses_generic_plain_text_output_contract(
        self,
        request_plan_mock,
    ) -> None:
        """Lyrics requests should use the same plain-text output contract for any provider."""
        request_plan_mock.return_value = (
            SimpleNamespace(
                caption="future garage",
                lyrics="[Verse 1]\nNeon rain keeps calling me home",
                bpm=130,
                duration=30.0,
                key_scale="F minor",
                vocal_language="en",
                time_signature="4/4",
            ),
            SimpleNamespace(label="OpenAI"),
            "gpt-4o-mini",
        )

        generate_lyrics_from_caption_with_external_provider(
            caption="future garage night drive",
            bpm=130,
            audio_duration=30.0,
            key_scale="F minor",
            time_signature="4/4",
            vocal_language="en",
        )

        intent = request_plan_mock.call_args.kwargs["intent"]
        self.assertIn("Return plain text lyrics only.", intent)
        self.assertIn("Do not return JSON, Python lists, dictionaries", intent)
        self.assertIn("Do not wrap sections inside objects", intent)
        self.assertIn("Example format: [Verse 1]", intent)
        self.assertIn("20 syllables or fewer", intent)
        self.assertIn("break it into a new lyric line", intent)

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_format_sample_includes_preservation_directives_in_intent(
        self,
        request_plan_mock,
    ) -> None:
        """Format intent should include preserve scaffolding when tags are present."""
        request_plan_mock.return_value = (
            SimpleNamespace(
                caption="formatted caption",
                lyrics="[Verse 1]\nformatted lyrics",
                bpm=100,
                duration=30.0,
                key_scale="A minor",
                vocal_language="en",
                time_signature="4/4",
            ),
            SimpleNamespace(label="Z.ai"),
            "glm-5",
        )

        format_sample_with_external_provider(
            caption="Synth intro with vocals and bass",
            lyrics="[Verse 1]\nplain line",
            user_metadata={},
        )

        intent = request_plan_mock.call_args.kwargs["intent"]
        self.assertIn("Preserve existing arrangement/instrument tags", intent)
        self.assertIn("Preserve arrangement tags exactly", intent)
        self.assertIn("Preserve instrument tags exactly", intent)
        self.assertEqual(120, request_plan_mock.call_args.kwargs["timeout_sec"])


if __name__ == "__main__":
    unittest.main()



