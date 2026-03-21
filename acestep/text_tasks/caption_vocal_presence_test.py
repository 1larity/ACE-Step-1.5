"""Tests for generated-caption vocal-presence normalization."""

from __future__ import annotations

import unittest

from acestep.text_tasks.caption_vocal_presence import ensure_caption_has_global_vocal_presence


class CaptionVocalPresenceTests(unittest.TestCase):
    """Validate when generated captions should receive an early vocal-presence block."""

    def test_injects_vocal_character_and_instrumentation_sentences_when_missing_early(self):
        """Later singer details should be pulled forward into the setup block."""
        result = ensure_caption_has_global_vocal_presence(
            (
                "A neon future garage track grows into a late-night drop. "
                "It later reveals a soulful female vocalist with dynamic delivery."
            ),
            lyrics="[Verse 1]\nWe run through static light",
            vocal_language="en",
        )

        self.assertTrue(result.startswith("Lead vocals stay present from the opening section onward."))
        self.assertIn("The lead singer is a soulful and dynamic female vocalist.", result)
        self.assertIn(
            "Core instrumentation is established from the opening section and stays central throughout.",
            result,
        )

    def test_does_not_invent_generic_singer_sentence_when_caption_never_defined_one(self):
        """Helper should not invent a neutral singer sentence when caption omitted singer character."""
        caption = "Lead vocals stay present from the opening section onward. The song blooms into a wider midnight lift."

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Verse 1]\nHold me in the glow",
            vocal_language="en",
        )

        self.assertNotIn("The lead singer delivers the topline", result)
        self.assertIn(
            "Core instrumentation is established from the opening section and stays central throughout.",
            result,
        )

    def test_inserts_only_instrumentation_after_existing_early_vocal_sentence_without_singer_detail(self):
        """When singer detail is absent, helper should add instrumentation but not invent a singer definition."""
        caption = (
            "Lead vocals stay present from the opening section onward. "
            "The song blooms into a wider midnight lift."
        )

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Verse 1]\nHold me in the glow",
            vocal_language="en",
        )

        self.assertTrue(result.startswith("Lead vocals stay present from the opening section onward."))
        self.assertNotIn("The lead singer delivers the topline", result)
        self.assertIn(
            "Core instrumentation is established from the opening section and stays central throughout.",
            result,
        )

    def test_preserves_caption_when_vocals_character_and_instrumentation_are_already_established_early(self):
        """Captions with early singer character and instrumentation should stay unchanged."""
        caption = "Soulful female vocals ride warm synth pads into a bright chorus."

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Verse 1]\nHold me in the glow",
            vocal_language="en",
        )

        self.assertEqual(caption, result)

    def test_preserves_caption_when_vocal_entry_is_explicitly_delayed(self):
        """Delayed-vocal captions should not be overridden with always-on setup sentences."""
        caption = "A spacious electronic build keeps things instrumental at first, with vocals entering later in the outro."

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Verse 1]\nHold me in the glow",
            vocal_language="en",
        )

        self.assertEqual(caption, result)

    def test_generic_end_timing_without_vocal_cue_does_not_skip_normalization(self):
        """Arrangement timing alone should not be treated as delayed vocal entry."""
        caption = "A shimmering synth motif lands at the end before the groove opens wider."

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Verse 1]\nHold me in the glow",
            vocal_language="en",
        )

        self.assertTrue(result.startswith("Lead vocals stay present from the opening section onward."))

    def test_skips_instrumental_captions(self):
        """Instrumental outputs should not receive vocal or instrumentation setup sentences."""
        caption = "A meditative solo piano piece slowly blooms into a cinematic ending."

        result = ensure_caption_has_global_vocal_presence(
            caption,
            lyrics="[Instrumental]",
            vocal_language="instrumental",
            instrumental=True,
        )

        self.assertEqual(caption, result)


if __name__ == "__main__":
    unittest.main()
