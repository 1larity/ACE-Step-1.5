"""Tests for enhancement prompt scaffold extraction helpers."""

from __future__ import annotations

import unittest

from acestep.text_tasks.enhancement_scaffold import (
    build_preservation_directives,
    extract_arrangement_tags,
    extract_instrument_tags,
)


class EnhancementScaffoldTests(unittest.TestCase):
    """Validate arrangement/instrument extraction used by enhancement prompts."""

    def test_extract_arrangement_tags_from_section_headers_and_caption(self) -> None:
        """Arrangement extraction should detect tags from lyrics headers and caption text."""
        caption = "Cinematic intro that builds into a strong chorus"
        lyrics = "[Verse 1]\nline one\n\n[Bridge]\nline two"
        tags = extract_arrangement_tags(caption=caption, lyrics=lyrics)
        self.assertEqual(["verse", "bridge", "intro", "chorus"], tags)

    def test_extract_instrument_tags_uses_common_aliases(self) -> None:
        """Instrument extraction should include canonical tags from common text aliases."""
        caption = "Warm synths, driving bassline, and expressive vocalist over piano"
        tags = extract_instrument_tags(caption=caption, lyrics="")
        self.assertEqual(["synth", "piano", "bass", "vocals"], tags)

    def test_build_preservation_directives_returns_empty_for_plain_text(self) -> None:
        """No directives should be emitted when no arrangement/instrument tags are present."""
        directives = build_preservation_directives(
            caption="dreamy and emotional",
            lyrics="falling into moonlight",
        )
        self.assertEqual("", directives)

    def test_extract_tags_ignore_plain_lyric_words_without_headers(self) -> None:
        """Plain lyric lines should not be promoted into arrangement or instrument tags."""
        caption = "dreamy and emotional"
        lyrics = "we drop into moonlight with drums in our chest"
        self.assertEqual([], extract_arrangement_tags(caption=caption, lyrics=lyrics))
        self.assertEqual([], extract_instrument_tags(caption=caption, lyrics=lyrics))

    def test_build_preservation_directives_contains_constraints(self) -> None:
        """Directive block should include preserve lines and non-contradiction rule."""
        directives = build_preservation_directives(
            caption="Acoustic guitar intro with female vocals",
            lyrics="[Verse 1]\ntext",
        )
        self.assertIn("Preserve arrangement tags exactly", directives)
        self.assertIn("Preserve instrument tags exactly", directives)
        self.assertIn("do not remove or contradict", directives)


if __name__ == "__main__":
    unittest.main()

