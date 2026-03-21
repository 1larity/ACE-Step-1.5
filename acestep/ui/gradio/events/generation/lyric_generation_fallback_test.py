"""Tests for lyric-generation fallback helpers."""

import unittest

from acestep.ui.gradio.events.generation.lyric_generation_fallback import (
    build_duration_aware_fallback_lyrics,
    build_lyrics_generation_seed,
    build_vocal_caption_prompt,
    is_tag_only_lyrics,
)


class LyricGenerationFallbackTests(unittest.TestCase):
    """Validate tag-only detection and deterministic lyric fallback behavior."""

    def test_is_tag_only_lyrics_true_for_only_bracket_tags(self):
        """Bracket-only content should be treated as tag-only."""
        self.assertTrue(is_tag_only_lyrics("[Instrumental]\n[Vocalise]"))

    def test_is_tag_only_lyrics_true_for_plain_instrumental_keyword(self):
        """Plain instrumental marker text should still be treated as tag-only."""
        self.assertTrue(is_tag_only_lyrics("instrumental"))

    def test_is_tag_only_lyrics_true_for_markup_wrapped_tags(self):
        """Markup wrappers around tag-only content should remain tag-only."""
        self.assertTrue(is_tag_only_lyrics("<lyrics>[Instrumental] [Vocalise]</lyrics>"))

    def test_is_tag_only_lyrics_false_for_content_lines(self):
        """Lyrics with text lines should not be treated as tag-only."""
        self.assertFalse(is_tag_only_lyrics("[Verse 1]\nwe rise tonight"))

    def test_is_tag_only_lyrics_false_for_japanese_content_lines(self):
        """Non-Latin lyric lines should be accepted as singable content."""
        self.assertFalse(is_tag_only_lyrics("[Verse 1]\n夜を越えて君と歌う"))

    def test_build_lyrics_generation_seed_includes_structure_and_timing(self):
        """Seed should include section scaffolding and timing hints."""
        seed = build_lyrics_generation_seed(
            vocal_language="ja",
            bpm=96,
            audio_duration=24,
            retry=True,
        )
        self.assertIn("[Verse 1]", seed)
        self.assertIn("[Chorus]", seed)
        self.assertIn("[Verse 2]", seed)
        self.assertIn("in ja", seed)
        self.assertIn("tempo: 96 bpm", seed)
        self.assertIn("different hook and imagery", seed)

    def test_build_vocal_caption_prompt_includes_compact_requirements(self):
        """Caption prompt should encode compact vocal + metadata requirements."""
        prompt = build_vocal_caption_prompt(
            caption="dreamy synth-pop skyline",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
        )
        self.assertIn("Write lead-vocal lyrics", prompt)
        self.assertIn("Language=en", prompt)
        self.assertIn("tempo=120 bpm", prompt)

    def test_build_duration_aware_fallback_lyrics_has_song_sections(self):
        """Fallback should return a multi-section lyric scaffold with non-tag lines."""
        lyrics = build_duration_aware_fallback_lyrics(
            caption="cinematic orchestral pop",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
        )
        self.assertIn("[Verse 1]", lyrics)
        self.assertIn("[Chorus]", lyrics)
        self.assertIn("[Verse 2]", lyrics)
        self.assertIn("\n", lyrics.strip())
        self.assertNotEqual("[Instrumental]", lyrics.strip())

    def test_build_duration_aware_fallback_lyrics_localizes_supported_language(self):
        """Supported non-English languages should receive localized fallback text."""
        lyrics = build_duration_aware_fallback_lyrics(
            caption="dreamy city-pop skyline",
            vocal_language="ja",
            bpm=96,
            audio_duration=24,
            variation_nonce=101,
        )
        self.assertIn("[Verse 1]", lyrics)
        self.assertIn("今この声で夜を越えていこう", lyrics)
        self.assertNotIn("Deliver these lyrics", lyrics)

    def test_build_duration_aware_fallback_lyrics_changes_with_variation_nonce(self):
        """Different variation nonces should produce different fallback scaffolds."""
        lyrics_a = build_duration_aware_fallback_lyrics(
            caption="cinematic orchestral pop",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
            variation_nonce=101,
        )
        lyrics_b = build_duration_aware_fallback_lyrics(
            caption="cinematic orchestral pop",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
            variation_nonce=202,
        )
        self.assertNotEqual(lyrics_a, lyrics_b)

    def test_build_duration_aware_fallback_lyrics_is_stable_for_same_nonce(self):
        """Same nonce should produce deterministic fallback output."""
        lyrics_a = build_duration_aware_fallback_lyrics(
            caption="cinematic orchestral pop",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
            variation_nonce=303,
        )
        lyrics_b = build_duration_aware_fallback_lyrics(
            caption="cinematic orchestral pop",
            vocal_language="en",
            bpm=120,
            audio_duration=30,
            variation_nonce=303,
        )
        self.assertEqual(lyrics_a, lyrics_b)


if __name__ == "__main__":
    unittest.main()