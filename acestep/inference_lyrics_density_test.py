"""Unit tests for lyric-density helpers in ``acestep.inference``."""

from __future__ import annotations

import unittest

from acestep.inference import (
    _LYRIC_DENSITY_SYLLABLE_BUDGET_MULTIPLIER,
    _apply_soft_lyric_density_guard,
    _count_lyric_syllables,
    _count_lyric_words,
    _estimate_calculated_vocal_syllable_capacity,
    _estimate_lyric_syllable_budget,
    _estimate_lyric_word_budget,
    _normalize_lyrics_for_generation,
)


def _build_lyrics(word: str, count: int, section: str = "[Verse 1]") -> str:
    """Build deterministic lyrics with a section tag and a fixed word count."""
    body = " ".join([word] * count)
    return f"{section}\n{body}"


class InferenceLyricDensityTests(unittest.TestCase):
    """Validate non-strict lyric-density warning and soft-trim behavior."""

    def test_syllable_budget_is_two_times_calculated_vocal_capacity(self) -> None:
        """Syllable budget should be 200% of estimated vocal syllable capacity."""
        vocal_capacity = _estimate_calculated_vocal_syllable_capacity(
            duration=20,
            bpm=120,
            time_signature="4/4",
        )
        syllable_budget = _estimate_lyric_syllable_budget(
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertIsNotNone(vocal_capacity)
        self.assertIsNotNone(syllable_budget)
        expected_budget = int(round(vocal_capacity * _LYRIC_DENSITY_SYLLABLE_BUDGET_MULTIPLIER))
        self.assertEqual(expected_budget, syllable_budget)

    def test_budget_accounts_for_time_signature(self) -> None:
        """6/8 should provide fewer words than 4/4 under the same tempo/duration."""
        budget_44 = _estimate_lyric_word_budget(duration=20, bpm=120, time_signature="4/4")
        budget_68 = _estimate_lyric_word_budget(duration=20, bpm=120, time_signature="6/8")

        self.assertIsNotNone(budget_44)
        self.assertIsNotNone(budget_68)
        self.assertGreater(budget_44, budget_68)

    def test_dense_lyrics_warn_without_trimming_when_not_extreme(self) -> None:
        """Moderately dense lyrics should emit a warning and preserve input text."""
        lyrics = _build_lyrics("groove", 120)

        adjusted, warning, trimmed = _apply_soft_lyric_density_guard(
            lyrics=lyrics,
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertFalse(trimmed)
        self.assertEqual(lyrics, adjusted)
        self.assertIn("too dense", warning or "")

    def test_extreme_density_triggers_soft_trim(self) -> None:
        """Very dense lyrics should be softly trimmed instead of hard-failing."""
        lyrics = _build_lyrics("pulse", 320)

        adjusted, warning, trimmed = _apply_soft_lyric_density_guard(
            lyrics=lyrics,
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertTrue(trimmed)
        self.assertIn("soft trim", warning or "")
        self.assertLess(_count_lyric_words(adjusted), _count_lyric_words(lyrics))
        self.assertIn("[Verse 1]", adjusted)

    def test_high_syllable_density_trims_even_with_moderate_word_count(self) -> None:
        """Long multisyllabic words should trigger density trim via syllable budget."""
        lyrics = _build_lyrics("imagination", 70)

        adjusted, warning, trimmed = _apply_soft_lyric_density_guard(
            lyrics=lyrics,
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertTrue(trimmed)
        self.assertIn("syllables", warning or "")
        self.assertLess(_count_lyric_syllables(adjusted), _count_lyric_syllables(lyrics))

    def test_non_latin_lyrics_are_counted_for_density(self) -> None:
        """Unicode lyric text should not collapse to zero word/syllable counts."""
        lyrics = _build_lyrics("\u5fc3\u8df3", 80)

        adjusted, warning, trimmed = _apply_soft_lyric_density_guard(
            lyrics=lyrics,
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertGreater(_count_lyric_words(lyrics), 0)
        self.assertGreater(_count_lyric_syllables(lyrics), 0)
        self.assertTrue(trimmed)
        self.assertIn("too dense", warning or "")
        self.assertLess(_count_lyric_words(adjusted), _count_lyric_words(lyrics))

    def test_instrumental_placeholder_is_ignored(self) -> None:
        """Instrumental placeholders should bypass lyric-density checks."""
        adjusted, warning, trimmed = _apply_soft_lyric_density_guard(
            lyrics="[Instrumental]",
            duration=20,
            bpm=120,
            time_signature="4/4",
        )

        self.assertEqual("[Instrumental]", adjusted)
        self.assertIsNone(warning)
        self.assertFalse(trimmed)

    def test_normalize_lyrics_for_generation_splits_commas_into_lines(self) -> None:
        """Comma-joined lyric clauses should be separated into distinct sung lines."""
        lyrics = "[Verse 1]\nWe're the rhythm of the city,where the wild hearts meet."

        adjusted = _normalize_lyrics_for_generation(lyrics)

        self.assertEqual(
            "[Verse 1]\nWe're the rhythm of the city\nwhere the wild hearts meet.",
            adjusted,
        )

    def test_normalize_lyrics_for_generation_preserves_lines_without_commas(self) -> None:
        """Already separated lyric lines should pass through unchanged."""
        lyrics = "[Verse 1]\nNeon rain keeps calling me home"

        adjusted = _normalize_lyrics_for_generation(lyrics)

        self.assertEqual(lyrics, adjusted)


if __name__ == "__main__":
    unittest.main()
