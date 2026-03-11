"""Helpers for robust lyric generation from caption-only user actions."""

from __future__ import annotations

import math
import random
import re
from typing import Any

from acestep.ui.gradio.events.generation.lyric_generation_fallback_locales import (
    resolve_fallback_section_pools,
)

_TAG_ONLY_LINE_PATTERN = re.compile(r"^\s*\[[^\]]+\]\s*$")
_BRACKET_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
_XML_TAG_PATTERN = re.compile(r"</?[^>]+>")
_CODE_FENCE_MARKER_PATTERN = re.compile(r"`{3,}")
_NON_LYRIC_TOKENS = set(
    "bar bars bridge breakdown chorus drop humming hook hum inst instrumental interlude intro la "
    "lyrics lyric music na none no only outro post pre refrain section tag verse vocal vocalise "
    "vocalize vocals".split()
)


def is_tag_only_lyrics(lyrics: str | None) -> bool:
    """Return ``True`` when lyrics are empty or have no singable word content."""
    if not lyrics or not lyrics.strip():
        return True

    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return True

    return not any(_line_contains_lyric_words(line) for line in lines)


def build_lyrics_generation_seed(
    *,
    vocal_language: str,
    bpm: Any,
    audio_duration: Any,
    retry: bool = False,
) -> str:
    """Build a compact section scaffold seed for lyric generation."""
    language_text = (
        vocal_language.strip()
        if vocal_language and vocal_language != "unknown"
        else "the requested language"
    )
    bpm_text = _to_positive_number_text(bpm, "bpm")
    duration_text = _to_positive_number_text(audio_duration, "seconds")
    retry_text = " Use a different hook and imagery from the previous draft." if retry else ""
    return (
        "[Verse 1]\n"
        f"(2-4 sung lines in {language_text}; start the narrative)\n\n"
        "[Chorus]\n"
        "(2-4 sung lines; memorable hook)\n\n"
        "[Verse 2]\n"
        "(2-4 sung lines; develop the narrative)\n\n"
        f"(tempo: {bpm_text}; duration: {duration_text}).{retry_text}"
    )


def build_vocal_caption_prompt(
    *,
    caption: str,
    vocal_language: str,
    bpm: Any,
    audio_duration: Any,
) -> str:
    """Build a concise vocal caption prompt for lyric generation."""
    base_caption = caption.strip() or "NO USER INPUT"
    language_text = vocal_language if vocal_language and vocal_language != "unknown" else "auto"
    bpm_text = _to_positive_number_text(bpm, "bpm")
    duration_text = _to_positive_number_text(audio_duration, "seconds")
    return (
        f"{base_caption}\n\n"
        "Write lead-vocal lyrics in Verse->Chorus->Verse form with singable lines. "
        f"Language={language_text}, tempo={bpm_text}, duration={duration_text}."
    )


def build_duration_aware_fallback_lyrics(
    *,
    caption: str,
    vocal_language: str,
    bpm: Any,
    audio_duration: Any,
    variation_nonce: int | None = None,
) -> str:
    """Generate deterministic structured lyrics when LM output is tag-only."""
    theme_tokens = _extract_word_tokens(caption or "")
    theme = " ".join(theme_tokens[:4]).lower() if theme_tokens else "the midnight skyline"
    line_count = _estimate_line_count(bpm=bpm, audio_duration=audio_duration)
    chorus_lines = 4 if line_count >= 8 else 2
    verse1_lines = max(3, (line_count - chorus_lines) // 2)
    verse2_lines = max(2, line_count - chorus_lines - verse1_lines)
    verse_pool, chorus_pool, verse2_pool = resolve_fallback_section_pools(vocal_language, theme)
    base_seed = int(variation_nonce or 0)
    verse1 = _build_section_lines(verse_pool, verse1_lines, seed=base_seed + 11)
    chorus = _build_section_lines(chorus_pool, chorus_lines, seed=base_seed + 29)
    verse2 = _build_section_lines(verse2_pool, verse2_lines, seed=base_seed + 53)

    return (
        "[Verse 1]\n"
        f"{verse1}\n\n"
        "[Chorus]\n"
        f"{chorus}\n\n"
        "[Verse 2]\n"
        f"{verse2}"
    )


def _build_section_lines(pool: list[str], count: int, seed: int | None = None) -> str:
    """Return ``count`` lines by seeded shuffled selection from provided pool."""
    ordered_pool = list(pool)
    if seed is not None:
        random.Random(seed).shuffle(ordered_pool)
    lines = [ordered_pool[idx % len(ordered_pool)] for idx in range(max(1, count))]
    return "\n".join(lines)


def _line_contains_lyric_words(line: str) -> bool:
    """Return ``True`` when a line has meaningful non-tag lexical content."""
    if _TAG_ONLY_LINE_PATTERN.fullmatch(line):
        return False

    normalized = _BRACKET_TAG_PATTERN.sub(" ", line)
    normalized = _XML_TAG_PATTERN.sub(" ", normalized)
    normalized = _CODE_FENCE_MARKER_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[*_#>|~]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    tokens = [token.lower() for token in _extract_word_tokens(normalized)]
    filtered_tokens = [
        token for token in tokens if not token.isdigit() and token not in _NON_LYRIC_TOKENS
    ]
    return bool(filtered_tokens)


def _extract_word_tokens(text: str) -> list[str]:
    """Split text into Unicode-aware word tokens using alphanumeric runs."""
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum() or (char == "'" and current):
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _estimate_line_count(*, bpm: Any, audio_duration: Any) -> int:
    """Estimate lyric line count from tempo and duration constraints."""
    bpm_val = _to_positive_float(bpm)
    duration_val = _to_positive_float(audio_duration)
    if bpm_val is None or duration_val is None:
        return 8
    beats = (bpm_val * duration_val) / 60.0
    line_count = int(math.ceil(beats / 8.0))
    return max(6, min(20, line_count))


def _to_positive_float(value: Any) -> float | None:
    """Parse positive float value or return ``None`` when unavailable."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_positive_number_text(value: Any, unit: str) -> str:
    """Render positive numeric metadata values for prompt text."""
    parsed = _to_positive_float(value)
    if parsed is None:
        return f"unspecified {unit}"
    if float(parsed).is_integer():
        return f"{int(parsed)} {unit}"
    return f"{parsed:.2f} {unit}"