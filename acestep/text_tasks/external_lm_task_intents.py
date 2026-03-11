"""Intent-building helpers for external LM task adapters."""

from __future__ import annotations

from typing import Any

from .enhancement_scaffold import build_preservation_directives
from .external_lm_task_contracts import build_lyrics_output_contract


def build_create_sample_intent(*, query: str, instrumental: bool, vocal_language: str) -> str:
    """Build create-sample intent text for an external LM request."""
    intent = query.strip() or "NO USER INPUT"
    intent += f"\n\ninstrumental: {'true' if instrumental else 'false'}"
    if vocal_language and vocal_language != "unknown":
        intent += f"\nvocal_language: {vocal_language}"
    return intent


def build_lyrics_generation_intent(
    *,
    caption: str,
    bpm: Any,
    audio_duration: Any,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    retry: bool,
) -> str:
    """Build lyrics-generation intent text for an external LM request."""
    intent_parts = [
        "Generate complete singable lyrics from the caption and metadata below.",
        f"Caption concept: {caption or ''}",
        "Return finished sung lyrics only, not placeholders, not instructions, and not tag-only output.",
        "Use explicit [Verse 1], [Chorus], and [Verse 2] section headers with real lyric lines under each section.",
        "Keep repeated sections structurally matched: Verse 1 and Verse 2 should use the same number of lines, and repeated choruses should keep the same line count and hook shape.",
        "Do not describe instrumentation, arrangement, or production cues instead of lyrics.",
    ]
    intent_parts.extend(build_lyrics_output_contract())
    for label, value in (
        ("bpm", bpm),
        ("duration", audio_duration),
        ("key_scale", key_scale),
        ("time_signature", time_signature),
        ("vocal_language", vocal_language),
    ):
        if value not in (None, "", "unknown"):
            intent_parts.append(f"{label}: {value}")
    if retry:
        intent_parts.append("Use a different hook and imagery from the previous draft.")
    return "\n".join(intent_parts)


def build_format_sample_intent(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
) -> str:
    """Build format-sample intent text for an external LM request."""
    intent_parts = [
        "Please format and enrich the following for ACE-Step generation. Expand sparse captions or keyword lists into a complete structured caption rather than leaving them terse.",
        f"Caption: {caption or ''}",
        f"Lyrics: {lyrics or ''}",
    ]
    if user_metadata:
        for key in ("bpm", "duration", "keyscale", "timesignature", "language"):
            value = user_metadata.get(key)
            if value not in (None, "", "unknown"):
                intent_parts.append(f"{key}: {value}")
    preservation_directives = build_preservation_directives(caption=caption, lyrics=lyrics)
    if preservation_directives:
        intent_parts.append(
            "Preserve existing arrangement/instrument tags from the user input while enhancing text:"
        )
        intent_parts.append(preservation_directives)
    return "\n".join(intent_parts)
