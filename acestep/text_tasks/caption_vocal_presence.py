"""Caption normalization helpers for vocal-presence conditioning."""

from __future__ import annotations

from typing import Optional

from .caption_vocal_presence_detection import (
    caption_has_early_instrumentation_block,
    caption_has_early_vocal_character_block,
    caption_has_early_vocal_presence,
    caption_has_explicit_delayed_vocal_entry,
    looks_like_vocal_song,
)
from .caption_vocal_presence_sentences import (
    build_instrumentation_sentence,
    build_vocal_character_sentence,
    insert_setup_sentences_after_first_sentence,
)

GLOBAL_VOCAL_PRESENCE_SENTENCE = "Lead vocals stay present from the opening section onward."
GENERIC_INSTRUMENTATION_SENTENCE = (
    "Core instrumentation is established from the opening section and stays central throughout."
)


def ensure_caption_has_global_vocal_presence(
    caption: str,
    *,
    lyrics: str = "",
    vocal_language: str = "unknown",
    instrumental: Optional[bool] = None,
) -> str:
    """Add missing up-front vocal, singer-character, and instrumentation context.

    Args:
        caption: Generated caption to normalize.
        lyrics: Associated lyrics, used to detect vocal intent.
        vocal_language: Current vocal language metadata.
        instrumental: Optional instrumental flag when the caller has one.

    Returns:
        The original caption when no change is needed, otherwise a caption with
        short global setup sentences inserted before the narrative arc.
    """
    normalized_caption = (caption or "").strip()
    if not normalized_caption:
        return caption
    if not looks_like_vocal_song(
        lyrics=lyrics,
        vocal_language=vocal_language,
        instrumental=instrumental,
    ):
        return normalized_caption
    if caption_has_explicit_delayed_vocal_entry(normalized_caption):
        return normalized_caption

    needs_vocal_sentence = not caption_has_early_vocal_presence(normalized_caption)
    needs_character_sentence = not caption_has_early_vocal_character_block(normalized_caption)
    needs_instrumentation_sentence = not caption_has_early_instrumentation_block(normalized_caption)
    if not needs_vocal_sentence and not needs_character_sentence and not needs_instrumentation_sentence:
        return normalized_caption

    setup_sentences: list[str] = []
    if needs_vocal_sentence:
        setup_sentences.append(GLOBAL_VOCAL_PRESENCE_SENTENCE)
    if needs_character_sentence:
        character_sentence = build_vocal_character_sentence(normalized_caption)
        if character_sentence:
            setup_sentences.append(character_sentence)
    if needs_instrumentation_sentence:
        setup_sentences.append(
            build_instrumentation_sentence(normalized_caption, GENERIC_INSTRUMENTATION_SENTENCE)
        )

    if needs_vocal_sentence:
        return " ".join(setup_sentences + [normalized_caption]).strip()
    return insert_setup_sentences_after_first_sentence(normalized_caption, setup_sentences)
