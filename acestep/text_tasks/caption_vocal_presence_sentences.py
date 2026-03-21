"""Sentence-building helpers for caption vocal-presence normalization."""

from __future__ import annotations

import re

from .caption_vocal_presence_detection import (
    GENDER_PATTERN,
    extract_delivery_descriptors,
    extract_vocal_context_window,
    normalize_gender,
    split_first_sentence,
)

INSTRUMENTATION_PHRASE_PATTERN = re.compile(
    r"\b(?:[a-z]+(?:-[a-z]+)?\s+){0,2}(drums?|percussion|bass(?:line)?|synth(?:s)?|"
    r"pads?|piano|guitar|strings|brass|horns|congas|timbales|beat(?:s)?|arpeggios?|arp(?:s)?)\b",
    flags=re.IGNORECASE,
)


def normalize_instrument_phrase(phrase: str) -> str:
    """Normalize an extracted instrumentation phrase for sentence insertion."""
    normalized = " ".join((phrase or "").strip().lower().split())
    return re.sub(r"^(?:a|an|the)\s+", "", normalized)


def join_readable_phrases(phrases: list[str]) -> str:
    """Join phrases into a short readable list."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def extract_instrumentation_phrases(caption: str) -> list[str]:
    """Extract up to three instrumentation phrases from the generated caption."""
    normalized = (caption or "").strip()
    if not normalized:
        return []

    phrases: list[str] = []
    seen: set[str] = set()
    for match in INSTRUMENTATION_PHRASE_PATTERN.finditer(normalized):
        phrase = normalize_instrument_phrase(match.group(0))
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= 3:
            break
    return phrases


def build_instrumentation_sentence(caption: str, generic_sentence: str) -> str:
    """Build a short global instrumentation sentence from the caption when possible."""
    phrases = extract_instrumentation_phrases(caption)
    if not phrases:
        return generic_sentence
    return (
        "Core instrumentation stays present throughout, built around "
        f"{join_readable_phrases(phrases)}."
    )


def build_vocal_character_sentence(caption: str) -> str:
    """Build a short singer-character sentence from later caption details when possible."""
    context_window = extract_vocal_context_window(caption)
    gender_match = GENDER_PATTERN.search(context_window)
    gender = normalize_gender(gender_match.group(1)) if gender_match else ""
    descriptor_text = join_readable_phrases(extract_delivery_descriptors(context_window))

    if gender and descriptor_text:
        return f"The lead singer is a {descriptor_text} {gender} vocalist."
    if gender:
        return f"The lead singer is a {gender} vocalist with expressive delivery."
    if descriptor_text:
        return f"The lead singer delivers the topline with {descriptor_text} phrasing."
    return ""


def insert_setup_sentences_after_first_sentence(caption: str, sentences: list[str]) -> str:
    """Insert setup sentences after the first sentence of the existing caption."""
    first_sentence, remainder = split_first_sentence(caption)
    if not first_sentence:
        return " ".join(sentences + [caption]).strip()
    if not remainder:
        return " ".join([first_sentence, *sentences]).strip()
    return " ".join([first_sentence, *sentences, remainder]).strip()
