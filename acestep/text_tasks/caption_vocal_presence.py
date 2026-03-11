"""Caption normalization helpers for vocal-presence conditioning."""

from __future__ import annotations

import re
from typing import Optional

_EARLY_VOCAL_MARKER_PATTERN = re.compile(
    r"\b(vocal(?:s|ist)?|voice|singer|singing|sung|lead vocal|female vocal|male vocal|"
    r"harmon(?:y|ies)|choir|duet|rap vocal|spoken[- ]word)\b",
    flags=re.IGNORECASE,
)
_EARLY_INSTRUMENTATION_MARKER_PATTERN = re.compile(
    r"\b(drums?|percussion|bass(?:line)?|synth(?:s)?|pads?|piano|guitar|strings|"
    r"brass|horns|congas|timbales|beat(?:s)?|arpeggios?|arp(?:s)?)\b",
    flags=re.IGNORECASE,
)
_INSTRUMENTATION_PHRASE_PATTERN = re.compile(
    r"\b(?:[a-z]+(?:-[a-z]+)?\s+){0,2}(drums?|percussion|bass(?:line)?|synth(?:s)?|"
    r"pads?|piano|guitar|strings|brass|horns|congas|timbales|beat(?:s)?|arpeggios?|arp(?:s)?)\b",
    flags=re.IGNORECASE,
)
_GENDER_PATTERN = re.compile(r"\b(female|male|woman|man|feminine|masculine)\b", flags=re.IGNORECASE)
_DELIVERY_DESCRIPTORS = (
    "soulful",
    "bright",
    "dynamic",
    "charismatic",
    "breathy",
    "raspy",
    "intimate",
    "tender",
    "urgent",
    "ethereal",
    "gentle",
    "gritty",
    "smooth",
    "powerful",
    "melancholic",
    "whispered",
    "commanding",
)
_EXPLICIT_DELAYED_VOCAL_CUES = (
    "vocals enter later",
    "vocal enters later",
    "vocals arrive later",
    "vocal arrives later",
    "voice enters later",
    "toward the end",
    "towards the end",
    "at the end",
    "in the final section",
    "in the final chorus",
    "in the outro",
    "ending with vocals",
    "ends with vocals",
    "brief vocal section",
    "late vocal entry",
)
_INSTRUMENTAL_LYRIC_MARKERS = {"[instrumental]", "[inst]", "instrumental"}
_GLOBAL_VOCAL_PRESENCE_SENTENCE = "Lead vocals stay present from the opening section onward."
_GENERIC_INSTRUMENTATION_SENTENCE = (
    "Core instrumentation is established from the opening section and stays central throughout."
)


def _looks_like_vocal_song(
    lyrics: str = "",
    vocal_language: str = "unknown",
    instrumental: Optional[bool] = None,
) -> bool:
    """Return whether the current caption should be treated as a vocal song."""
    normalized_lyrics = (lyrics or "").strip().lower()
    if instrumental is True or normalized_lyrics in _INSTRUMENTAL_LYRIC_MARKERS:
        return False
    if lyrics and normalized_lyrics:
        return True

    normalized_language = (vocal_language or "").strip().lower()
    if normalized_language and normalized_language not in {"unknown", "instrumental", "auto", "n/a"}:
        return True

    return instrumental is False



def _split_first_sentence(caption: str) -> tuple[str, str]:
    """Split caption into first sentence and trailing remainder."""
    normalized = (caption or "").strip()
    if not normalized:
        return "", ""

    match = re.search(r"[.!?](?:\s|$)", normalized)
    if not match:
        return normalized, ""

    split_index = match.end()
    first_sentence = normalized[:split_index].strip()
    remainder = normalized[split_index:].strip()
    return first_sentence, remainder



def _caption_has_early_vocal_presence(caption: str) -> bool:
    """Return whether the caption already establishes vocals near the start."""
    normalized = (caption or "").strip()
    if not normalized:
        return False

    first_sentence, _ = _split_first_sentence(normalized)
    return bool(_EARLY_VOCAL_MARKER_PATTERN.search(first_sentence))



def _caption_has_early_instrumentation_block(caption: str) -> bool:
    """Return whether the caption already establishes instrumentation near the start."""
    normalized = (caption or "").strip()
    if not normalized:
        return False

    first_sentence, _ = _split_first_sentence(normalized)
    return bool(_EARLY_INSTRUMENTATION_MARKER_PATTERN.search(first_sentence))



def _caption_has_explicit_delayed_vocal_entry(caption: str) -> bool:
    """Return whether the caption intentionally delays vocal entry."""
    normalized = (caption or "").strip().lower()
    return any(cue in normalized for cue in _EXPLICIT_DELAYED_VOCAL_CUES)



def _normalize_instrument_phrase(phrase: str) -> str:
    """Normalize an extracted instrumentation phrase for sentence insertion."""
    normalized = " ".join((phrase or "").strip().lower().split())
    return re.sub(r"^(?:a|an|the)\s+", "", normalized)



def _join_readable_phrases(phrases: list[str]) -> str:
    """Join phrases into a short readable list."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"



def _extract_instrumentation_phrases(caption: str) -> list[str]:
    """Extract up to three instrumentation phrases from the generated caption."""
    normalized = (caption or "").strip()
    if not normalized:
        return []

    phrases: list[str] = []
    seen: set[str] = set()
    for match in _INSTRUMENTATION_PHRASE_PATTERN.finditer(normalized):
        phrase = _normalize_instrument_phrase(match.group(0))
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= 3:
            break
    return phrases



def _build_instrumentation_sentence(caption: str) -> str:
    """Build a short global instrumentation sentence from the caption when possible."""
    phrases = _extract_instrumentation_phrases(caption)
    if not phrases:
        return _GENERIC_INSTRUMENTATION_SENTENCE
    return (
        "Core instrumentation stays present throughout, built around "
        f"{_join_readable_phrases(phrases)}."
    )



def _extract_vocal_context_window(caption: str) -> str:
    """Extract a short local window around the first vocal marker."""
    normalized = (caption or "").strip()
    if not normalized:
        return ""

    match = _EARLY_VOCAL_MARKER_PATTERN.search(normalized)
    if not match:
        return normalized
    start = max(0, match.start() - 80)
    end = min(len(normalized), match.end() + 80)
    return normalized[start:end]



def _normalize_gender(gender: str) -> str:
    """Normalize extracted gender wording to a stable adjective."""
    normalized = (gender or "").strip().lower()
    if normalized in {"woman", "feminine"}:
        return "female"
    if normalized in {"man", "masculine"}:
        return "male"
    return normalized



def _extract_delivery_descriptors(text: str) -> list[str]:
    """Extract up to two delivery adjectives from the provided text."""
    normalized = (text or "").strip().lower()
    descriptors: list[str] = []
    for descriptor in _DELIVERY_DESCRIPTORS:
        if re.search(rf"\b{re.escape(descriptor)}\b", normalized) and descriptor not in descriptors:
            descriptors.append(descriptor)
        if len(descriptors) >= 2:
            break
    return descriptors



def _caption_has_early_vocal_character_block(caption: str) -> bool:
    """Return whether the caption already establishes singer character near the start."""
    normalized = (caption or "").strip()
    if not normalized:
        return False

    first_sentence, _ = _split_first_sentence(normalized)
    return bool(_GENDER_PATTERN.search(first_sentence) or _extract_delivery_descriptors(first_sentence))



def _build_vocal_character_sentence(caption: str) -> str:
    """Build a short singer-character sentence from later caption details when possible."""
    context_window = _extract_vocal_context_window(caption)
    gender_match = _GENDER_PATTERN.search(context_window)
    gender = _normalize_gender(gender_match.group(1)) if gender_match else ""
    descriptors = _extract_delivery_descriptors(context_window)
    descriptor_text = _join_readable_phrases(descriptors)

    if gender and descriptor_text:
        return f"The lead singer is a {descriptor_text} {gender} vocalist."
    if gender:
        return f"The lead singer is a {gender} vocalist with expressive delivery."
    if descriptor_text:
        return f"The lead singer delivers the topline with {descriptor_text} phrasing."
    return ""



def _insert_setup_sentences_after_first_sentence(caption: str, sentences: list[str]) -> str:
    """Insert setup sentences after the first sentence of the existing caption."""
    first_sentence, remainder = _split_first_sentence(caption)
    if not first_sentence:
        return " ".join(sentences + [caption]).strip()
    if not remainder:
        return " ".join([first_sentence, *sentences]).strip()
    return " ".join([first_sentence, *sentences, remainder]).strip()



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
    if not _looks_like_vocal_song(
        lyrics=lyrics,
        vocal_language=vocal_language,
        instrumental=instrumental,
    ):
        return normalized_caption
    if _caption_has_explicit_delayed_vocal_entry(normalized_caption):
        return normalized_caption

    needs_vocal_sentence = not _caption_has_early_vocal_presence(normalized_caption)
    needs_character_sentence = not _caption_has_early_vocal_character_block(normalized_caption)
    needs_instrumentation_sentence = not _caption_has_early_instrumentation_block(normalized_caption)
    if not needs_vocal_sentence and not needs_character_sentence and not needs_instrumentation_sentence:
        return normalized_caption

    setup_sentences: list[str] = []
    if needs_vocal_sentence:
        setup_sentences.append(_GLOBAL_VOCAL_PRESENCE_SENTENCE)
    if needs_character_sentence:
        character_sentence = _build_vocal_character_sentence(normalized_caption)
        if character_sentence:
            setup_sentences.append(character_sentence)
    if needs_instrumentation_sentence:
        setup_sentences.append(_build_instrumentation_sentence(normalized_caption))

    if needs_vocal_sentence:
        return " ".join(setup_sentences + [normalized_caption]).strip()
    return _insert_setup_sentences_after_first_sentence(normalized_caption, setup_sentences)
