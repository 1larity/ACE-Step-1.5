"""Detection helpers for caption vocal-presence normalization."""

from __future__ import annotations

import re
from typing import Optional

EARLY_VOCAL_MARKER_PATTERN = re.compile(
    r"\b(vocal(?:s|ist)?|voice|singer|singing|sung|lead vocal|female vocal|male vocal|"
    r"harmon(?:y|ies)|choir|duet|rap vocal|spoken[- ]word)\b",
    flags=re.IGNORECASE,
)
EARLY_INSTRUMENTATION_MARKER_PATTERN = re.compile(
    r"\b(drums?|percussion|bass(?:line)?|synth(?:s)?|pads?|piano|guitar|strings|"
    r"brass|horns|congas|timbales|beat(?:s)?|arpeggios?|arp(?:s)?)\b",
    flags=re.IGNORECASE,
)
GENDER_PATTERN = re.compile(r"\b(female|male|woman|man|feminine|masculine)\b", flags=re.IGNORECASE)
DELIVERY_DESCRIPTORS = (
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
EXPLICIT_DELAYED_VOCAL_CUES = (
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
INSTRUMENTAL_LYRIC_MARKERS = {"[instrumental]", "[inst]", "instrumental"}


def looks_like_vocal_song(
    lyrics: str = "",
    vocal_language: str = "unknown",
    instrumental: Optional[bool] = None,
) -> bool:
    """Return whether the current caption should be treated as a vocal song."""
    normalized_lyrics = (lyrics or "").strip().lower()
    if instrumental is True or normalized_lyrics in INSTRUMENTAL_LYRIC_MARKERS:
        return False
    if lyrics and normalized_lyrics:
        return True

    normalized_language = (vocal_language or "").strip().lower()
    if normalized_language and normalized_language not in {"unknown", "instrumental", "auto", "n/a"}:
        return True
    return instrumental is False


def split_first_sentence(caption: str) -> tuple[str, str]:
    """Split caption into first sentence and trailing remainder."""
    normalized = (caption or "").strip()
    if not normalized:
        return "", ""

    match = re.search(r"[.!?](?:\s|$)", normalized)
    if not match:
        return normalized, ""

    split_index = match.end()
    return normalized[:split_index].strip(), normalized[split_index:].strip()


def caption_has_early_vocal_presence(caption: str) -> bool:
    """Return whether the caption already establishes vocals near the start."""
    first_sentence, _ = split_first_sentence((caption or "").strip())
    return bool(first_sentence and EARLY_VOCAL_MARKER_PATTERN.search(first_sentence))


def caption_has_early_instrumentation_block(caption: str) -> bool:
    """Return whether the caption already establishes instrumentation near the start."""
    first_sentence, _ = split_first_sentence((caption or "").strip())
    return bool(first_sentence and EARLY_INSTRUMENTATION_MARKER_PATTERN.search(first_sentence))


def caption_has_explicit_delayed_vocal_entry(caption: str) -> bool:
    """Return whether the caption intentionally delays vocal entry."""
    normalized = (caption or "").strip().lower()
    return any(cue in normalized for cue in EXPLICIT_DELAYED_VOCAL_CUES)


def extract_vocal_context_window(caption: str) -> str:
    """Extract a short local window around the first vocal marker."""
    normalized = (caption or "").strip()
    if not normalized:
        return ""

    match = EARLY_VOCAL_MARKER_PATTERN.search(normalized)
    if not match:
        return normalized
    start = max(0, match.start() - 80)
    end = min(len(normalized), match.end() + 80)
    return normalized[start:end]


def normalize_gender(gender: str) -> str:
    """Normalize extracted gender wording to a stable adjective."""
    normalized = (gender or "").strip().lower()
    if normalized in {"woman", "feminine"}:
        return "female"
    if normalized in {"man", "masculine"}:
        return "male"
    return normalized


def extract_delivery_descriptors(text: str) -> list[str]:
    """Extract up to two delivery adjectives from the provided text."""
    normalized = (text or "").strip().lower()
    descriptors: list[str] = []
    for descriptor in DELIVERY_DESCRIPTORS:
        if re.search(rf"\b{re.escape(descriptor)}\b", normalized) and descriptor not in descriptors:
            descriptors.append(descriptor)
        if len(descriptors) >= 2:
            break
    return descriptors


def caption_has_early_vocal_character_block(caption: str) -> bool:
    """Return whether the caption already establishes singer character near the start."""
    first_sentence, _ = split_first_sentence((caption or "").strip())
    return bool(first_sentence) and bool(
        GENDER_PATTERN.search(first_sentence) or extract_delivery_descriptors(first_sentence)
    )
