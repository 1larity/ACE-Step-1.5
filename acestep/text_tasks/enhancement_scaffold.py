"""Helpers to preserve arrangement/instrument scaffolds during text enhancement."""

from __future__ import annotations

import re


_ARRANGEMENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("intro", re.compile(r"\bintro\b", re.IGNORECASE)),
    ("verse", re.compile(r"\bverse\b", re.IGNORECASE)),
    ("pre-chorus", re.compile(r"\bpre[- ]chorus\b", re.IGNORECASE)),
    ("chorus", re.compile(r"\bchorus\b", re.IGNORECASE)),
    ("post-chorus", re.compile(r"\bpost[- ]chorus\b", re.IGNORECASE)),
    ("bridge", re.compile(r"\bbridge\b", re.IGNORECASE)),
    ("hook", re.compile(r"\bhook\b", re.IGNORECASE)),
    ("refrain", re.compile(r"\brefrain\b", re.IGNORECASE)),
    ("drop", re.compile(r"\bdrop\b", re.IGNORECASE)),
    ("breakdown", re.compile(r"\bbreakdown\b", re.IGNORECASE)),
    ("interlude", re.compile(r"\binterlude\b", re.IGNORECASE)),
    ("outro", re.compile(r"\boutro\b", re.IGNORECASE)),
]

_INSTRUMENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("woodwinds", re.compile(r"\bwoodwinds?\b", re.IGNORECASE)),
    ("brass", re.compile(r"\bbrass\b", re.IGNORECASE)),
    ("fx", re.compile(r"\bfx\b|\beffects?\b", re.IGNORECASE)),
    ("synth", re.compile(r"\bsynth(?:s|es|wave|esized|esizer)?\b", re.IGNORECASE)),
    ("strings", re.compile(r"\bstrings?\b", re.IGNORECASE)),
    ("percussion", re.compile(r"\bpercussion\b", re.IGNORECASE)),
    ("keyboard", re.compile(r"\bkeyboard\b|\bkeys\b", re.IGNORECASE)),
    ("piano", re.compile(r"\bpiano\b", re.IGNORECASE)),
    ("guitar", re.compile(r"\bguitar(?:s)?\b", re.IGNORECASE)),
    ("bass", re.compile(r"\bbass(?:line)?\b", re.IGNORECASE)),
    ("drums", re.compile(r"\bdrums?\b", re.IGNORECASE)),
    ("backing_vocals", re.compile(r"\bbacking vocals?\b", re.IGNORECASE)),
    ("vocals", re.compile(r"\bvocals?\b|\bvocalist\b|\bsinger\b", re.IGNORECASE)),
]

_SECTION_TAG_PATTERN = re.compile(r"\[([^\]]+)\]")


def _unique_in_order(values: list[str]) -> list[str]:
    """Return values deduplicated while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def extract_arrangement_tags(caption: str, lyrics: str) -> list[str]:
    """Extract arrangement tags from caption text and lyrics section headers."""
    combined = f"{caption or ''}\n{lyrics or ''}"
    extracted: list[str] = []

    for header in _SECTION_TAG_PATTERN.findall(lyrics or ""):
        for name, pattern in _ARRANGEMENT_RULES:
            if pattern.search(header):
                extracted.append(name)
                break

    for name, pattern in _ARRANGEMENT_RULES:
        if pattern.search(combined):
            extracted.append(name)

    return _unique_in_order(extracted)


def extract_instrument_tags(caption: str, lyrics: str) -> list[str]:
    """Extract instrument tags from caption/lyrics text."""
    combined = f"{caption or ''}\n{lyrics or ''}"
    extracted: list[str] = []
    for name, pattern in _INSTRUMENT_RULES:
        if pattern.search(combined):
            extracted.append(name)
    return extracted


def build_preservation_directives(caption: str, lyrics: str) -> str:
    """Build concise preservation constraints for enhancement prompts."""
    arrangement_tags = extract_arrangement_tags(caption=caption, lyrics=lyrics)
    instrument_tags = extract_instrument_tags(caption=caption, lyrics=lyrics)

    lines: list[str] = []
    if arrangement_tags:
        lines.append(
            "- Preserve arrangement tags exactly: "
            + ", ".join(arrangement_tags)
            + "."
        )
    if instrument_tags:
        lines.append(
            "- Preserve instrument tags exactly: "
            + ", ".join(instrument_tags)
            + "."
        )
    if not lines:
        return ""

    lines.append(
        "- You may enrich detail and transitions, but do not remove or contradict these tags."
    )
    return "\n".join(lines)

