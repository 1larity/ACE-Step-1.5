"""Task-level output contracts for external LM requests."""

from __future__ import annotations


def build_lyrics_output_contract() -> list[str]:
    """Return generic lyric-output rules for nondeterministic external LM providers."""
    return [
        "Return plain text lyrics only.",
        "Do not return JSON, Python lists, dictionaries, arrays, YAML, or XML.",
        "Do not wrap sections inside objects like {'section': ..., 'lines': ...}.",
        "Write the final answer exactly as song sections with lyric lines beneath each header.",
        "Example format: [Verse 1] then lyric lines, [Chorus] then lyric lines, [Verse 2] then lyric lines.",
        "Keep each sung lyric line short enough to sing naturally, ideally 20 syllables or fewer.",
        "If a phrase would run past a comma or pause, break it into a new lyric line instead of one long sentence.",
    ]
