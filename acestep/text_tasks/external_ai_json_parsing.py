"""JSON and lyrics parsing helpers for external AI text-task responses."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from .external_ai_debug import is_external_ai_debug_enabled, preview_text
from .external_ai_types import ExternalAIClientError


def extract_plain_lyrics_content(content: str) -> str | None:
    """Return plain lyric text for lyrics-focused providers that ignore JSON instructions."""
    normalized = (content or "").strip()
    if not normalized:
        return None
    if normalized.lower() in {"instrumental", "[instrumental]"}:
        return "[Instrumental]"

    section_pattern = re.compile(
        r"(?im)^\s*(?:\[(?:verse|chorus|bridge|pre-chorus|intro|outro|hook|refrain|drop)[^\]]*\]|(?:verse\s*\d+|chorus|bridge|pre-chorus|intro|outro|hook|refrain|drop)\s*:)"
    )
    if not section_pattern.search(normalized):
        return None
    if "{" in normalized or "}" in normalized:
        return None
    return normalized


def load_plan_json_object(content: str, task_focus: str = "all") -> dict[str, Any]:
    """Load the best JSON object candidate from provider text content."""
    last_error: json.JSONDecodeError | None = None
    candidates = iter_json_candidates(content)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed

    content_preview = preview_text(content)
    if is_external_ai_debug_enabled():
        candidate_previews = [preview_text(candidate, limit=240) for candidate in candidates[:5]]
        logger.debug(
            "External AI JSON parse failure content_preview={} candidates={} last_error={}",
            content_preview,
            candidate_previews,
            last_error,
        )
    normalized_focus = (task_focus or "all").strip().lower()
    strict_hint = ""
    if normalized_focus != "lyrics":
        strict_hint = f" Task focus '{normalized_focus}' requires valid JSON."
    raise ExternalAIClientError(
        f"External AI content is not valid JSON.{strict_hint} Content preview: {content_preview}"
    ) from last_error


def iter_json_candidates(content: str) -> list[str]:
    """Return de-duplicated JSON candidates from wrapped model content."""
    normalized = normalize_model_content(content)
    candidates: list[str] = []
    for candidate in [extract_json_block(normalized), *extract_balanced_json_objects(normalized)]:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        candidates.append(cleaned)
        repaired = repair_json_candidate(cleaned)
        if repaired != cleaned:
            candidates.append(repaired)
    return list(dict.fromkeys(candidates))


def normalize_model_content(content: str) -> str:
    """Strip common reasoning wrappers before JSON extraction."""
    normalized = (content or "").strip().lstrip("ï»¿")
    normalized = re.sub(r"<think>.*?</think>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    normalized = re.sub(r"<analysis>.*?</analysis>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    return normalized.strip()


def extract_balanced_json_objects(content: str) -> list[str]:
    """Extract balanced top-level JSON object candidates from free-form text."""
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(content):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(content[start : index + 1])
                start = None
    return objects


def repair_json_candidate(candidate: str) -> str:
    """Apply small non-destructive repairs for common provider JSON defects."""
    repaired = candidate.strip()
    repaired = repaired.replace("â€œ", '"').replace("â€", '"')
    repaired = repaired.replace("â€˜", "'").replace("â€™", "'")
    repaired = repaired.replace("Â ", " ")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def extract_json_block(content: str) -> str:
    """Extract JSON object string from plain or markdown-fenced content."""
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1)

    first = content.find("{")
    last = content.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return content
    return content[first : last + 1]


def to_bool(value: Any) -> bool:
    """Coerce common bool-like values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def to_int(value: Any) -> int | None:
    """Coerce optional integer field."""
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    """Coerce optional float field."""
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
