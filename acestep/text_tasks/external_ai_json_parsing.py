"""JSON parsing helpers for external AI text-task responses."""

from __future__ import annotations

import json
import re
from typing import Any

from .external_ai_types import ExternalAIClientError


def load_plan_json_object(content: str, task_focus: str = "all") -> dict[str, Any]:
    """Load the best JSON object candidate from provider text content."""

    last_error: json.JSONDecodeError | None = None
    for candidate in iter_json_candidates(content):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed

    labelled_fields = extract_labelled_plan_fields(content)
    if labelled_fields:
        return labelled_fields

    raise ExternalAIClientError(
        f"External AI content is not valid JSON for task focus '{task_focus}'."
    ) from last_error


def iter_json_candidates(content: str) -> list[str]:
    """Return de-duplicated JSON candidates from free-form content."""

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

    normalized = (content or "").strip().lstrip("\ufeff")
    normalized = re.sub(r"<think>.*?</think>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    normalized = re.sub(r"<analysis>.*?</analysis>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    return normalized.strip()


def extract_balanced_json_objects(content: str) -> list[str]:
    """Extract balanced top-level JSON object candidates."""

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
    """Apply small non-destructive repairs for common JSON defects."""

    repaired = candidate.strip()
    repaired = repaired.replace("â€œ", '"').replace("â€", '"')
    repaired = repaired.replace("â€˜", "'").replace("â€™", "'")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def extract_json_block(content: str) -> str:
    """Extract a likely JSON object from plain or fenced content."""

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


def extract_labelled_plan_fields(content: str) -> dict[str, Any]:
    """Extract plan fields from plain labelled text when JSON parsing fails."""

    normalized = normalize_model_content(content)
    field_map = {
        "caption": "caption",
        "lyrics": "lyrics",
        "bpm": "bpm",
        "duration": "duration",
        "key_scale": "key_scale",
        "key scale": "key_scale",
        "time_signature": "time_signature",
        "time signature": "time_signature",
        "vocal_language": "vocal_language",
        "vocal language": "vocal_language",
        "instrumental": "instrumental",
    }
    pattern = re.compile(
        r"^\s*(caption|lyrics|bpm|duration|key[_ ]scale|time[_ ]signature|"
        r"vocal[_ ]language|instrumental)\s*:\s*(.+?)(?=^\s*(?:caption|lyrics|bpm|"
        r"duration|key[_ ]scale|time[_ ]signature|vocal[_ ]language|instrumental)\s*:|\Z)",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    parsed: dict[str, Any] = {}
    for match in pattern.finditer(normalized):
        raw_key = match.group(1).strip().lower().replace("-", " ")
        key = field_map.get(raw_key)
        if not key:
            continue
        value = match.group(2).strip().strip("`")
        if value:
            parsed[key] = value
    return parsed


def to_bool(value: Any) -> bool:
    """Coerce bool-like values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def to_int(value: Any) -> int | None:
    """Coerce optional integer values."""

    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    """Coerce optional float values."""

    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
