"""Metadata parsing helpers for generation JSON imports."""

from __future__ import annotations

from typing import Any


def parse_optional_int(value: Any) -> int | None:
    """Parse an optional integer metadata field."""
    if value in (None, "N/A", ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_optional_float(value: Any, default: float | None = None) -> float | None:
    """Parse an optional float metadata field."""
    if value in (None, "N/A", ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def normalize_optional_text(value: Any, empty_values: set[Any] | None = None) -> str:
    """Normalize optional text metadata values to a safe string."""
    empties = empty_values or {None, "N/A"}
    if value in empties:
        return ""
    return value if isinstance(value, str) else (str(value) if value else "")


def clamp_batch_size(batch_size: Any, max_batch_size: int) -> int:
    """Clamp a batch size value to the current GPU/runtime limit."""
    try:
        return min(int(batch_size), max_batch_size)
    except Exception:
        return min(2, max_batch_size)
