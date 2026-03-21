"""Debug helpers shared across external AI text-task modules."""

from __future__ import annotations

import re

from acestep.lm_task_debug import is_lm_task_debug_enabled


def is_external_ai_debug_enabled() -> bool:
    """Return ``True`` when verbose LM task logging is enabled."""
    return is_lm_task_debug_enabled()


def preview_text(text: str, limit: int = 600) -> str:
    """Return a compact single-line preview for logs and error messages."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."
