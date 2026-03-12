"""Shared debug switch helpers for local and external LM task logging."""

from __future__ import annotations

import os

from acestep.constants import DEBUG_EXTERNAL_AI, DEBUG_LLM

_TRUTHY_VALUES = {"1", "true", "yes", "on", "enabled", "verbose"}


def _is_truthy_debug_value(value: str) -> bool:
    """Return ``True`` when a debug config string enables logging."""
    return (value or "").strip().lower() in _TRUTHY_VALUES



def is_lm_task_debug_enabled() -> bool:
    """Return ``True`` when LM task prompt/response logging is globally enabled."""
    env_value = os.getenv("ACESTEP_DEBUG_LM_TASKS", "").strip()
    if env_value:
        return _is_truthy_debug_value(env_value)

    external_ai_env = os.getenv("ACESTEP_EXTERNAL_AI_DEBUG", "").strip()
    if external_ai_env:
        return _is_truthy_debug_value(external_ai_env)

    return _is_truthy_debug_value(DEBUG_LLM) or _is_truthy_debug_value(DEBUG_EXTERNAL_AI)
