"""Shared support helpers for external LM setup actions."""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any, Callable

from acestep.text_tasks.external_lm_providers import get_external_provider_profile


def as_markdown_status(text: str) -> str:
    """Render multiline status text in a readable monospaced block."""
    safe_text = (text or "").strip() or "No status."
    return f"```text\n{safe_text}\n```"


def build_lm_dropdown_choices(
    llm_handler: Any | None,
    get_external_choices: Callable[[], list[str]],
) -> list[str] | None:
    """Build LM dropdown choices using local 5Hz models plus configured external models."""
    if llm_handler is None:
        return None
    local_models = llm_handler.get_available_5hz_lm_models() or []
    return list(dict.fromkeys(local_models + get_external_choices()))


def build_runtime_summary_line(
    provider: str,
    resolve_external_api_key_for_runtime: Callable[[str], str],
    secret_error_cls: type[Exception],
) -> str:
    """Return concise runtime readiness summary line after save."""
    profile = get_external_provider_profile(provider)
    if not profile.api_key_required:
        return "External runtime status: ready"
    try:
        key = resolve_external_api_key_for_runtime(provider)
    except secret_error_cls:
        return "External runtime status: passphrase/API key not yet available for non-interactive runtime"
    if key:
        return "External runtime status: ready"
    return "External runtime status: passphrase/API key not yet available for non-interactive runtime"


def secret_tool_available() -> bool:
    """Return whether `secret-tool` is available on PATH."""
    return bool(shutil.which("secret-tool"))


def python_keyring_available() -> bool:
    """Return whether python keyring module is installed."""
    return bool(importlib.util.find_spec("keyring"))
