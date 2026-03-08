"""External LM task adapters for create-sample and format-sample flows."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from .external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_protocol,
    get_active_external_lm_provider,
    resolve_external_api_key_for_runtime,
)
from .external_lm_providers import get_external_provider_profile
from .glm_text_tasks import GlmClientError, request_glm_plan
from .secure_secret_store import SecretStoreError


def _external_base_url(provider: str) -> str:
    """Return provider-aware endpoint URL for external provider calls."""
    profile = get_external_provider_profile(provider)
    generic = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    if generic:
        return generic

    if provider == "zai":
        zai_url = os.getenv("ACESTEP_GLM_BASE_URL", "").strip()
        if zai_url:
            return zai_url

    provider_specific_env = {
        "openai": "ACESTEP_OPENAI_BASE_URL",
        "ollama": "ACESTEP_OLLAMA_BASE_URL",
        "claude": "ACESTEP_ANTHROPIC_BASE_URL",
    }.get(provider)
    if provider_specific_env:
        configured = os.getenv(provider_specific_env, "").strip()
        if configured:
            return configured

    return profile.default_base_url


def _request_external_plan(intent: str, timeout_sec: int, task_focus: str):
    """Run provider-specific planning request and return parsed structured plan."""
    provider = get_active_external_lm_provider()
    model = get_active_external_lm_model()
    protocol = get_active_external_lm_protocol()
    profile = get_external_provider_profile(provider)

    try:
        api_key = resolve_external_api_key_for_runtime(provider)
    except SecretStoreError as exc:
        raise GlmClientError(str(exc)) from exc

    plan = request_glm_plan(
        api_key=api_key,
        intent=intent,
        model=model,
        base_url=_external_base_url(provider),
        timeout_sec=timeout_sec,
        task_focus=task_focus,
        protocol=protocol,
    )
    return plan, profile, model


def create_sample_with_external_provider(
    *,
    query: str,
    instrumental: bool,
    vocal_language: str,
    timeout_sec: int = 60,
) -> Any:
    """Return CreateSample-like result object via active external LM provider."""
    intent = query.strip() or "NO USER INPUT"
    intent += f"\n\ninstrumental: {'true' if instrumental else 'false'}"
    if vocal_language and vocal_language != "unknown":
        intent += f"\nvocal_language: {vocal_language}"

    plan, profile, model = _request_external_plan(
        intent=intent,
        timeout_sec=timeout_sec,
        task_focus="all",
    )

    lyrics = plan.lyrics or ("[Instrumental]" if (instrumental or plan.instrumental) else "")
    return SimpleNamespace(
        success=True,
        caption=plan.caption,
        lyrics=lyrics,
        bpm=plan.bpm,
        duration=plan.duration,
        keyscale=plan.key_scale,
        language=plan.vocal_language,
        timesignature=plan.time_signature,
        instrumental=instrumental or plan.instrumental,
        status_message=f"External {profile.label} sample created ({model})",
        error=None,
    )


def format_sample_with_external_provider(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
    timeout_sec: int = 60,
) -> Any:
    """Return FormatSample-like result object via active external LM provider."""
    intent_parts = [
        "Please format and enrich the following for ACE-Step generation.",
        f"Caption: {caption or ''}",
        f"Lyrics: {lyrics or ''}",
    ]
    if user_metadata:
        for key in ("bpm", "duration", "keyscale", "timesignature", "language"):
            value = user_metadata.get(key)
            if value not in (None, "", "unknown"):
                intent_parts.append(f"{key}: {value}")

    plan, profile, model = _request_external_plan(
        intent="\n".join(intent_parts),
        timeout_sec=timeout_sec,
        task_focus="format",
    )

    return SimpleNamespace(
        success=True,
        caption=plan.caption or caption,
        lyrics=plan.lyrics or lyrics,
        bpm=plan.bpm,
        duration=plan.duration,
        keyscale=plan.key_scale,
        language=plan.vocal_language,
        timesignature=plan.time_signature,
        status_message=f"External {profile.label} format completed ({model})",
        error=None,
    )


__all__ = [
    "GlmClientError",
    "create_sample_with_external_provider",
    "format_sample_with_external_provider",
]
