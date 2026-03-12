"""External LM task adapters for create-sample and format-sample flows."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from .external_ai_text_tasks import ExternalAIClientError, request_external_ai_plan
from .external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_protocol,
    get_active_external_lm_provider,
    resolve_external_api_key_for_runtime,
)
from .external_lm_providers import get_external_provider_profile
from .external_lm_task_intents import (
    build_create_sample_intent,
    build_format_sample_intent,
    build_lyrics_generation_intent,
)
from .secure_secret_store import SecretStoreError


def _external_base_url(provider: str) -> str:
    """Return provider-aware endpoint URL for external provider calls."""
    profile = get_external_provider_profile(provider)
    generic = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    if generic:
        return generic
    if provider == "zai":
        zai_url = os.getenv("ACESTEP_ZAI_BASE_URL", "").strip()
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
    """Run provider-specific planning request and return a structured external plan."""
    provider = get_active_external_lm_provider()
    model = get_active_external_lm_model()
    protocol = get_active_external_lm_protocol()
    profile = get_external_provider_profile(provider)
    try:
        api_key = resolve_external_api_key_for_runtime(provider)
    except SecretStoreError as exc:
        raise ExternalAIClientError(str(exc)) from exc

    plan = request_external_ai_plan(
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
    timeout_sec: int = 120,
) -> Any:
    """Return CreateSample-like result object via active external LM provider."""
    plan, profile, model = _request_external_plan(
        intent=build_create_sample_intent(
            query=query,
            instrumental=instrumental,
            vocal_language=vocal_language,
        ),
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


def generate_lyrics_from_caption_with_external_provider(
    *,
    caption: str,
    bpm: Any,
    audio_duration: Any,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    timeout_sec: int = 120,
    retry: bool = False,
) -> Any:
    """Return lyric-generation result object via active external LM provider."""
    plan, profile, model = _request_external_plan(
        intent=build_lyrics_generation_intent(
            caption=caption,
            bpm=bpm,
            audio_duration=audio_duration,
            key_scale=key_scale,
            time_signature=time_signature,
            vocal_language=vocal_language,
            retry=retry,
        ),
        timeout_sec=timeout_sec,
        task_focus="lyrics",
    )
    return SimpleNamespace(
        success=True,
        caption=plan.caption or caption,
        lyrics=plan.lyrics,
        bpm=plan.bpm,
        duration=plan.duration,
        keyscale=plan.key_scale,
        language=plan.vocal_language,
        timesignature=plan.time_signature,
        status_message=f"External {profile.label} lyrics generated ({model})",
        error=None,
    )


def format_sample_with_external_provider(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
    timeout_sec: int = 120,
) -> Any:
    """Return FormatSample-like result object via active external LM provider."""
    plan, profile, model = _request_external_plan(
        intent=build_format_sample_intent(
            caption=caption,
            lyrics=lyrics,
            user_metadata=user_metadata,
        ),
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
    "ExternalAIClientError",
    "create_sample_with_external_provider",
    "format_sample_with_external_provider",
    "generate_lyrics_from_caption_with_external_provider",
]
