"""Minimal external LM adapters for format-sample flows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from .external_lm_captioning import (
    apply_user_metadata_overrides,
    build_fallback_caption,
    build_format_request_intent,
    caption_needs_retry,
)
from .external_lm_http_common import coerce_keep_alive_value
from .external_lm_plan_requests import request_external_plan
from .external_lm_warmup import warm_up_external_provider


from .external_ai_types import ExternalAIClientError


GlmClientError = ExternalAIClientError

_caption_needs_retry = caption_needs_retry
_apply_user_metadata_overrides = apply_user_metadata_overrides
_build_fallback_caption = build_fallback_caption
_coerce_keep_alive_value = coerce_keep_alive_value
_request_external_plan = request_external_plan


def format_sample_with_external_provider(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
    timeout_sec: int = 60,
    debug: bool = False,
) -> Any:
    """Return FormatSample-like result object via active external LM provider."""

    request_intent = build_format_request_intent(
        caption=caption,
        lyrics=lyrics,
        user_metadata=user_metadata,
    )
    plan, profile, model = _request_external_plan(
        intent=request_intent,
        timeout_sec=timeout_sec,
        task_focus="format",
        debug=debug,
    )
    if caption_needs_retry(original_caption=caption, generated_caption=plan.caption):
        retry_intent = (
            request_intent
            + "\n\nRetry instruction: the previous caption was unchanged or too short. "
            + "Rewrite it into a fuller linear narrative caption and do not echo the input wording."
        )
        try:
            plan, profile, model = _request_external_plan(
                intent=retry_intent,
                timeout_sec=timeout_sec,
                task_focus="format",
                debug=debug,
            )
        except ExternalAIClientError:
            plan.caption = build_fallback_caption(caption=caption, user_metadata=user_metadata)
    if caption_needs_retry(original_caption=caption, generated_caption=plan.caption):
        plan.caption = build_fallback_caption(caption=caption, user_metadata=user_metadata)
    plan = apply_user_metadata_overrides(plan=plan, user_metadata=user_metadata)
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


def create_sample_with_external_provider(
    *,
    query: str,
    instrumental: bool,
    vocal_language: str,
    timeout_sec: int = 60,
    debug: bool = False,
) -> Any:
    """Return CreateSample-like result object via the active external provider."""

    intent = query.strip() or "NO USER INPUT"
    intent += f"\n\ninstrumental: {'true' if instrumental else 'false'}"
    if vocal_language and vocal_language != "unknown":
        intent += f"\nvocal_language: {vocal_language}"

    plan, profile, model = _request_external_plan(
        intent=intent,
        timeout_sec=timeout_sec,
        task_focus="all",
        debug=debug,
    )
    resolved_instrumental = _resolve_instrumental_flag(
        requested_instrumental=instrumental,
        provider_instrumental=plan.instrumental,
    )
    lyrics = plan.lyrics or ("[Instrumental]" if resolved_instrumental else "")
    return SimpleNamespace(
        success=True,
        caption=plan.caption,
        lyrics=lyrics,
        bpm=plan.bpm,
        duration=plan.duration,
        keyscale=plan.key_scale,
        language=plan.vocal_language,
        timesignature=plan.time_signature,
        instrumental=resolved_instrumental,
        status_message=f"External {profile.label} sample created ({model})",
        error=None,
    )


def _resolve_instrumental_flag(
    *,
    requested_instrumental: bool,
    provider_instrumental: bool,
) -> bool:
    """Preserve an explicit user vocal request over provider instrumental drift."""

    _ = provider_instrumental
    return bool(requested_instrumental)
