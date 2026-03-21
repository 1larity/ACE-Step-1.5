"""Minimal external LM adapters for format-sample flows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .external_ai_types import ExternalAIClientError
from .external_lm_captioning import (
    apply_user_metadata_overrides,
    build_fallback_caption,
    build_format_request_intent,
    caption_needs_retry,
)
from .external_lm_plan_requests import request_external_plan
from .external_lm_warmup import warm_up_external_provider


GlmClientError = ExternalAIClientError


def _build_lyrics_generation_intent(
    *,
    caption: str,
    bpm: Any,
    audio_duration: Any,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    retry: bool,
) -> str:
    """Build a compact lyrics-generation request for the active external provider."""

    intent_lines = [
        (caption or "").strip() or "NO USER INPUT",
        "",
        "Write lead-vocal lyrics for this music concept.",
        "Return singable lyrics with a [Verse 1], [Chorus], [Verse 2] structure.",
        "Do not return [Instrumental], stage directions, empty tags, or placeholder scaffolds.",
    ]
    if bpm not in (None, "", 0):
        intent_lines.append(f"Preferred BPM: {bpm}")
    if audio_duration not in (None, "", 0):
        intent_lines.append(f"Preferred duration seconds: {audio_duration}")
    if key_scale:
        intent_lines.append(f"Preferred key: {key_scale}")
    if time_signature:
        intent_lines.append(f"Preferred time signature: {time_signature}")
    if vocal_language and vocal_language != "unknown":
        intent_lines.append(f"Preferred vocal language: {vocal_language}")
    if retry:
        intent_lines.append("Use a different hook and imagery from the previous draft.")
    return "\n".join(intent_lines)


def format_sample_with_external_provider(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
    timeout_sec: int = 60,
    debug: bool = False,
) -> Any:
    """Return a FormatSample-like result object via the active external provider."""

    request_intent = build_format_request_intent(
        caption=caption,
        lyrics=lyrics,
        user_metadata=user_metadata,
    )
    plan, profile, model = request_external_plan(
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
            plan, profile, model = request_external_plan(
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
    """Return a CreateSample-like result object via the active external provider."""

    intent = query.strip() or "NO USER INPUT"
    intent += f"\n\ninstrumental: {'true' if instrumental else 'false'}"
    if vocal_language and vocal_language != "unknown":
        intent += f"\nvocal_language: {vocal_language}"

    plan, profile, model = request_external_plan(
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


def generate_lyrics_from_caption_with_external_provider(
    *,
    caption: str,
    bpm: Any,
    audio_duration: Any,
    key_scale: str,
    time_signature: str,
    vocal_language: str,
    timeout_sec: int = 60,
    debug: bool = False,
    retry: bool = False,
) -> Any:
    """Return a lyric-generation result object via the active external provider."""

    plan, profile, model = request_external_plan(
        intent=_build_lyrics_generation_intent(
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
        debug=debug,
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


def _resolve_instrumental_flag(
    *,
    requested_instrumental: bool,
    provider_instrumental: bool,
) -> bool:
    """Preserve an explicit user vocal request over provider instrumental drift."""

    _ = provider_instrumental
    return bool(requested_instrumental)
