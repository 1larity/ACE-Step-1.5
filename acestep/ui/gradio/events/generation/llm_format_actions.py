"""LLM formatting action handlers for generation UI text fields."""
import random
from typing import Optional

import gradio as gr

from acestep.inference import format_sample
from acestep.text_tasks.external_lm_mode import is_external_lm_active
from acestep.text_tasks.external_lm_tasks import (
    GlmClientError,
    format_sample_with_external_provider,
)
from acestep.ui.gradio.i18n import t

from .llm_action_params import build_user_metadata, convert_lm_params
from .validation import clamp_duration_to_gpu_limit


_RANDOM_NARRATIVE_GENRES = (
    "cinematic orchestral pop",
    "neo-soul",
    "melodic drum and bass",
    "ambient folk",
    "dark synthwave",
    "afro-house",
    "indie rock ballad",
    "future garage",
    "latin pop",
    "lo-fi hip-hop",
)


def _format_failure_response(update_count: int, status_message: str):
    """Build a standardized failure response with update placeholders."""
    return (*([gr.update()] * update_count), status_message)


def _clean_optional_wrapped_quotes(text: Optional[str]) -> Optional[str]:
    """Strip a single layer of leading/trailing quote characters when present."""
    if text is None:
        return None
    if len(text) >= 2 and (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        return text[1:-1]
    return text


def _build_random_narrative_caption_seed() -> tuple[str, str]:
    """Return ``(genre, prompt)`` seed used for random caption-from-scratch generation."""
    genre = random.choice(_RANDOM_NARRATIVE_GENRES)
    prompt = (
        f"Create a brand-new {genre} music concept. "
        "Write a narrative caption with a clear linear structure "
        "(intro -> build -> chorus/drop -> outro), core instrumentation, "
        "vocalist timbre/delivery, and energy/mix trajectory."
    )
    return genre, prompt


def _execute_format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool,
    vocal_language: str = "unknown",
):
    """Run shared format-sample workflow.

    Returns:
        Tuple of ``(result_or_none, audio_duration_value_or_none, status_message)``.
    """
    external_active = is_external_lm_active()
    if not llm_handler.llm_initialized and not external_active:
        status_message = t("messages.lm_not_initialized")
        gr.Warning(status_message)
        return None, None, status_message

    user_metadata = build_user_metadata(
        bpm,
        audio_duration,
        key_scale,
        time_signature,
        vocal_language=vocal_language,
    )
    if external_active:
        try:
            result = format_sample_with_external_provider(
                caption=caption,
                lyrics=lyrics,
                user_metadata=user_metadata,
            )
        except GlmClientError as exc:
            status_message = str(exc)
            gr.Warning(status_message)
            return None, None, status_message
    else:
        top_k_value, top_p_value = convert_lm_params(lm_top_k, lm_top_p)
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            user_metadata=user_metadata,
            temperature=lm_temperature,
            top_k=top_k_value,
            top_p=top_p_value,
            use_constrained_decoding=True,
            constrained_decoding_debug=constrained_decoding_debug,
        )

    if not result.success:
        status_message = result.status_message or t("messages.format_failed")
        gr.Warning(status_message)
        return None, None, status_message

    gr.Info(t("messages.format_success"))
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    duration_value = clamped_duration if clamped_duration and clamped_duration > 0 else -1
    return result, duration_value, result.status_message


def handle_format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format caption and lyrics together via LLM."""
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=8, status_message=status_message)

    return (
        result.caption,
        result.lyrics,
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def handle_format_caption(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format only caption via LLM while leaving lyrics unchanged in UI wiring.

    Any outer single/double quotes added by the LLM are stripped from the
    returned caption for cleaner textbox display.
    """
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    return (
        _clean_optional_wrapped_quotes(result.caption),
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def handle_format_lyrics(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Format only lyrics via LLM while leaving caption unchanged in UI wiring.

    Any outer single/double quotes added by the LLM are stripped from the
    returned lyrics for cleaner textbox display.
    """
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    return (
        _clean_optional_wrapped_quotes(result.lyrics),
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        status_message,
    )


def handle_generate_lyrics_from_caption(
    llm_handler,
    caption: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    vocal_language: str = "unknown",
    lm_temperature: float = 0.85,
    lm_top_k: int = 0,
    lm_top_p: float = 0.9,
    constrained_decoding_debug: bool = False,
):
    """Generate fresh lyrics from caption while honoring selected metadata/language."""
    lyrics_seed = ""
    if vocal_language and vocal_language != "unknown":
        lyrics_seed = f"[Generate full lyrics in {vocal_language}]"

    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics_seed,
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    generated_status = "Lyrics generated from caption."
    if status_message:
        generated_status = f"{status_message} | {generated_status}"
    return (
        _clean_optional_wrapped_quotes(result.lyrics),
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        generated_status,
    )


def handle_generate_random_narrative_caption(
    llm_handler,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    vocal_language: str = "unknown",
):
    """Generate a caption from scratch using a random genre narrative seed."""
    genre, seed_prompt = _build_random_narrative_caption_seed()
    result, duration_value, status_message = _execute_format_sample(
        llm_handler=llm_handler,
        caption=seed_prompt,
        lyrics="",
        bpm=bpm,
        audio_duration=audio_duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        lm_temperature=lm_temperature,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if result is None:
        return _format_failure_response(update_count=7, status_message=status_message)

    random_status = f"{status_message} | Random genre seed: {genre}"
    return (
        _clean_optional_wrapped_quotes(result.caption),
        result.bpm,
        duration_value,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        random_status,
    )
