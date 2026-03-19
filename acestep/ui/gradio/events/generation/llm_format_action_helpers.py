"""Shared helpers for LM format actions in the generation UI."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from acestep.constants import DEBUG_LLM
from acestep.debug_utils import is_debug_enabled
from acestep.inference import format_sample
from acestep.text_tasks.external_lm_mode import is_external_lm_active
from acestep.text_tasks.external_lm_tasks import (
    GlmClientError,
    format_sample_with_external_provider,
)
from acestep.ui.gradio.i18n import t

from .llm_action_params import build_user_metadata, convert_lm_params
from .validation import clamp_duration_to_gpu_limit


def format_failure_response(update_count: int, status_message: str):
    """Build a standardized failure response with update placeholders."""

    return (*([gr.update()] * update_count), status_message)


def clean_optional_wrapped_quotes(text: Optional[str]) -> Optional[str]:
    """Strip a single layer of leading and trailing quotes when present."""

    if text is None:
        return None
    if len(text) >= 2 and (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        return text[1:-1]
    return text


def effective_llm_debug_enabled(constrained_decoding_debug: bool) -> bool:
    """Return whether terminal LM debug should be enabled for format actions."""

    return bool(constrained_decoding_debug) or is_debug_enabled(DEBUG_LLM)


def execute_format_sample(
    *,
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
):
    """Run the shared format-sample workflow and normalize its result tuple."""

    effective_debug = effective_llm_debug_enabled(constrained_decoding_debug)
    external_active = is_external_lm_active()
    if not llm_handler.llm_initialized and not external_active:
        status_message = t("messages.lm_not_initialized")
        gr.Warning(status_message)
        return None, None, status_message

    user_metadata = build_user_metadata(bpm, audio_duration, key_scale, time_signature)
    if external_active:
        try:
            result = format_sample_with_external_provider(
                caption=caption,
                lyrics=lyrics,
                user_metadata=user_metadata,
                debug=effective_debug,
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
            constrained_decoding_debug=effective_debug,
        )

    if not result.success:
        status_message = result.status_message or t("messages.format_failed")
        gr.Warning(status_message)
        return None, None, status_message

    gr.Info(t("messages.format_success"))
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    duration_value = clamped_duration if clamped_duration and clamped_duration > 0 else -1
    return result, duration_value, result.status_message


_effective_llm_debug_enabled = effective_llm_debug_enabled
