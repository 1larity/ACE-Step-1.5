"""LLM formatting action handlers for generation UI text fields."""

from .llm_format_action_helpers import (
    clean_optional_wrapped_quotes as _clean_optional_wrapped_quotes,
    effective_llm_debug_enabled as _effective_llm_debug_enabled,
    execute_format_sample as _execute_format_sample,
    format_failure_response as _format_failure_response,
)


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
