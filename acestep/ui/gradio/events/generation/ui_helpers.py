"""Small UI toggle/helper functions for generation handlers.

Contains functions for visibility toggles, auto-checkbox management,
instrumental handling, and other lightweight UI helpers.
"""

from __future__ import annotations

import gradio as gr
from typing import Optional

from acestep.constants import VALID_LANGUAGES
from acestep.ui.gradio.i18n import get_i18n

from .ui_helpers_audio import (
    convert_src_audio_to_codes_wrapper,
    handle_instrumental_checkbox,
    handle_simple_instrumental_change,
    update_audio_components_visibility,
    update_audio_cover_strength_visibility,
    update_audio_uploads_accordion,
    update_transcribe_button_text,
)
from .ui_helpers_auto import on_auto_checkbox_change, reset_all_auto, uncheck_auto_for_populated_fields


def update_negative_prompt_visibility(init_llm_checked):
    """Update negative prompt visibility: show if Initialize 5Hz LM checkbox is checked."""
    return gr.update(visible=init_llm_checked)


def sync_vocal_language_after_lyrics_generation(lyrics, vocal_language):
    """Force the UI into a vocal-language state when generated lyrics contain vocals."""
    stripped_lyrics = (lyrics or "").strip()
    if not stripped_lyrics or stripped_lyrics.lower() == "[instrumental]":
        return gr.update(), gr.update(), gr.update()
    normalized_language = (vocal_language or "").strip().lower()
    if normalized_language not in VALID_LANGUAGES or normalized_language == "unknown":
        ui_language = getattr(get_i18n(), "current_language", "")
        normalized_ui_language = (ui_language or "").strip().lower()
        normalized_language = normalized_ui_language if normalized_ui_language in VALID_LANGUAGES and normalized_ui_language != "unknown" else "en"
    return gr.update(value=False), gr.update(value=False), gr.update(value=normalized_language, interactive=True)


def update_instruction_ui(
    dit_handler,
    task_type_value: str,
    track_name_value: Optional[str],
    complete_track_classes_value: list,
    init_llm_checked: bool = False,
    reference_audio=None,
) -> tuple:
    """Update instruction text based on task type."""
    return dit_handler.generate_instruction(
        task_type=task_type_value,
        track_name=track_name_value,
        complete_track_classes=complete_track_classes_value,
    )


def reset_format_caption_flag():
    """Reset is_format_caption to False when user manually edits caption/metadata."""
    return False
