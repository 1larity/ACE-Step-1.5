"""Audio and instrumental UI helpers for generation handlers."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t

from .validation import _has_reference_audio


def update_audio_cover_strength_visibility(task_type_value, init_llm_checked, reference_audio=None):
    """Update audio_cover_strength visibility and label."""
    has_reference = _has_reference_audio(reference_audio)
    is_visible = (task_type_value == "cover") or init_llm_checked or has_reference
    if task_type_value == "cover":
        label = t("generation.cover_strength_label")
        help_text = t("generation.cover_strength_info")
    elif init_llm_checked:
        label = t("generation.codes_strength_label")
        help_text = t("generation.codes_strength_info")
    elif has_reference:
        label = t("generation.similarity_denoise_label")
        help_text = t("generation.similarity_denoise_info")
    else:
        label = t("generation.cover_strength_label")
        help_text = t("generation.cover_strength_info")
    return gr.update(visible=is_visible, label=label, info=help_text, elem_classes=["has-info-container"])


def convert_src_audio_to_codes_wrapper(dit_handler, src_audio):
    """Wrapper for converting src audio to codes."""
    return dit_handler.convert_src_audio_to_codes(src_audio)


def update_transcribe_button_text(audio_code_string):
    """Update the transcribe button text based on input content."""
    return gr.update(value="Generate Example" if not audio_code_string or not audio_code_string.strip() else "Transcribe")


def update_audio_uploads_accordion(reference_audio, src_audio):
    """Update Audio Uploads accordion open state based on whether audio files are present."""
    return gr.Accordion(open=(reference_audio is not None) or (src_audio is not None))


def handle_instrumental_checkbox(instrumental_checked, current_lyrics, saved_lyrics):
    """Handle instrumental checkbox changes."""
    if instrumental_checked:
        return "[Instrumental]", current_lyrics
    return (saved_lyrics if saved_lyrics else ""), ""


def handle_simple_instrumental_change(is_instrumental: bool):
    """Handle simple mode instrumental checkbox changes."""
    if is_instrumental:
        return gr.update(value="unknown", interactive=False)
    return gr.update(interactive=True)


def update_audio_components_visibility(batch_size):
    """Show/hide individual audio components based on batch size (1-8)."""
    try:
        batch_size = min(max(int(batch_size or 1), 1), 8)
    except (TypeError, ValueError):
        batch_size = 1
    updates_row1 = (
        gr.update(visible=True),
        gr.update(visible=batch_size >= 2),
        gr.update(visible=batch_size >= 3),
        gr.update(visible=batch_size >= 4),
    )
    show_row_5_8 = batch_size >= 5
    updates_row2 = (
        gr.update(visible=show_row_5_8),
        gr.update(visible=batch_size >= 5),
        gr.update(visible=batch_size >= 6),
        gr.update(visible=batch_size >= 7),
        gr.update(visible=batch_size >= 8),
    )
    return updates_row1 + updates_row2
