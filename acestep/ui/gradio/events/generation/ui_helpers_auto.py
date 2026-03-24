"""Auto-field helpers for generation UI state."""

from __future__ import annotations

import gradio as gr

AUTO_DEFAULTS = {
    "bpm": None,
    "key_scale": "",
    "time_signature": "",
    "vocal_language": "unknown",
    "audio_duration": -1,
}


def on_auto_checkbox_change(auto_checked: bool, field_name: str):
    """Toggle a field between auto and manual."""
    if auto_checked:
        return gr.update(value=AUTO_DEFAULTS[field_name], interactive=False)
    return gr.update(interactive=True)


def reset_all_auto():
    """Reset all optional-parameter Auto checkboxes to checked."""
    return (
        gr.update(value=True),
        gr.update(value=True),
        gr.update(value=True),
        gr.update(value=True),
        gr.update(value=True),
        gr.update(value=AUTO_DEFAULTS["bpm"], interactive=False),
        gr.update(value=AUTO_DEFAULTS["key_scale"], interactive=False),
        gr.update(value=AUTO_DEFAULTS["time_signature"], interactive=False),
        gr.update(value=AUTO_DEFAULTS["vocal_language"], interactive=False),
        gr.update(value=AUTO_DEFAULTS["audio_duration"], interactive=False),
    )


def uncheck_auto_for_populated_fields(bpm, key_scale, time_signature, vocal_language, audio_duration):
    """Uncheck Auto checkboxes for fields populated by external events."""
    bpm_has_value = bpm is not None and bpm != AUTO_DEFAULTS["bpm"]
    key_has_value = bool(key_scale and key_scale != AUTO_DEFAULTS["key_scale"])
    ts_has_value = bool(time_signature and time_signature != AUTO_DEFAULTS["time_signature"])
    vl_has_value = vocal_language not in (None, "", AUTO_DEFAULTS["vocal_language"])
    dur_has_value = audio_duration is not None and audio_duration != AUTO_DEFAULTS["audio_duration"] and audio_duration > 0
    return (
        gr.update(value=not bpm_has_value),
        gr.update(value=not key_has_value),
        gr.update(value=not ts_has_value),
        gr.update(value=not vl_has_value),
        gr.update(value=not dur_has_value),
        gr.update(interactive=bpm_has_value),
        gr.update(interactive=key_has_value),
        gr.update(interactive=ts_has_value),
        gr.update(interactive=vl_has_value),
        gr.update(interactive=dur_has_value),
    )
