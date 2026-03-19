"""LM picker synchronization helpers for the external LM setup panel."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_mode import (
    get_external_lm_choices,
    parse_external_lm_selection,
)
from acestep.text_tasks.external_lm_providers import (
    get_external_base_url_preset_choices,
    get_external_base_url_preset_value,
    get_external_provider_profile,
)
from acestep.text_tasks.external_lm_runtime_store import load_external_lm_runtime_settings

from .external_lm_setup_defaults import as_markdown_status


def build_external_lm_dropdown_sync_updates(
    lm_model_path: str,
) -> tuple[dict, dict, dict, dict, dict, str]:
    """Build UI updates when the main LM dropdown switches external providers."""

    selection = parse_external_lm_selection(lm_model_path)
    if not selection:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "")

    profile = get_external_provider_profile(selection.provider)
    saved_settings = load_external_lm_runtime_settings(selection.provider) or {}
    base_url_value = (
        saved_settings.get("base_url", "")
        or os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
        or profile.default_base_url
    )
    protocol_value = saved_settings.get("protocol", "") or profile.protocol
    model_value = saved_settings.get("model", "") or selection.model
    status = as_markdown_status(
        "\n".join(
            [
                f"Provider: {profile.label}",
                f"Protocol: {protocol_value}",
                f"Model: {model_value}",
                f"Base URL: {base_url_value}",
            ]
        )
    )
    return (
        gr.update(value=profile.provider_id),
        gr.update(value=protocol_value),
        gr.update(choices=[model_value], value=model_value),
        gr.update(
            choices=get_external_base_url_preset_choices(profile.provider_id),
            value=get_external_base_url_preset_value(profile.provider_id, base_url_value),
        ),
        gr.update(value=base_url_value),
        status,
    )


def build_external_lm_inactive_updates() -> tuple[dict, dict, dict, dict, dict, str]:
    """Build UI updates for when the main LM picker switches back to a local model."""

    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(value=""),
        as_markdown_status("External LM inactive: main LM picker is using a local model."),
    )


def build_lm_dropdown_choices(llm_handler: Any | None, lm_model_choice: str) -> list[str]:
    """Build the LM dropdown choice list including any external LM entry."""

    existing_choices = get_external_lm_choices()
    if llm_handler is not None:
        existing_choices = llm_handler.get_available_5hz_lm_models() + existing_choices
    return list(dict.fromkeys(existing_choices + [lm_model_choice]))
