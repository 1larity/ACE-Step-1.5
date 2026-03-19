"""External LM setup controls embedded in generation service settings."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_model_cache import load_cached_external_models
from acestep.text_tasks.external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_provider,
)
from acestep.text_tasks.external_lm_providers import (
    get_external_base_url_preset_choices,
    get_external_base_url_preset_value,
    get_external_provider_choices,
    get_external_provider_profile,
)
from acestep.text_tasks.external_lm_runtime_store import (
    hydrate_external_lm_env_from_store,
    load_external_lm_runtime_settings,
)


def create_external_lm_config_content(
    init_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the external-LM setup controls inside service configuration."""

    hydrate_external_lm_env_from_store()
    selected_provider = _resolve_initial_provider(init_params)
    provider_profile = get_external_provider_profile(selected_provider)
    saved_settings = load_external_lm_runtime_settings(selected_provider) or {}

    protocol_value = (
        os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL", "").strip()
        or saved_settings.get("protocol", "").strip()
        or provider_profile.protocol
    )
    model_value = (
        os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
        or saved_settings.get("model", "").strip()
        or get_active_external_lm_model(provider_profile.default_model)
    )
    base_url_value = (
        os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
        or saved_settings.get("base_url", "").strip()
        or provider_profile.default_base_url
    )
    base_url_preset_value = get_external_base_url_preset_value(
        selected_provider, base_url_value
    )
    model_choices = load_cached_external_models(
        provider=selected_provider,
        protocol=protocol_value,
        base_url=base_url_value,
    ) or [model_value]
    if model_value not in model_choices:
        model_choices = [model_value, *model_choices]

    with gr.Accordion("🧠 External LM", open=False):
        gr.Markdown(
            "Configure provider preferences, endpoint defaults, and encrypted "
            "credentials for external language-model workflows."
        )
        with gr.Row():
            external_lm_provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=get_external_provider_choices(),
                value=selected_provider,
                info="Select the external provider profile.",
                elem_classes=["has-info-container"],
            )
            external_lm_protocol_dropdown = gr.Dropdown(
                label="Protocol",
                choices=["openai_chat", "anthropic_messages"],
                value=protocol_value,
                info="Protocol expected by the provider endpoint.",
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            with gr.Column(scale=1):
                external_lm_model_input = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    value=model_value,
                    allow_custom_value=True,
                    info="Type or select the provider model identifier to surface in the main LM picker.",
                    elem_classes=["has-info-container"],
                )
                external_lm_fetch_models_btn = gr.Button("Get Models", size="sm")
            with gr.Column(scale=1):
                gr.Markdown(
                    "Try a coding model and coding endpoint if your subscription includes "
                    "coding access."
                )
                external_lm_base_url_preset_dropdown = gr.Dropdown(
                    label="Endpoint Preset",
                    choices=get_external_base_url_preset_choices(selected_provider),
                    value=base_url_preset_value,
                    info="Pick a common endpoint, or switch to Custom and edit the URL below.",
                    elem_classes=["has-info-container"],
                )
                external_lm_base_url_input = gr.Textbox(
                    label="Base URL",
                    value=base_url_value,
                    placeholder="Provider API endpoint URL",
                    info=(
                        "Examples include the standard chat endpoint and, for Z.ai users with "
                        "coding access, the coding endpoint."
                    ),
                )
        with gr.Row():
            external_lm_api_key_input = gr.Textbox(
                label="API Key",
                type="text",
                placeholder="Enter provider API key if required",
                info="Saved encrypted on disk when you also supply a passphrase.",
            )
            external_lm_store_passphrase_input = gr.Textbox(
                label="Store Passphrase",
                type="text",
                placeholder="Passphrase for encrypted key storage",
                info="Optionally stored in the system keyring on Linux, macOS, or Windows.",
            )
        with gr.Row():
            external_lm_save_passphrase_checkbox = gr.Checkbox(
                label="Save passphrase to system keyring",
                value=True,
            )
        with gr.Row():
            external_lm_save_btn = gr.Button(
                "Save External LM Settings",
                variant="primary",
            )
            external_lm_defaults_btn = gr.Button("Load Provider Defaults")
            external_lm_doctor_btn = gr.Button("Check External Runtime")
        external_lm_status = gr.Markdown(value="")

    return {
        "external_lm_provider_dropdown": external_lm_provider_dropdown,
        "external_lm_protocol_dropdown": external_lm_protocol_dropdown,
        "external_lm_model_input": external_lm_model_input,
        "external_lm_fetch_models_btn": external_lm_fetch_models_btn,
        "external_lm_base_url_preset_dropdown": external_lm_base_url_preset_dropdown,
        "external_lm_base_url_input": external_lm_base_url_input,
        "external_lm_api_key_input": external_lm_api_key_input,
        "external_lm_store_passphrase_input": external_lm_store_passphrase_input,
        "external_lm_save_passphrase_checkbox": external_lm_save_passphrase_checkbox,
        "external_lm_save_btn": external_lm_save_btn,
        "external_lm_defaults_btn": external_lm_defaults_btn,
        "external_lm_doctor_btn": external_lm_doctor_btn,
        "external_lm_status": external_lm_status,
    }


def _resolve_initial_provider(init_params: dict[str, Any] | None) -> str:
    """Resolve the initial provider from env or init params."""

    provider = os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip().lower()
    if provider:
        return provider

    model_path = (init_params or {}).get("lm_model_path")
    if isinstance(model_path, str) and model_path.startswith("external:"):
        tokens = model_path.split(":", 2)
        if len(tokens) >= 3:
            return tokens[1].strip().lower() or "zai"
    return get_active_external_lm_provider()
