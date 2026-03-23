"""External LLM configuration builders for generation settings."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_provider,
)
from acestep.text_tasks.external_lm_runtime_store import hydrate_external_lm_env_from_store
from acestep.text_tasks.external_lm_providers import (
    get_external_provider_choices,
    get_external_provider_profile,
)


def create_external_lm_config_content(init_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the dedicated External LLM settings accordion content."""
    selected_provider = _resolve_initial_provider(init_params)
    hydrate_external_lm_env_from_store(selected_provider)
    provider_profile = get_external_provider_profile(selected_provider)

    protocol_value = (
        os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL", "").strip()
        or provider_profile.protocol
    )
    model_value = (
        os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
        or get_active_external_lm_model(provider_profile.default_model)
    )
    base_url_value = (
        os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
        or provider_profile.default_base_url
    )

    with gr.Accordion("🧠 External LM", open=False, elem_classes=["has-info-container"]):
        gr.Markdown(
            "Configure provider, protocol, model, endpoint, and credentials for external "
            "language tasks (sample/format/Think-CoT bridge)."
        )
        with gr.Row():
            external_lm_provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=get_external_provider_choices(),
                value=selected_provider,
                info="Select provider profile. Values update defaults below.",
                elem_classes=["has-info-container"],
            )
            external_lm_protocol_dropdown = gr.Dropdown(
                label="Protocol",
                choices=["openai_chat", "anthropic_messages"],
                value=protocol_value,
                info="API protocol expected by the selected endpoint.",
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            with gr.Column(scale=1):
                external_lm_model_input = gr.Dropdown(
                    label="Model",
                    choices=[model_value],
                    value=model_value,
                    allow_custom_value=True,
                    info="Pick discovered models or type a custom model ID.",
                    elem_classes=["has-info-container"],
                )
                external_lm_fetch_models_btn = gr.Button("Fetch Models", size="sm")
            with gr.Column(scale=1):
                external_lm_base_url_input = gr.Dropdown(
                    label="Base URL",
                    choices=list(provider_profile.base_url_presets),
                    value=base_url_value,
                    allow_custom_value=True,
                    info="Choose a provider preset or type a custom chat endpoint URL.",
                    elem_classes=["has-info-container"],
                )
        with gr.Row():
            external_lm_api_key_input = gr.Textbox(
                label="API Key",
                type="text",
                placeholder="Enter provider API key (if required)",
                info="Visible while typing. Saved encrypted on disk when passphrase is provided.",
            )
            external_lm_store_passphrase_input = gr.Textbox(
                label="Store Passphrase",
                type="text",
                placeholder="Passphrase for encrypted key storage",
                info="Used to encrypt/decrypt provider key files.",
            )
        with gr.Row():
            external_lm_save_passphrase_checkbox = gr.Checkbox(
                label="Save passphrase to system keyring",
                value=True,
            )
        with gr.Row():
            external_lm_save_btn = gr.Button("Save External LLM Settings", variant="primary")
            external_lm_defaults_btn = gr.Button("Load Provider Defaults")
            external_lm_doctor_btn = gr.Button("Check External Runtime")
        external_lm_status = gr.Markdown(value="")

    return {
        "external_lm_provider_dropdown": external_lm_provider_dropdown,
        "external_lm_protocol_dropdown": external_lm_protocol_dropdown,
        "external_lm_model_input": external_lm_model_input,
        "external_lm_base_url_input": external_lm_base_url_input,
        "external_lm_api_key_input": external_lm_api_key_input,
        "external_lm_store_passphrase_input": external_lm_store_passphrase_input,
        "external_lm_save_passphrase_checkbox": external_lm_save_passphrase_checkbox,
        "external_lm_save_btn": external_lm_save_btn,
        "external_lm_defaults_btn": external_lm_defaults_btn,
        "external_lm_fetch_models_btn": external_lm_fetch_models_btn,
        "external_lm_doctor_btn": external_lm_doctor_btn,
        "external_lm_status": external_lm_status,
    }


def _resolve_initial_provider(init_params: dict[str, Any] | None) -> str:
    """Resolve initial provider from init params, env, or active runtime."""
    model_path = (init_params or {}).get("lm_model_path")
    if isinstance(model_path, str) and model_path.startswith("external:"):
        token = model_path.split(":", 2)
        if len(token) >= 3:
            return token[1].strip().lower() or "zai"

    provider = os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip().lower()
    if provider:
        return provider

    return get_active_external_lm_provider()
