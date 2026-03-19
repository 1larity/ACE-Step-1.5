"""Persistence and discovery actions for the external LM setup panel."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_model_discovery import (
    ExternalModelDiscoveryError,
    discover_external_models,
)
from acestep.text_tasks.external_lm_model_cache import (
    invalidate_cached_external_models,
    save_cached_external_models,
)
from acestep.text_tasks.external_lm_mode import resolve_external_api_key_for_runtime
from acestep.text_tasks.external_lm_providers import (
    build_external_model_choice,
    get_external_provider_profile,
)
from acestep.text_tasks.external_lm_runtime_store import save_external_lm_runtime_settings
from acestep.text_tasks.secure_secret_store import SecretStoreError

from .external_lm_setup_credentials import store_provider_credentials
from .external_lm_setup_defaults import as_markdown_status, hydrate_external_env
from .external_lm_setup_sync import build_lm_dropdown_choices


def save_external_lm_settings_from_ui(
    provider: str,
    protocol: str,
    model: str,
    base_url: str,
    api_key: str,
    store_passphrase: str,
    save_passphrase_to_keyring: bool,
    llm_handler: Any | None = None,
) -> tuple[str, dict, dict, dict]:
    """Persist external LM settings and surface them in the main LM picker."""

    profile = get_external_provider_profile(provider)
    provider_id = profile.provider_id
    model_value = (model or "").strip() or profile.default_model
    protocol_value = (protocol or "").strip() or profile.protocol
    base_url_value = (base_url or "").strip() or profile.default_base_url
    api_key_value = (api_key or "").strip()
    passphrase_value = (store_passphrase or "").strip()

    status_lines = [
        f"Provider set to: {profile.label}",
        f"Protocol: {protocol_value}",
        f"Model: {model_value}",
        f"Base URL: {base_url_value}",
    ]

    maybe_error = store_provider_credentials(
        profile=profile,
        api_key_value=api_key_value,
        passphrase_value=passphrase_value,
        save_passphrase_to_keyring=save_passphrase_to_keyring,
        status_lines=status_lines,
    )
    if maybe_error is not None:
        return maybe_error

    invalidate_cached_external_models(provider=provider_id)

    hydrate_external_env(
        provider_id=provider_id,
        protocol_value=protocol_value,
        model_value=model_value,
        base_url_value=base_url_value,
    )

    try:
        persisted_path = save_external_lm_runtime_settings(
            provider=provider_id,
            protocol=protocol_value,
            model=model_value,
            base_url=base_url_value,
        )
        status_lines.append(f"External LM config persisted at: {persisted_path}")
    except OSError as exc:
        message = f"Failed to persist external LM config: {exc}"
        status_lines.append(message)
        gr.Warning(message)

    lm_model_choice = build_external_model_choice(provider_id, model_value)
    lm_dropdown_choices = build_lm_dropdown_choices(llm_handler, lm_model_choice)
    lm_dropdown_update = gr.update(choices=lm_dropdown_choices, value=lm_model_choice)
    status_lines.append(f"Main LM picker synced to: {lm_model_choice}")
    gr.Info("External LM settings saved.")
    return (
        as_markdown_status("\n".join(status_lines)),
        gr.update(value=""),
        gr.update(value=""),
        lm_dropdown_update,
    )


def fetch_external_lm_models_from_ui(
    provider: str,
    protocol: str,
    base_url: str,
    api_key: str,
    current_model: str,
) -> tuple[dict, str]:
    """Fetch external model IDs and update the model dropdown."""

    profile = get_external_provider_profile(provider)
    protocol_value = (protocol or "").strip() or profile.protocol
    base_url_value = (base_url or "").strip() or profile.default_base_url
    api_key_value = (api_key or "").strip()

    if not api_key_value and profile.api_key_required:
        try:
            api_key_value = resolve_external_api_key_for_runtime(profile.provider_id)
        except SecretStoreError as exc:
            message = (
                f"Model fetch requires {profile.label} credentials: {exc}. "
                "Provide an API key or save credentials first."
            )
            gr.Warning(message)
            return gr.update(), as_markdown_status(message)

    try:
        models = discover_external_models(
            provider=profile.provider_id,
            protocol=protocol_value,
            base_url=base_url_value,
            api_key=api_key_value,
        )
    except ExternalModelDiscoveryError as exc:
        message = f"Model discovery failed: {exc}"
        gr.Warning(message)
        return gr.update(), as_markdown_status(message)

    try:
        save_cached_external_models(
            provider=profile.provider_id,
            protocol=protocol_value,
            base_url=base_url_value,
            models=models,
        )
    except OSError as exc:
        gr.Warning(f"Model cache could not be updated: {exc}")

    selected = (current_model or "").strip()
    if selected not in models:
        selected = models[0]
    status_lines = [
        f"Provider: {profile.label}",
        f"Discovered models: {len(models)}",
        f"Selected: {selected}",
        "Top results: " + ", ".join(models[:10]),
    ]
    gr.Info(f"Fetched {len(models)} models from {profile.label}.")
    return gr.update(choices=models, value=selected), as_markdown_status("\n".join(status_lines))
