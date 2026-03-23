"""Gradio handlers for external LLM setup tab actions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_model_discovery import ExternalModelDiscoveryError, discover_external_models
from acestep.text_tasks.external_lm_mode import get_external_lm_choices, resolve_external_api_key_for_runtime
from acestep.text_tasks.external_lm_providers import (
    build_external_model_choice,
    get_external_provider_profile,
)
from acestep.text_tasks.external_lm_runtime_store import load_external_lm_runtime_settings_for_provider, save_external_lm_runtime_settings
from acestep.text_tasks.passphrase_store import EXTERNAL_AI_SECRET_SERVICE, EXTERNAL_AI_SECRET_USERNAME, resolve_runtime_passphrase, store_runtime_passphrase
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError

from .external_lm_setup_defaults import fetch_models_data, load_provider_defaults_data
from .external_lm_setup_runtime import build_runtime_status
from .external_lm_setup_save import ExternalLmSetupSaveError, save_external_lm_settings
from .external_lm_setup_support import as_markdown_status as _as_markdown_status
from .external_lm_setup_support import build_lm_dropdown_choices as _support_lm_dropdown_choices
from .external_lm_setup_support import build_runtime_summary_line as _support_runtime_summary_line
from .external_lm_setup_support import python_keyring_available as _python_keyring_available
from .external_lm_setup_support import secret_tool_available as _secret_tool_available


def load_external_lm_provider_defaults(provider: str) -> tuple[dict, dict, dict, str]:
    """Return provider settings from saved prefs when available, else provider defaults."""
    protocol, choices, model, base_url, status = load_provider_defaults_data(
        provider,
        get_external_provider_profile=get_external_provider_profile,
        load_external_lm_runtime_settings_for_provider=load_external_lm_runtime_settings_for_provider,
        as_markdown_status=_as_markdown_status,
    )
    return (
        gr.update(value=protocol),
        gr.update(choices=choices, value=model),
        gr.update(
            choices=list(get_external_provider_profile(provider).base_url_presets),
            value=base_url,
        ),
        status,
    )


def load_external_lm_provider_defaults_with_lm_dropdown(provider: str, llm_handler: Any | None = None) -> tuple[dict, dict, dict, str, dict]:
    """Load provider defaults and refresh the service LM dropdown choices."""
    protocol_update, model_update, base_url_update, status = load_external_lm_provider_defaults(provider)
    return protocol_update, model_update, base_url_update, status, _build_lm_dropdown_preview_update(provider, model_update.get("value"), llm_handler)


def fetch_external_lm_models_from_ui(provider: str, protocol: str, base_url: str, api_key: str, current_model: str) -> tuple[dict, str]:
    """Fetch model IDs from selected external endpoint and update model dropdown."""
    try:
        models, selected, status_text = fetch_models_data(
            provider=provider,
            protocol=protocol,
            base_url=base_url,
            api_key=api_key,
            current_model=current_model,
            get_external_provider_profile=get_external_provider_profile,
            resolve_external_api_key_for_runtime=resolve_external_api_key_for_runtime,
            discover_external_models=discover_external_models,
        )
    except ExternalModelDiscoveryError as exc:
        message = f"Model discovery failed: {exc}"
        gr.Warning(message)
        return gr.update(), _as_markdown_status(message)
    gr.Info(f"Fetched {len(models)} models from {get_external_provider_profile(provider).label}.")
    return gr.update(choices=models, value=selected), _as_markdown_status(status_text)


def fetch_external_lm_models_from_ui_with_lm_dropdown(provider: str, protocol: str, base_url: str, api_key: str, current_model: str, llm_handler: Any | None = None) -> tuple[dict, str, dict]:
    """Fetch provider models and refresh the service LM dropdown choices."""
    model_update, status = fetch_external_lm_models_from_ui(provider, protocol, base_url, api_key, current_model)
    return model_update, status, _build_lm_dropdown_preview_update(provider, model_update.get("value") or current_model, llm_handler)


def save_external_lm_settings_from_ui(provider: str, protocol: str, model: str, base_url: str, api_key: str, store_passphrase: str, save_passphrase_to_keyring: bool, llm_handler: Any | None = None) -> tuple[str, dict, dict, dict]:
    """Persist external provider settings and optional encrypted credentials."""
    try:
        status_text, model_value = save_external_lm_settings(
            provider=provider,
            protocol=protocol,
            model=model,
            base_url=base_url,
            api_key=api_key,
            store_passphrase=store_passphrase,
            save_passphrase_to_keyring=save_passphrase_to_keyring,
            get_external_provider_profile=get_external_provider_profile,
            resolve_secret_store_for_provider=_resolve_secret_store_for_provider,
            store_runtime_passphrase=store_runtime_passphrase,
            resolve_external_api_key_for_runtime=resolve_external_api_key_for_runtime,
            save_external_lm_runtime_settings=save_external_lm_runtime_settings,
            build_runtime_summary_line=_build_runtime_summary_line,
            warning_fn=gr.Warning,
            as_markdown_status=_as_markdown_status,
        )
    except ExternalLmSetupSaveError as exc:
        gr.Warning(str(exc))
        return _as_markdown_status(str(exc)), gr.update(), gr.update(), gr.update()
    provider_id = get_external_provider_profile(provider).provider_id
    choice = build_external_model_choice(provider_id, model_value)
    choices = _support_lm_dropdown_choices(llm_handler, get_external_lm_choices)
    update = gr.update(choices=choices, value=choice) if choices is not None else gr.update(value=choice)
    gr.Info("External LLM settings saved.")
    return status_text, gr.update(value=""), gr.update(value=""), update


def check_external_lm_runtime_from_ui(provider: str, protocol: str | None = None, model: str | None = None, base_url: str | None = None) -> str:
    """Report external runtime readiness for selected provider."""
    try:
        status, ready = build_runtime_status(
            provider=provider,
            protocol=protocol,
            model=model,
            base_url=base_url,
            get_external_provider_profile=get_external_provider_profile,
            resolve_secret_store_for_provider=_resolve_secret_store_for_provider,
            resolve_runtime_passphrase=resolve_runtime_passphrase,
            secret_tool_available=_secret_tool_available,
            python_keyring_available=_python_keyring_available,
            secret_service=os.getenv("ACESTEP_EXTERNAL_AI_SECRET_SERVICE", EXTERNAL_AI_SECRET_SERVICE),
            secret_username=os.getenv("ACESTEP_EXTERNAL_AI_SECRET_USERNAME", EXTERNAL_AI_SECRET_USERNAME),
            as_markdown_status=_as_markdown_status,
        )
    except SecretStoreError as exc:
        message = f"Secret store unavailable: {exc}"
        gr.Warning(message)
        return _as_markdown_status(message)
    (gr.Info if ready else gr.Warning)("External runtime is ready." if ready else "External runtime is not ready.")
    return status


def _build_lm_dropdown_preview_update(provider: str, model: str | None, llm_handler: Any | None) -> dict:
    """Build LM dropdown update that includes a staged external provider/model choice."""
    choices = _support_lm_dropdown_choices(llm_handler, get_external_lm_choices)
    if choices is None:
        return gr.update()
    preview_choice = build_external_model_choice(provider, (model or "").strip() or get_external_provider_profile(provider).default_model)
    return gr.update(choices=list(dict.fromkeys(choices + [preview_choice])))


def _resolve_secret_store_for_provider(provider: str) -> EncryptedSecretStore:
    """Resolve provider-specific encrypted secret store path."""
    profile = get_external_provider_profile(provider)
    configured = os.getenv(profile.secret_path_env, "").strip()
    if configured:
        return EncryptedSecretStore(secret_path=Path(configured).expanduser())
    return EncryptedSecretStore(secret_path=EncryptedSecretStore.resolve_existing_default_path(filename=profile.secret_file_name))


def _build_runtime_summary_line(provider: str) -> str:
    """Return concise runtime readiness summary line after save."""
    return _support_runtime_summary_line(provider, resolve_external_api_key_for_runtime=resolve_external_api_key_for_runtime, secret_error_cls=SecretStoreError)
