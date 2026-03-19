"""Default and status helpers for the external LM setup panel."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from acestep.text_tasks.external_lm_model_cache import load_cached_external_models
from acestep.text_tasks.external_lm_mode import resolve_external_api_key_for_runtime
from acestep.text_tasks.external_lm_providers import (
    CUSTOM_BASE_URL_PRESET,
    get_external_base_url_preset_choices,
    get_external_base_url_preset_value,
    get_external_provider_profile,
)
from acestep.text_tasks.external_lm_runtime_store import load_external_lm_runtime_settings
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


def load_external_lm_provider_defaults(provider: str) -> tuple[dict, dict, dict, dict, str]:
    """Return protocol, model, and base-URL defaults for a provider selection."""

    profile = get_external_provider_profile(provider)
    saved_settings = load_external_lm_runtime_settings(profile.provider_id) or {}
    protocol_value = saved_settings.get("protocol", "") or profile.protocol
    base_url_value = saved_settings.get("base_url", "") or profile.default_base_url
    selected_model = saved_settings.get("model", "") or profile.default_model
    cached_models = load_cached_external_models(
        provider=profile.provider_id,
        protocol=protocol_value,
        base_url=base_url_value,
    )
    model_choices = cached_models or [selected_model]
    if selected_model not in model_choices:
        model_choices = [selected_model, *model_choices]
    status_lines = [
        f"Provider: {profile.label}",
        f"Protocol: {protocol_value}",
        f"Default model: {selected_model}",
        f"Default base URL: {base_url_value}",
        f"API key env: {profile.api_key_env}",
        "Tip: try a coding model and coding endpoint if your subscription includes coding access.",
    ]
    if cached_models:
        status_lines.append(f"Cached models available: {len(cached_models)}")
    return (
        gr.update(value=protocol_value),
        gr.update(choices=model_choices, value=selected_model),
        gr.update(
            choices=get_external_base_url_preset_choices(profile.provider_id),
            value=get_external_base_url_preset_value(
                profile.provider_id, base_url_value
            ),
        ),
        gr.update(value=base_url_value),
        as_markdown_status("\n".join(status_lines)),
    )


def apply_external_lm_base_url_preset(
    provider: str,
    preset_value: str,
    current_base_url: str,
) -> dict:
    """Resolve a preset selection into the editable base-URL textbox value."""

    profile = get_external_provider_profile(provider)
    normalized_preset = (preset_value or "").strip()
    if not normalized_preset or normalized_preset == CUSTOM_BASE_URL_PRESET:
        return gr.update(value=(current_base_url or "").strip() or profile.default_base_url)
    return gr.update(value=normalized_preset)


def check_external_lm_runtime_from_ui(
    provider: str,
    protocol: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Report whether the selected external provider configuration is usable."""

    profile = get_external_provider_profile(provider)
    provider_id = profile.provider_id
    configured_model = (model or "").strip() or profile.default_model
    configured_protocol = (protocol or "").strip() or profile.protocol
    configured_base_url = (base_url or "").strip() or profile.default_base_url

    status_lines = [
        f"Provider: {profile.label}",
        f"Protocol: {configured_protocol}",
        f"Configured model: {configured_model}",
        f"Configured base URL: {configured_base_url}",
        f"External LM mode enabled: {'yes' if os.getenv('ACESTEP_EXTERNAL_LM_ENABLED', '').strip().lower() in {'1', 'true', 'yes'} else 'no'}",
    ]
    if profile.api_key_required:
        try:
            resolve_external_api_key_for_runtime(provider_id)
            status_lines.append("Credentials: available")
        except SecretStoreError as exc:
            status_lines.append(f"Credentials: missing ({exc})")
    else:
        status_lines.append("Credentials: not required by default")

    store = resolve_secret_store_for_provider(provider_id)
    status_lines.append(
        f"Encrypted key path: {store.secret_path} ({'present' if Path(store.secret_path).exists() else 'missing'})"
    )
    return as_markdown_status("\n".join(status_lines))


def resolve_secret_store_for_provider(provider: str) -> EncryptedSecretStore:
    """Resolve the provider-specific encrypted secret store path."""

    profile = get_external_provider_profile(provider)
    secret_path_raw = os.getenv(profile.secret_path_env, "").strip()
    if secret_path_raw:
        return EncryptedSecretStore(secret_path=Path(secret_path_raw).expanduser())
    return EncryptedSecretStore(
        secret_path=EncryptedSecretStore.resolve_existing_default_path(
            filename=profile.secret_file_name
        )
    )


def hydrate_external_env(
    *,
    provider_id: str,
    protocol_value: str,
    model_value: str,
    base_url_value: str,
) -> None:
    """Apply external LM settings to the process environment."""

    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = provider_id
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = protocol_value
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = model_value
    os.environ["ACESTEP_EXTERNAL_BASE_URL"] = base_url_value
    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_TEXT_PROVIDER"] = provider_id
    if provider_id == "zai":
        os.environ["ACESTEP_GLM_MODEL"] = model_value
        os.environ["ACESTEP_GLM_BASE_URL"] = base_url_value


def as_markdown_status(text: str) -> str:
    """Render multi-line plain text as a Markdown bullet block."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)
