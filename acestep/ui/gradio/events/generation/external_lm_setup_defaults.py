"""Defaults and model-discovery helpers for external LM setup actions."""

from __future__ import annotations

from typing import Any, Callable

from acestep.text_tasks.external_lm_model_discovery import ExternalModelDiscoveryError
from acestep.text_tasks.secure_secret_store import SecretStoreError


def load_provider_defaults_data(
    provider: str,
    *,
    get_external_provider_profile: Callable[[str], Any],
    load_external_lm_runtime_settings_for_provider: Callable[[str], dict[str, str] | None],
    as_markdown_status: Callable[[str], str],
) -> tuple[str, list[str], str, str, str]:
    """Return default protocol/model/base-url values plus status text."""
    profile = get_external_provider_profile(provider)
    saved_settings = load_external_lm_runtime_settings_for_provider(profile.provider_id)
    protocol_value = saved_settings.get("protocol", "").strip() if saved_settings else ""
    model_value = saved_settings.get("model", "").strip() if saved_settings else ""
    base_url_value = saved_settings.get("base_url", "").strip() if saved_settings else ""

    selected_protocol = protocol_value or profile.protocol
    selected_model = model_value or profile.default_model
    selected_base_url = base_url_value or profile.default_base_url
    model_choices = list(dict.fromkeys([selected_model, profile.default_model]))

    status_lines = [
        f"Provider: {profile.label}",
        f"Protocol: {selected_protocol}",
        f"Model: {selected_model}",
        f"Base URL: {selected_base_url}",
        f"API key env: {profile.api_key_env}",
        "Loaded saved provider preferences." if saved_settings else "Using built-in provider defaults.",
    ]
    if profile.provider_id == "zai":
        status_lines.append(
            "Coding Plan tip: use https://api.z.ai/api/coding/paas/v4/chat/completions "
            "with a coding-plan-supported model if your quota is on the Coding Plan lane."
        )
    return (
        selected_protocol,
        model_choices,
        selected_model,
        selected_base_url,
        as_markdown_status("\n".join(status_lines)),
    )


def fetch_models_data(
    *,
    provider: str,
    protocol: str,
    base_url: str,
    api_key: str,
    current_model: str,
    get_external_provider_profile: Callable[[str], Any],
    resolve_external_api_key_for_runtime: Callable[[str], str],
    discover_external_models: Callable[..., list[str]],
) -> tuple[list[str], str, str]:
    """Return fetched models, selected model, and status text for the UI."""
    profile = get_external_provider_profile(provider)
    protocol_value = (protocol or "").strip() or profile.protocol
    base_url_value = (base_url or "").strip() or profile.default_base_url
    api_key_value = (api_key or "").strip()

    if not api_key_value and profile.api_key_required:
        try:
            api_key_value = resolve_external_api_key_for_runtime(profile.provider_id)
        except SecretStoreError as exc:
            raise ExternalModelDiscoveryError(
                f"Model fetch requires {profile.label} credentials: {exc}. "
                "Provide API key field or configure runtime credentials first."
            ) from exc

    models = discover_external_models(
        provider=profile.provider_id,
        protocol=protocol_value,
        base_url=base_url_value,
        api_key=api_key_value,
    )
    selected = (current_model or "").strip()
    if selected not in models:
        selected = models[0] if models else ""

    top_results = ", ".join(models[:10]) if models else "(none)"
    status_text = "\n".join(
        [
            f"Provider: {profile.label}",
            f"Discovered models: {len(models)}",
            f"Selected: {selected or '(none)'}",
            "Top results: " + top_results,
        ]
    )
    return models, selected, status_text
