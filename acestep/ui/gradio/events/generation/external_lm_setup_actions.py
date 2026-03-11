"""Gradio handlers for external LLM setup tab actions."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_model_discovery import (
    ExternalModelDiscoveryError,
    discover_external_models,
)
from acestep.text_tasks.external_lm_mode import (
    get_external_lm_choices,
    resolve_external_api_key_for_runtime,
)
from acestep.text_tasks.external_lm_runtime_store import (
    load_external_lm_runtime_settings_for_provider,
    save_external_lm_runtime_settings,
)
from acestep.text_tasks.external_lm_providers import (
    build_external_model_choice,
    get_external_provider_profile,
)
from acestep.text_tasks.passphrase_store import (
    EXTERNAL_AI_SECRET_SERVICE,
    EXTERNAL_AI_SECRET_USERNAME,
    resolve_runtime_passphrase,
    store_runtime_passphrase,
)
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


def load_external_lm_provider_defaults(provider: str) -> tuple[dict, dict, dict, str]:
    """Return provider settings from saved prefs when available, else provider defaults."""
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
        (
            "Loaded saved provider preferences."
            if saved_settings
            else "Using built-in provider defaults."
        ),
    ]
    if profile.provider_id == "zai":
        status_lines.append(
            "Coding Plan tip: use https://api.z.ai/api/coding/paas/v4/chat/completions "
            "with a coding-plan-supported model if your quota is on the Coding Plan lane."
        )
    status = _as_markdown_status("\n".join(status_lines))
    return (
        gr.update(value=selected_protocol),
        gr.update(choices=model_choices, value=selected_model),
        gr.update(value=selected_base_url),
        status,
    )


def load_external_lm_provider_defaults_with_lm_dropdown(
    provider: str,
    llm_handler: Any | None = None,
) -> tuple[dict, dict, dict, str, dict]:
    """Load provider defaults and refresh the service LM dropdown choices."""
    protocol_update, model_update, base_url_update, status = load_external_lm_provider_defaults(provider)
    lm_dropdown_update = _build_lm_dropdown_preview_update(
        provider=provider,
        model=model_update.get("value"),
        llm_handler=llm_handler,
    )
    return protocol_update, model_update, base_url_update, status, lm_dropdown_update


def fetch_external_lm_models_from_ui(
    provider: str,
    protocol: str,
    base_url: str,
    api_key: str,
    current_model: str,
) -> tuple[dict, str]:
    """Fetch model IDs from selected external endpoint and update model dropdown."""
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
                "Provide API key field or configure runtime credentials first."
            )
            gr.Warning(message)
            return gr.update(), _as_markdown_status(message)

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
        return gr.update(), _as_markdown_status(message)

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
    return (
        gr.update(choices=models, value=selected),
        _as_markdown_status("\n".join(status_lines)),
    )


def fetch_external_lm_models_from_ui_with_lm_dropdown(
    provider: str,
    protocol: str,
    base_url: str,
    api_key: str,
    current_model: str,
    llm_handler: Any | None = None,
) -> tuple[dict, str, dict]:
    """Fetch provider models and refresh the service LM dropdown choices."""
    model_update, status = fetch_external_lm_models_from_ui(
        provider=provider,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
        current_model=current_model,
    )
    lm_dropdown_update = _build_lm_dropdown_preview_update(
        provider=provider,
        model=model_update.get("value") or current_model,
        llm_handler=llm_handler,
    )
    return model_update, status, lm_dropdown_update


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
    """Persist external provider settings and optional encrypted credentials."""
    profile = get_external_provider_profile(provider)
    provider_id = profile.provider_id
    model_value = (model or "").strip() or profile.default_model
    protocol_value = (protocol or "").strip() or profile.protocol
    base_url_value = (base_url or "").strip() or profile.default_base_url
    api_key_value = (api_key or "").strip()
    passphrase_value = (store_passphrase or "").strip()

    status_lines = [
        f"Provider set to: {profile.label}",
        "External LM mode: enabled",
        f"Protocol: {protocol_value}",
        f"Model: {model_value}",
        f"Base URL: {base_url_value}",
    ]
    if profile.provider_id == "zai" and "api/coding/paas/v4" not in base_url_value.lower():
        status_lines.append(
            "Coding Plan tip: if your quota is on Z.ai Coding Plan, switch Base URL to "
            "https://api.z.ai/api/coding/paas/v4/chat/completions."
        )

    if api_key_value:
        os.environ[profile.api_key_env] = api_key_value
        status_lines.append(f"Session API key set via env: {profile.api_key_env}")

        if passphrase_value:
            try:
                store = _resolve_secret_store_for_provider(provider_id)
                store.save(secret=api_key_value, passphrase=passphrase_value)
                status_lines.append(f"Encrypted API key stored at: {store.secret_path}")
                os.environ["ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE"] = passphrase_value
            except SecretStoreError as exc:
                message = f"Failed to store encrypted API key: {exc}"
                gr.Warning(message)
                return _as_markdown_status(message), gr.update(), gr.update(), gr.update()

            if save_passphrase_to_keyring:
                ok, keyring_message = store_runtime_passphrase(passphrase_value)
                status_lines.append(keyring_message)
                if not ok:
                    status_lines.append(
                        "Passphrase keyring persistence failed; runtime still works for this session."
                    )
            else:
                status_lines.append("Passphrase persistence disabled; available only for current session.")
        elif profile.api_key_required:
            status_lines.append(
                "API key not encrypted: set Store Passphrase to persist encrypted key across restarts."
            )
    else:
        if profile.api_key_required:
            try:
                _ = resolve_external_api_key_for_runtime(provider_id)
                status_lines.append("Using existing provider credentials from env/secret store.")
            except SecretStoreError as exc:
                message = f"{profile.label} API key is required: {exc}"
                gr.Warning(message)
                return _as_markdown_status(message), gr.update(), gr.update(), gr.update()
        else:
            status_lines.append("Provider does not require API key by default.")

    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = provider_id
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = protocol_value
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = model_value
    os.environ["ACESTEP_EXTERNAL_BASE_URL"] = base_url_value
    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_TEXT_PROVIDER"] = provider_id
    if provider_id == "zai":
        os.environ["ACESTEP_ZAI_MODEL"] = model_value
        os.environ["ACESTEP_ZAI_BASE_URL"] = base_url_value
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

    ready_message = _build_runtime_summary_line(provider_id)
    status_lines.append(ready_message)
    status_text = _as_markdown_status("\n".join(status_lines))
    lm_model_choice = build_external_model_choice(provider_id, model_value)
    lm_dropdown_choices = _build_lm_dropdown_choices(llm_handler)
    lm_dropdown_update = (
        gr.update(choices=lm_dropdown_choices, value=lm_model_choice)
        if lm_dropdown_choices is not None
        else gr.update(value=lm_model_choice)
    )
    gr.Info("External LLM settings saved.")
    return (
        status_text,
        gr.update(value=""),
        gr.update(value=""),
        lm_dropdown_update,
    )


def check_external_lm_runtime_from_ui(
    provider: str,
    protocol: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Report external runtime readiness for selected provider."""
    profile = get_external_provider_profile(provider)
    provider_id = profile.provider_id
    try:
        store = _resolve_secret_store_for_provider(provider_id)
    except SecretStoreError as exc:
        message = f"Secret store unavailable: {exc}"
        gr.Warning(message)
        return _as_markdown_status(message)
    has_direct_key = bool(os.getenv(profile.api_key_env, "").strip())
    runtime_passphrase = resolve_runtime_passphrase()
    saved_model = os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
    saved_protocol = os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL", "").strip()
    saved_base_url = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    configured_model = (model or "").strip() or saved_model or profile.default_model
    configured_protocol = (protocol or "").strip() or saved_protocol or profile.protocol
    configured_base_url = (base_url or "").strip() or saved_base_url or profile.default_base_url
    external_enabled = os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").strip().lower() in {"1", "true", "yes"}

    status_lines = [
        f"Provider: {profile.label}",
        f"Protocol: {configured_protocol}",
        f"Configured model: {configured_model}",
        f"Configured base URL: {configured_base_url}",
        f"External LM mode enabled: {'yes' if external_enabled else 'no'}",
        f"Encrypted key file: {store.secret_path}",
        f"Encrypted key exists: {'yes' if store.exists() else 'no'}",
        f"Direct API key env set ({profile.api_key_env}): {'yes' if has_direct_key else 'no'}",
        f"Runtime passphrase source found: {'yes' if bool(runtime_passphrase) else 'no'}",
        f"secret-tool available: {'yes' if _secret_tool_available() else 'no'}",
        f"python keyring available: {'yes' if _python_keyring_available() else 'no'}",
        "Secret lookup identity: "
        f"service={os.getenv('ACESTEP_EXTERNAL_AI_SECRET_SERVICE', EXTERNAL_AI_SECRET_SERVICE)} "
        f"username={os.getenv('ACESTEP_EXTERNAL_AI_SECRET_USERNAME', EXTERNAL_AI_SECRET_USERNAME)}",
    ]
    if saved_model and configured_model != saved_model:
        status_lines.append(
            "UI model differs from saved runtime model. Click 'Save External LLM Settings' to apply."
        )

    ready = not profile.api_key_required
    if profile.api_key_required:
        ready = has_direct_key
        if not ready and runtime_passphrase and store.exists():
            try:
                ready = bool(store.load(passphrase=runtime_passphrase).strip())
            except SecretStoreError as exc:
                status_lines.append(f"Decrypt check failed: {exc}")

    status_lines.append(f"External runtime status: {'ready' if ready else 'not ready'}")
    status = _as_markdown_status("\n".join(status_lines))
    if ready:
        gr.Info("External runtime is ready.")
    else:
        gr.Warning("External runtime is not ready.")
    return status




def _build_runtime_summary_line(provider: str) -> str:
    """Return concise runtime readiness summary line after save."""
    profile = get_external_provider_profile(provider)
    if not profile.api_key_required:
        return "External runtime status: ready"

    try:
        key = resolve_external_api_key_for_runtime(provider)
    except SecretStoreError:
        return "External runtime status: passphrase/API key not yet available for non-interactive runtime"
    if key:
        return "External runtime status: ready"
    return "External runtime status: passphrase/API key not yet available for non-interactive runtime"


def _build_lm_dropdown_choices(llm_handler: Any | None) -> list[str] | None:
    """Build LM dropdown choices using local 5Hz models plus configured external model."""
    if llm_handler is None:
        return None

    local_models = llm_handler.get_available_5hz_lm_models() or []
    return list(dict.fromkeys(local_models + get_external_lm_choices()))


def _build_lm_dropdown_preview_update(
    provider: str,
    model: str | None,
    llm_handler: Any | None,
) -> dict:
    """Build LM dropdown update that includes a staged external provider/model choice."""
    choices = _build_lm_dropdown_choices(llm_handler)
    if choices is None:
        return gr.update()

    model_value = (model or "").strip() or get_external_provider_profile(provider).default_model
    preview_choice = build_external_model_choice(provider, model_value)
    return gr.update(choices=list(dict.fromkeys(choices + [preview_choice])))


def _resolve_secret_store_for_provider(provider: str) -> EncryptedSecretStore:
    """Resolve provider-specific encrypted secret store path."""
    profile = get_external_provider_profile(provider)
    configured = os.getenv(profile.secret_path_env, "").strip()
    if configured:
        return EncryptedSecretStore(secret_path=Path(configured).expanduser())
    return EncryptedSecretStore(
        secret_path=EncryptedSecretStore.resolve_existing_default_path(
            filename=profile.secret_file_name,
        )
    )


def _as_markdown_status(text: str) -> str:
    """Render multiline status text in a readable monospaced block."""
    safe_text = (text or "").strip() or "No status."
    return f"```text\n{safe_text}\n```"


def _secret_tool_available() -> bool:
    """Return whether `secret-tool` is available on PATH."""
    return bool(shutil.which("secret-tool"))


def _python_keyring_available() -> bool:
    """Return whether python keyring module is installed."""
    return bool(importlib.util.find_spec("keyring"))
