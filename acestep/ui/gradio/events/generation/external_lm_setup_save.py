"""Save-flow helpers for external LM setup actions."""

from __future__ import annotations

import os
from typing import Any, Callable

from acestep.text_tasks.secure_secret_store import SecretStoreError


class ExternalLmSetupSaveError(RuntimeError):
    """Raised when saving external LM settings fails validation."""


def save_external_lm_settings(
    *,
    provider: str,
    protocol: str,
    model: str,
    base_url: str,
    api_key: str,
    store_passphrase: str,
    save_passphrase_to_keyring: bool,
    get_external_provider_profile: Callable[[str], Any],
    resolve_secret_store_for_provider: Callable[[str], Any],
    store_runtime_passphrase: Callable[[str], tuple[bool, str]],
    resolve_external_api_key_for_runtime: Callable[[str], str],
    save_external_lm_runtime_settings: Callable[..., Any],
    build_runtime_summary_line: Callable[[str], str],
    warning_fn: Callable[[str], None],
    as_markdown_status: Callable[[str], str],
) -> tuple[str, str]:
    """Persist external settings and return status text plus external model token."""
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
    if provider_id == "zai" and "api/coding/paas/v4" not in base_url_value.lower():
        status_lines.append(
            "Coding Plan tip: if your quota is on Z.ai Coding Plan, switch Base URL to "
            "https://api.z.ai/api/coding/paas/v4/chat/completions."
        )

    _update_session_api_key_env(profile.api_key_env, api_key_value, status_lines)
    _update_passphrase_env(passphrase_value, status_lines)

    if api_key_value:
        _handle_optional_encrypted_storage(
            profile=profile,
            provider_id=provider_id,
            api_key_value=api_key_value,
            passphrase_value=passphrase_value,
            save_passphrase_to_keyring=save_passphrase_to_keyring,
            resolve_secret_store_for_provider=resolve_secret_store_for_provider,
            store_runtime_passphrase=store_runtime_passphrase,
            status_lines=status_lines,
            warning_fn=warning_fn,
        )
    else:
        _validate_existing_credentials(
            profile=profile,
            provider_id=provider_id,
            resolve_external_api_key_for_runtime=resolve_external_api_key_for_runtime,
            status_lines=status_lines,
        )

    _apply_runtime_env(provider_id, protocol_value, model_value, base_url_value, status_lines)
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
        warning_fn(message)

    status_lines.append(build_runtime_summary_line(provider_id))
    return as_markdown_status("\n".join(status_lines)), model_value


def _handle_optional_encrypted_storage(
    *,
    profile: Any,
    provider_id: str,
    api_key_value: str,
    passphrase_value: str,
    save_passphrase_to_keyring: bool,
    resolve_secret_store_for_provider: Callable[[str], Any],
    store_runtime_passphrase: Callable[[str], tuple[bool, str]],
    status_lines: list[str],
    warning_fn: Callable[[str], None],
) -> None:
    """Persist encrypted credentials when the user provided a passphrase."""
    if passphrase_value:
        try:
            store = resolve_secret_store_for_provider(provider_id)
            store.save(secret=api_key_value, passphrase=passphrase_value)
            status_lines.append(f"Encrypted API key stored at: {store.secret_path}")
            os.environ["ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE"] = passphrase_value
        except SecretStoreError as exc:
            message = f"Failed to store encrypted API key: {exc}"
            warning_fn(message)
            raise ExternalLmSetupSaveError(message) from exc

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


def _validate_existing_credentials(
    *,
    profile: Any,
    provider_id: str,
    resolve_external_api_key_for_runtime: Callable[[str], str],
    status_lines: list[str],
) -> None:
    """Validate that existing credentials are available when no new key was provided."""
    if not profile.api_key_required:
        status_lines.append("Provider does not require API key by default.")
        return
    try:
        resolve_external_api_key_for_runtime(provider_id)
        status_lines.append("Using existing provider credentials from env/secret store.")
    except SecretStoreError as exc:
        raise ExternalLmSetupSaveError(f"{profile.label} API key is required: {exc}") from exc


def _update_session_api_key_env(
    api_key_env: str,
    api_key_value: str,
    status_lines: list[str],
) -> None:
    """Update the current-session API key environment variable."""
    if api_key_value:
        os.environ[api_key_env] = api_key_value
        status_lines.append(f"Session API key set via env: {api_key_env}")
        return
    if os.environ.pop(api_key_env, None) is not None:
        status_lines.append(f"Cleared session API key env: {api_key_env}")


def _update_passphrase_env(passphrase_value: str, status_lines: list[str]) -> None:
    """Update or clear the current-session encrypted-store passphrase."""
    if passphrase_value:
        os.environ["ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE"] = passphrase_value
        return
    if os.environ.pop("ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE", None) is not None:
        status_lines.append("Cleared session encrypted-store passphrase.")


def _apply_runtime_env(
    provider_id: str,
    protocol_value: str,
    model_value: str,
    base_url_value: str,
    status_lines: list[str],
) -> None:
    """Apply active external LM runtime environment variables."""
    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = provider_id
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = protocol_value
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = model_value
    os.environ["ACESTEP_EXTERNAL_BASE_URL"] = base_url_value
    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_TEXT_PROVIDER"] = provider_id
    if provider_id == "zai":
        os.environ["ACESTEP_ZAI_MODEL"] = model_value
        os.environ["ACESTEP_ZAI_BASE_URL"] = base_url_value
    else:
        cleared = []
        if os.environ.pop("ACESTEP_ZAI_MODEL", None) is not None:
            cleared.append("ACESTEP_ZAI_MODEL")
        if os.environ.pop("ACESTEP_ZAI_BASE_URL", None) is not None:
            cleared.append("ACESTEP_ZAI_BASE_URL")
        if cleared:
            status_lines.append("Cleared stale Z.ai runtime env: " + ", ".join(cleared))
