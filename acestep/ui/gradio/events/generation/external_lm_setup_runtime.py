"""Runtime diagnostics helpers for external LM setup actions."""

from __future__ import annotations

import os
from typing import Any, Callable

from acestep.text_tasks.secure_secret_store import SecretStoreError


def build_runtime_status(
    *,
    provider: str,
    protocol: str | None,
    model: str | None,
    base_url: str | None,
    get_external_provider_profile: Callable[[str], Any],
    resolve_secret_store_for_provider: Callable[[str], Any],
    resolve_runtime_passphrase: Callable[[], str | None],
    secret_tool_available: Callable[[], bool],
    python_keyring_available: Callable[[], bool],
    secret_service: str,
    secret_username: str,
    as_markdown_status: Callable[[str], str],
) -> tuple[str, bool]:
    """Build runtime doctor status text and readiness flag for a provider."""
    profile = get_external_provider_profile(provider)
    provider_id = profile.provider_id
    store = resolve_secret_store_for_provider(provider_id)
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
        f"secret-tool available: {'yes' if secret_tool_available() else 'no'}",
        f"python keyring available: {'yes' if python_keyring_available() else 'no'}",
        f"Secret lookup identity: service={secret_service} username={secret_username}",
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
    return as_markdown_status("\n".join(status_lines)), ready
