"""Runtime access helpers for external LM configuration and credentials."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib import parse

from .external_lm_providers import get_external_provider_profile
from .external_lm_runtime_store import hydrate_external_lm_env_from_store
from .passphrase_store import resolve_runtime_passphrase
from .secure_secret_store import EncryptedSecretStore, SecretStoreError


def has_external_lm_runtime_credentials(provider: str | None, normalize_provider) -> bool:
    """Return whether runtime credentials exist for the selected provider."""

    profile = get_external_provider_profile(normalize_provider(provider or get_active_external_lm_provider()))
    if not profile.api_key_required:
        return True
    try:
        return bool(resolve_external_api_key_for_runtime(profile.provider_id, normalize_provider))
    except SecretStoreError:
        return False


def has_external_lm_runtime_availability(provider: str | None, normalize_provider) -> bool:
    """Return whether the selected external provider appears usable right now."""

    normalized_provider = normalize_provider(provider or get_active_external_lm_provider())
    if not has_external_lm_runtime_credentials(normalized_provider, normalize_provider):
        return False
    if normalized_provider == "ollama":
        return _is_ollama_endpoint_reachable()
    return True


def get_active_external_lm_provider(default: str = "zai") -> str:
    """Return the configured external provider identifier."""

    hydrate_external_lm_env_from_store()
    return os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip() or default


def get_active_external_lm_model(default: str = "glm-4.5-flash") -> str:
    """Return the configured external model identifier."""

    hydrate_external_lm_env_from_store()
    provider = normalize_provider(get_active_external_lm_provider())
    provider_default = get_external_provider_profile(provider).default_model
    return (
        os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
        or os.getenv("ACESTEP_GLM_MODEL", "").strip()
        or provider_default
        or default
    )


def get_active_external_lm_protocol() -> str:
    """Return the configured external protocol or provider default."""

    hydrate_external_lm_env_from_store()
    protocol = os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL", "").strip()
    if protocol:
        return protocol
    return get_external_provider_profile(normalize_provider(get_active_external_lm_provider())).protocol


def resolve_external_api_key_for_runtime(provider: str | None, normalize_provider) -> str:
    """Resolve the active provider API key from env or stored credentials."""

    profile = get_external_provider_profile(normalize_provider(provider or get_active_external_lm_provider()))
    for env_name in _direct_api_key_env_candidates(profile.provider_id):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    if not profile.api_key_required:
        return ""

    store = _resolve_secret_store(profile.provider_id)
    if store.uses_native_keyring():
        return store.load(passphrase="").strip()

    passphrase = resolve_runtime_passphrase()
    if not passphrase:
        raise SecretStoreError(
            f"Missing {profile.label} credentials. Set {profile.api_key_env} or configure "
            "the encrypted-store passphrase via ACESTEP_GLM_STORE_PASSPHRASE, "
            "ACESTEP_GLM_STORE_PASSPHRASE_FILE, or the system keyring."
        )
    return store.load(passphrase=passphrase).strip()


def normalize_provider(provider: str | None) -> str:
    """Normalize provider aliases to stable internal identifiers."""

    token = (provider or "").strip().lower()
    if token in {"", "glm", "zhipu", "zai"}:
        return "zai"
    if token in {"anthropic", "claud", "claude"}:
        return "claude"
    if token in {"openai", "ollama"}:
        return token
    return "zai"


def _is_ollama_endpoint_reachable() -> bool:
    """Return whether the configured Ollama host and port accept TCP connections."""

    base_url = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip() or os.getenv(
        "ACESTEP_OLLAMA_BASE_URL",
        "",
    ).strip()
    if not base_url:
        base_url = get_external_provider_profile("ollama").default_base_url
    parsed_url = parse.urlparse(base_url)
    hostname = (parsed_url.hostname or "").strip()
    port = parsed_url.port or 11434
    if not hostname:
        return False
    try:
        with socket.create_connection((hostname, port), timeout=0.2):
            return True
    except OSError:
        return False


def _direct_api_key_env_candidates(provider: str) -> list[str]:
    """Return API-key env vars to probe for a provider."""

    profile = get_external_provider_profile(provider)
    candidates = ["ACESTEP_EXTERNAL_API_KEY", profile.api_key_env]
    if provider == "zai":
        candidates.insert(1, "ACESTEP_GLM_API_KEY")
    if provider == "claude":
        candidates.append("ACESTEP_CLAUDE_API_KEY")
    return candidates


def _resolve_secret_store(provider: str) -> EncryptedSecretStore:
    """Resolve the encrypted credential file path for a provider."""

    profile = get_external_provider_profile(provider)
    secret_path_raw = os.getenv(profile.secret_path_env, "").strip()
    if secret_path_raw:
        return EncryptedSecretStore(secret_path=Path(secret_path_raw).expanduser())
    return EncryptedSecretStore(
        secret_path=EncryptedSecretStore.resolve_existing_default_path(
            filename=profile.secret_file_name
        )
    )
