"""External LM mode helpers for dropdown-driven provider activation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .external_lm_providers import (
    build_external_model_choice,
    get_external_provider_profile,
)
from .external_lm_runtime_store import hydrate_external_lm_env_from_store
from .passphrase_store import resolve_runtime_passphrase
from .secure_secret_store import EncryptedSecretStore, SecretStoreError


EXTERNAL_MODEL_PREFIX = "external:"


@dataclass(frozen=True)
class ExternalLmSelection:
    """Parsed external LM selection metadata from dropdown value."""

    provider: str
    model: str


def get_external_lm_choices() -> list[str]:
    """Return dropdown model choices for configured external LM selections.

    By default this returns at most one external entry, reflecting current
    runtime configuration (provider + model). This keeps the service dropdown
    focused on built-in 5Hz models plus the actively configured external model.
    """
    hydrate_external_lm_env_from_store()
    raw = os.getenv("ACESTEP_EXTERNAL_LM_CHOICES", "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values

    provider = _normalize_provider(os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip() or "zai")
    model = os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()

    # Legacy compatibility for older GLM-only configuration.
    if not model and provider == "zai":
        model = os.getenv("ACESTEP_GLM_MODEL", "").strip()

    if not model and os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").lower() in {"1", "true", "yes"}:
        model = get_external_provider_profile(provider).default_model

    if not model:
        return []

    return [build_external_model_choice(provider, model)]


def is_external_lm_model(model_path: str | None) -> bool:
    """Return whether ``model_path`` represents an external provider selection."""
    return bool(model_path and model_path.startswith(EXTERNAL_MODEL_PREFIX))


def parse_external_lm_selection(model_path: str | None) -> ExternalLmSelection | None:
    """Parse dropdown value into provider/model selection."""
    if not is_external_lm_model(model_path):
        return None

    token = model_path[len(EXTERNAL_MODEL_PREFIX) :].strip()
    if not token:
        return None

    # Legacy format: external:glm-4.5-flash
    if ":" not in token:
        return ExternalLmSelection(provider="zai", model=token)

    provider_token, model = token.split(":", 1)
    provider = _normalize_provider(provider_token)
    normalized_model = model.strip()
    if not normalized_model:
        normalized_model = get_external_provider_profile(provider).default_model
    return ExternalLmSelection(provider=provider, model=normalized_model)


def activate_external_lm_mode(model_path: str | None) -> ExternalLmSelection | None:
    """Activate external LM environment flags from dropdown model selection."""
    selection = parse_external_lm_selection(model_path)
    if selection is None:
        deactivate_external_lm_mode()
        return None

    profile = get_external_provider_profile(selection.provider)
    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = profile.provider_id
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = selection.model
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = profile.protocol
    os.environ.setdefault("ACESTEP_EXTERNAL_BASE_URL", profile.default_base_url)
    os.environ["ACESTEP_TEXT_PROVIDER"] = profile.provider_id

    if profile.provider_id == "zai":
        os.environ["ACESTEP_GLM_MODEL"] = selection.model
        os.environ.setdefault("ACESTEP_GLM_BASE_URL", profile.default_base_url)

    return selection


def deactivate_external_lm_mode() -> None:
    """Disable external LM runtime mode while keeping saved external configuration.

    This keeps provider/model/protocol env values intact so the configured
    external LM dropdown entry remains available after users temporarily switch
    to a local 5Hz model.
    """
    os.environ.pop("ACESTEP_EXTERNAL_LM_ENABLED", None)
    if os.getenv("ACESTEP_TEXT_PROVIDER", "") in {"zai", "glm", "openai", "ollama", "claude"}:
        os.environ.pop("ACESTEP_TEXT_PROVIDER", None)


def is_external_lm_active() -> bool:
    """Return whether external LM mode is currently active."""
    return os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").lower() in {"1", "true", "yes"}


def get_active_external_lm_provider(default: str = "zai") -> str:
    """Return active external provider identifier from env or fallback default."""
    hydrate_external_lm_env_from_store()
    return _normalize_provider(os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip() or default)


def get_active_external_lm_model(default: str = "glm-4.5-flash") -> str:
    """Return active external model from env or fallback default."""
    hydrate_external_lm_env_from_store()
    provider = get_active_external_lm_provider()
    provider_default = get_external_provider_profile(provider).default_model
    return (
        os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
        or os.getenv("ACESTEP_GLM_MODEL", "").strip()
        or provider_default
        or default
    )


def get_active_external_lm_protocol() -> str:
    """Return active external protocol from env or provider default."""
    hydrate_external_lm_env_from_store()
    protocol = os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL", "").strip()
    if protocol:
        return protocol
    return get_external_provider_profile(get_active_external_lm_provider()).protocol


def resolve_external_api_key_for_runtime(provider: str | None = None) -> str:
    """Resolve external API key from env or encrypted local secret storage."""
    profile = get_external_provider_profile(_normalize_provider(provider or get_active_external_lm_provider()))

    for env_name in _direct_api_key_env_candidates(profile.provider_id):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    if not profile.api_key_required:
        return ""

    passphrase = resolve_runtime_passphrase()
    if not passphrase:
        raise SecretStoreError(
            f"Missing {profile.label} credentials. Set {profile.api_key_env} or "
            "configure encrypted-store passphrase via ACESTEP_GLM_STORE_PASSPHRASE, "
            "ACESTEP_GLM_STORE_PASSPHRASE_FILE, or system keyring setup."
        )

    store = _resolve_secret_store(profile.provider_id)
    return store.load(passphrase=passphrase).strip()


def resolve_glm_api_key_for_runtime() -> str:
    """Resolve Z.ai/GLM API key from env or encrypted user-local secret store."""
    return resolve_external_api_key_for_runtime("zai")


def _normalize_provider(provider: str | None) -> str:
    """Normalize provider aliases to stable internal provider identifiers."""
    token = (provider or "").strip().lower()
    if token in {"", "glm", "zhipu", "zai"}:
        return "zai"
    if token in {"anthropic", "claud", "claude"}:
        return "claude"
    if token in {"openai", "ollama"}:
        return token
    return "zai"


def _direct_api_key_env_candidates(provider: str) -> list[str]:
    """Return candidate API-key env vars in lookup priority order."""
    profile = get_external_provider_profile(provider)
    candidates = ["ACESTEP_EXTERNAL_API_KEY", profile.api_key_env]
    if provider == "zai":
        candidates.insert(1, "ACESTEP_GLM_API_KEY")
    if provider == "claude":
        candidates.append("ACESTEP_CLAUDE_API_KEY")
    return candidates


def _resolve_secret_store(provider: str) -> EncryptedSecretStore:
    """Resolve encrypted secret store path for provider credentials."""
    profile = get_external_provider_profile(provider)
    secret_path_raw = os.getenv(profile.secret_path_env, "").strip()
    if secret_path_raw:
        return EncryptedSecretStore(secret_path=Path(secret_path_raw).expanduser())

    if provider == "zai":
        return EncryptedSecretStore(
            secret_path=EncryptedSecretStore.resolve_existing_default_path(filename=profile.secret_file_name)
        )

    return EncryptedSecretStore(
        secret_path=EncryptedSecretStore.resolve_existing_default_path(filename=profile.secret_file_name)
    )
