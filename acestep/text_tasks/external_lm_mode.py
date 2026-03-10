"""External LM mode helpers for dropdown-driven provider activation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .external_lm_credentials import (
    direct_api_key_env_candidates,
    resolve_secret_store as _resolve_secret_store,
)
from .external_lm_providers import build_external_model_choice, get_external_provider_profile
from .external_lm_runtime_store import (
    hydrate_external_lm_env_from_store,
    load_all_external_lm_runtime_settings,
    load_external_lm_runtime_settings_for_provider,
)
from .passphrase_store import resolve_runtime_passphrase
from .secure_secret_store import SecretStoreError

EXTERNAL_MODEL_PREFIX = "external:"


@dataclass(frozen=True)
class ExternalLmSelection:
    """Parsed external LM selection metadata from dropdown value."""

    provider: str
    model: str


def get_external_lm_choices() -> list[str]:
    """Return dropdown model choices for all configured external LM selections."""
    hydrate_external_lm_env_from_store()
    raw = os.getenv("ACESTEP_EXTERNAL_LM_CHOICES", "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values

    configured_choices = [
        build_external_model_choice(provider_id, settings["model"])
        for provider_id, settings in load_all_external_lm_runtime_settings().items()
        if settings.get("model", "").strip()
    ]
    current_choice = _build_current_external_choice()
    if current_choice:
        configured_choices.append(current_choice)
    return list(dict.fromkeys(configured_choices))


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
    if ":" not in token:
        return ExternalLmSelection(provider="zai", model=token)

    provider_token, model = token.split(":", 1)
    provider = _normalize_provider(provider_token)
    normalized_model = model.strip() or get_external_provider_profile(provider).default_model
    return ExternalLmSelection(provider=provider, model=normalized_model)


def activate_external_lm_mode(model_path: str | None) -> ExternalLmSelection | None:
    """Activate external LM environment flags from dropdown model selection."""
    selection = parse_external_lm_selection(model_path)
    if selection is None:
        deactivate_external_lm_mode()
        return None

    profile = get_external_provider_profile(selection.provider)
    saved_settings = load_external_lm_runtime_settings_for_provider(selection.provider)
    protocol = (saved_settings or {}).get("protocol", "").strip() or profile.protocol
    base_url = (saved_settings or {}).get("base_url", "").strip() or profile.default_base_url

    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = profile.provider_id
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = selection.model
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = protocol
    os.environ["ACESTEP_EXTERNAL_BASE_URL"] = base_url
    os.environ["ACESTEP_TEXT_PROVIDER"] = profile.provider_id
    if profile.provider_id == "zai":
        os.environ["ACESTEP_ZAI_MODEL"] = selection.model
        os.environ["ACESTEP_ZAI_BASE_URL"] = base_url
    return selection


def deactivate_external_lm_mode() -> None:
    """Disable external LM runtime mode while keeping saved external configuration."""
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
        or os.getenv("ACESTEP_ZAI_MODEL", "").strip()
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
    for env_name in direct_api_key_env_candidates(profile.provider_id):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    if not profile.api_key_required:
        return ""

    passphrase = resolve_runtime_passphrase()
    if not passphrase:
        raise SecretStoreError(
            f"Missing {profile.label} credentials. Set {profile.api_key_env} or "
            "configure encrypted-store passphrase via ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE, "
            "ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE_FILE, or system keyring setup."
        )
    return _resolve_secret_store(profile.provider_id).load(passphrase=passphrase).strip()


def resolve_zai_api_key_for_runtime() -> str:
    """Resolve the active Z.ai API key from env or encrypted secret storage."""
    return resolve_external_api_key_for_runtime("zai")


def _build_current_external_choice() -> str | None:
    """Build the current runtime external-model dropdown token, if configured."""
    provider = _normalize_provider(os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip() or "zai")
    model = os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
    if not model and os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").lower() in {"1", "true", "yes"}:
        model = get_external_provider_profile(provider).default_model
    if not model:
        return None
    return build_external_model_choice(provider, model)


def _normalize_provider(provider: str | None) -> str:
    """Normalize provider aliases to stable internal provider identifiers."""
    token = (provider or "").strip().lower()
    if token in {"", "zhipu", "zai"}:
        return "zai"
    if token in {"anthropic", "claud", "claude"}:
        return "claude"
    if token in {"openai", "ollama"}:
        return token
    return "zai"
