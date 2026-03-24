"""Helpers for external LM dropdown choices and active-mode state."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .external_lm_providers import (
    build_external_model_choice,
    get_external_provider_profile,
)
from .external_lm_runtime_store import hydrate_external_lm_env_from_store
from .external_lm_runtime_access import (
    get_active_external_lm_model as _get_active_external_lm_model,
    get_active_external_lm_protocol as _get_active_external_lm_protocol,
    get_active_external_lm_provider as _get_active_external_lm_provider,
    has_external_lm_runtime_availability as _has_external_lm_runtime_availability,
    has_external_lm_runtime_credentials as _has_external_lm_runtime_credentials,
    normalize_provider as _normalize_provider_impl,
    resolve_external_api_key_for_runtime as _resolve_external_api_key_for_runtime,
)


EXTERNAL_MODEL_PREFIX = "external:"


@dataclass(frozen=True)
class ExternalLmSelection:
    """Parsed external LM selection metadata from the LM dropdown."""

    provider: str
    model: str


def get_external_lm_choices() -> list[str]:
    """Return configured external LM entries for the main LM dropdown."""

    hydrate_external_lm_env_from_store()
    raw = os.getenv("ACESTEP_EXTERNAL_LM_CHOICES", "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values

    provider = _normalize_provider(os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip() or "zai")
    model = os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
    if not model and provider == "zai":
        model = os.getenv("ACESTEP_GLM_MODEL", "").strip()
    if not model and os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").lower() in {"1", "true", "yes"}:
        model = get_external_provider_profile(provider).default_model
    if not model:
        return []
    return [build_external_model_choice(provider, model)]


def is_external_lm_model(model_path: str | None) -> bool:
    """Return whether a dropdown value represents an external provider."""

    return bool(model_path and model_path.startswith(EXTERNAL_MODEL_PREFIX))


def parse_external_lm_selection(model_path: str | None) -> ExternalLmSelection | None:
    """Parse a dropdown value into provider/model metadata."""

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
    """Activate external LM env flags from the LM dropdown selection."""

    selection = parse_external_lm_selection(model_path)
    if selection is None:
        deactivate_external_lm_mode()
        return None

    profile = get_external_provider_profile(selection.provider)
    previous_provider = _normalize_provider(os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip())
    current_base_url = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    current_glm_base_url = os.getenv("ACESTEP_GLM_BASE_URL", "").strip()

    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = profile.provider_id
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = selection.model
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = profile.protocol
    if previous_provider != profile.provider_id or not current_base_url:
        os.environ["ACESTEP_EXTERNAL_BASE_URL"] = profile.default_base_url
    os.environ["ACESTEP_TEXT_PROVIDER"] = profile.provider_id
    if profile.provider_id == "zai":
        os.environ["ACESTEP_GLM_MODEL"] = selection.model
        if previous_provider != profile.provider_id or not current_glm_base_url:
            os.environ["ACESTEP_GLM_BASE_URL"] = profile.default_base_url
    return selection


def deactivate_external_lm_mode() -> None:
    """Disable external LM mode without clearing saved provider preferences."""

    os.environ.pop("ACESTEP_EXTERNAL_LM_ENABLED", None)
    if os.getenv("ACESTEP_TEXT_PROVIDER", "") in {"zai", "glm", "openai", "ollama", "claude"}:
        os.environ.pop("ACESTEP_TEXT_PROVIDER", None)


def is_external_lm_active() -> bool:
    """Return whether external LM mode is active for text actions."""

    return os.getenv("ACESTEP_EXTERNAL_LM_ENABLED", "").lower() in {"1", "true", "yes"}


def is_lm_ready(
    llm_handler: object | None = None,
    lm_model_path: str | None = None,
) -> bool:
    """Return whether either a local LM or the active external LM is available."""

    if bool(getattr(llm_handler, "llm_initialized", False)):
        return True
    if is_external_lm_active():
        return has_external_lm_runtime_availability()
    if is_external_lm_model(lm_model_path):
        selection = parse_external_lm_selection(lm_model_path)
        return has_external_lm_runtime_availability(selection.provider if selection else None)
    return False


def has_external_lm_runtime_credentials(provider: str | None = None) -> bool:
    """Return whether runtime credentials exist for the selected external provider."""

    return _has_external_lm_runtime_credentials(provider, _normalize_provider)


def has_external_lm_runtime_availability(provider: str | None = None) -> bool:
    """Return whether the selected external provider appears usable right now."""

    return _has_external_lm_runtime_availability(provider, _normalize_provider)


def get_active_external_lm_provider(default: str = "zai") -> str:
    """Return the configured external provider identifier."""

    return _normalize_provider(_get_active_external_lm_provider(default))


def get_active_external_lm_model(default: str = "glm-4.5-flash") -> str:
    """Return the configured external model identifier."""

    return _get_active_external_lm_model(default)


def get_active_external_lm_protocol() -> str:
    """Return the configured external protocol or provider default."""

    return _get_active_external_lm_protocol()


def resolve_external_api_key_for_runtime(provider: str | None = None) -> str:
    """Resolve the active provider API key from env or encrypted storage."""

    return _resolve_external_api_key_for_runtime(provider, _normalize_provider)


def _normalize_provider(provider: str | None) -> str:
    """Normalize provider aliases to stable internal identifiers."""

    return _normalize_provider_impl(provider)
