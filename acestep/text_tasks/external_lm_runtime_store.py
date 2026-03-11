"""Persistent non-secret runtime settings for external LM provider selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_RUNTIME_KEYS = ("provider", "protocol", "model", "base_url")


def external_lm_settings_path() -> Path:
    """Return user-local persistent path for external LM runtime settings JSON."""
    xdg_data_home = os.getenv("XDG_DATA_HOME", "").strip()
    base = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    return base / "acestep" / "config" / "external_lm_runtime.json"


def save_external_lm_runtime_settings(
    *,
    provider: str,
    protocol: str,
    model: str,
    base_url: str,
) -> Path:
    """Persist active and provider-scoped external LM runtime settings to user-local JSON."""
    current_settings = _normalize_runtime_settings(
        {
            "provider": provider,
            "protocol": protocol,
            "model": model,
            "base_url": base_url,
        }
    )
    existing_payload = _load_external_lm_runtime_payload() or {}
    provider_settings = _normalize_provider_settings(existing_payload.get("providers"))
    existing_active = _normalize_runtime_settings(existing_payload)
    if existing_active:
        existing_provider_id = existing_active.get("provider", "").strip().lower()
        if existing_provider_id and existing_provider_id not in provider_settings:
            provider_settings[existing_provider_id] = existing_active

    provider_id = current_settings["provider"]
    if provider_id:
        provider_settings[provider_id] = current_settings

    payload: dict[str, Any] = dict(current_settings)
    payload["providers"] = provider_settings

    path = external_lm_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    path.chmod(0o600)
    return path


def load_external_lm_runtime_settings() -> dict[str, str] | None:
    """Load persisted active external LM runtime settings, if available."""
    payload = _load_external_lm_runtime_payload()
    if payload is None:
        return None
    return _normalize_runtime_settings(payload)


def load_external_lm_runtime_settings_for_provider(provider: str) -> dict[str, str] | None:
    """Load persisted runtime settings for a specific provider, if available."""
    provider_id = (provider or "").strip().lower()
    if not provider_id:
        return None

    all_settings = load_all_external_lm_runtime_settings()
    return all_settings.get(provider_id)


def load_all_external_lm_runtime_settings() -> dict[str, dict[str, str]]:
    """Load all persisted provider-scoped runtime settings keyed by provider id."""
    payload = _load_external_lm_runtime_payload()
    if payload is None:
        return {}

    provider_settings = _normalize_provider_settings(payload.get("providers"))
    active = _normalize_runtime_settings(payload)
    if active:
        provider_id = active.get("provider", "").strip().lower()
        if provider_id and provider_id not in provider_settings:
            provider_settings[provider_id] = active
    return provider_settings


def hydrate_external_lm_env_from_store() -> bool:
    """Fill missing external LM env vars from persisted settings file.

    Returns:
        True when persisted settings were loaded and at least one env var was populated.
    """
    settings = load_external_lm_runtime_settings()
    if not settings:
        return False

    changed = False
    mappings = {
        "provider": "ACESTEP_EXTERNAL_LM_PROVIDER",
        "protocol": "ACESTEP_EXTERNAL_LM_PROTOCOL",
        "model": "ACESTEP_EXTERNAL_LM_MODEL",
        "base_url": "ACESTEP_EXTERNAL_BASE_URL",
    }
    for key, env_name in mappings.items():
        if os.getenv(env_name, "").strip():
            continue
        value = settings.get(key, "")
        if value:
            os.environ[env_name] = value
            changed = True

    provider = os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER", "").strip().lower()
    if provider == "zai":
        model = os.getenv("ACESTEP_EXTERNAL_LM_MODEL", "").strip()
        base_url = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
        if model and not os.getenv("ACESTEP_ZAI_MODEL", "").strip():
            os.environ["ACESTEP_ZAI_MODEL"] = model
            changed = True
        if base_url and not os.getenv("ACESTEP_ZAI_BASE_URL", "").strip():
            os.environ["ACESTEP_ZAI_BASE_URL"] = base_url
            changed = True

    return changed


def _load_external_lm_runtime_payload() -> dict[str, Any] | None:
    """Load the persisted runtime settings JSON payload, if it is a mapping."""
    path = external_lm_settings_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_runtime_settings(raw_settings: Any) -> dict[str, str] | None:
    """Return normalized runtime-setting fields from an arbitrary mapping."""
    if not isinstance(raw_settings, dict):
        return None
    normalized = {
        key: str(raw_settings.get(key, "")).strip()
        for key in _RUNTIME_KEYS
    }
    if not any(normalized.values()):
        return None
    return normalized


def _normalize_provider_settings(raw_provider_settings: Any) -> dict[str, dict[str, str]]:
    """Return normalized provider-scoped settings keyed by provider id."""
    if not isinstance(raw_provider_settings, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for provider_id, provider_settings in raw_provider_settings.items():
        key = str(provider_id or "").strip().lower()
        settings = _normalize_runtime_settings(provider_settings)
        if key and settings:
            normalized[key] = settings
    return normalized
