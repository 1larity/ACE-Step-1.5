"""Persistent non-secret runtime settings for external LM provider selection."""

from __future__ import annotations

import json
import os
from pathlib import Path


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
    """Persist non-secret external LM runtime settings to user-local JSON."""
    payload = {
        "provider": (provider or "").strip(),
        "protocol": (protocol or "").strip(),
        "model": (model or "").strip(),
        "base_url": (base_url or "").strip(),
    }
    path = external_lm_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    path.chmod(0o600)
    return path


def load_external_lm_runtime_settings() -> dict[str, str] | None:
    """Load persisted non-secret external LM runtime settings, if available."""
    path = external_lm_settings_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        "provider": str(parsed.get("provider", "")).strip(),
        "protocol": str(parsed.get("protocol", "")).strip(),
        "model": str(parsed.get("model", "")).strip(),
        "base_url": str(parsed.get("base_url", "")).strip(),
    }


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
        if model and not os.getenv("ACESTEP_GLM_MODEL", "").strip():
            os.environ["ACESTEP_GLM_MODEL"] = model
            changed = True
        if base_url and not os.getenv("ACESTEP_GLM_BASE_URL", "").strip():
            os.environ["ACESTEP_GLM_BASE_URL"] = base_url
            changed = True

    return changed
