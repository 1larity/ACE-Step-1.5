"""Path helpers for encrypted external AI secret storage."""

from __future__ import annotations

import os
from pathlib import Path


def default_secret_path(filename: str = "external_ai_api_key.enc") -> Path:
    """Return default encrypted secret path in persistent user data storage."""
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
    return base / "acestep" / "secrets" / filename


def legacy_secret_path(filename: str = "external_ai_api_key.enc") -> Path:
    """Return historical encrypted secret path under ``~/.local/share``."""
    return Path.home() / ".local" / "share" / "acestep" / "secrets" / filename


def resolve_existing_secret_path(filename: str = "external_ai_api_key.enc") -> Path:
    """Return existing default path with legacy fallback when available."""
    primary = default_secret_path(filename=filename)
    if primary.exists():
        return primary
    legacy = legacy_secret_path(filename=filename)
    if legacy.exists():
        return legacy
    return primary
