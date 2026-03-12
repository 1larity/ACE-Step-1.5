"""Credential resolution helpers for external LM runtime activation."""

from __future__ import annotations

import os
from pathlib import Path

from .external_lm_providers import get_external_provider_profile
from .secure_secret_store import EncryptedSecretStore


def direct_api_key_env_candidates(provider: str) -> list[str]:
    """Return candidate API-key env vars in lookup priority order."""
    profile = get_external_provider_profile(provider)
    candidates = [profile.api_key_env]
    if provider == "claude" and "ACESTEP_CLAUDE_API_KEY" != profile.api_key_env:
        candidates.append("ACESTEP_CLAUDE_API_KEY")
    candidates.append("ACESTEP_EXTERNAL_API_KEY")
    return candidates


def resolve_secret_store(provider: str) -> EncryptedSecretStore:
    """Resolve encrypted secret store path for provider credentials."""
    profile = get_external_provider_profile(provider)
    secret_path_raw = os.getenv(profile.secret_path_env, "").strip()
    if secret_path_raw:
        return EncryptedSecretStore(secret_path=Path(secret_path_raw).expanduser())
    return EncryptedSecretStore(
        secret_path=EncryptedSecretStore.resolve_existing_default_path(
            filename=profile.secret_file_name,
        )
    )
