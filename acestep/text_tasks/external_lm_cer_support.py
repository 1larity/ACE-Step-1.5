"""Shared helpers for external LM CER campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .external_lm_model_discovery import ExternalModelDiscoveryError, discover_external_models
from .external_lm_mode import get_active_external_lm_provider, resolve_external_api_key_for_runtime
from .external_lm_ollama_catalog import OllamaModelInfo, list_ollama_models
from .external_lm_providers import get_external_provider_profile

DEFAULT_OLLAMA_MAX_MODEL_GB = 6.0


def is_quota_like_api_error(message: str) -> bool:
    """Return whether an error message looks like quota or balance exhaustion."""

    normalized = (message or "").strip().lower()
    markers = (
        "insufficient_quota",
        "no credits",
        "no credit",
        "no account balance",
        "insufficient balance",
        "billing",
        "quota exceeded",
        "quota is unavailable",
    )
    if any(marker in normalized for marker in markers):
        return True
    return "quota" in normalized and "insufficient" in normalized


def is_model_access_error(message: str) -> bool:
    """Return whether an error means the current model is unavailable to the account."""

    normalized = (message or "").strip().lower()
    markers = (
        "does not yet include access",
        "model not found",
        "access denied",
        "permission denied",
        "not available for your account",
        "you do not have access",
    )
    if any(marker in normalized for marker in markers):
        return True
    return '"code":"1311"' in normalized or "'code': '1311'" in normalized


def resolve_provider_base_url(provider: str) -> str:
    """Resolve the configured provider base URL or the provider default."""

    profile = get_external_provider_profile(provider)
    env_name = {
        "zai": "ACESTEP_GLM_BASE_URL",
        "openai": "ACESTEP_OPENAI_BASE_URL",
        "ollama": "ACESTEP_OLLAMA_BASE_URL",
        "claude": "ACESTEP_ANTHROPIC_BASE_URL",
    }.get(provider)
    generic = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    if generic:
        return generic
    if env_name:
        specific = os.getenv(env_name, "").strip()
        if specific:
            return specific
    return profile.default_base_url


def resolve_coding_base_url(provider: str, base_url: str) -> str | None:
    """Return a coding-endpoint variant for providers that support one."""

    if provider != "zai":
        return None
    normalized = (base_url or "").strip()
    if "/api/coding/paas/" in normalized:
        return normalized
    if "/api/paas/" in normalized:
        return normalized.replace("/api/paas/", "/api/coding/paas/")
    for label, preset in get_external_provider_profile(provider).base_url_presets:
        if "coding" in label.lower():
            return preset
    return None


def configure_external_lm_env(*, provider: str, protocol: str, model: str, base_url: str) -> None:
    """Configure process-local external LM env vars for CER calls."""

    os.environ["ACESTEP_EXTERNAL_LM_ENABLED"] = "true"
    os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = provider
    os.environ["ACESTEP_EXTERNAL_LM_PROTOCOL"] = protocol
    os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = model
    os.environ["ACESTEP_EXTERNAL_BASE_URL"] = base_url
    if provider == "zai":
        os.environ["ACESTEP_GLM_MODEL"] = model
        os.environ["ACESTEP_GLM_BASE_URL"] = base_url


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a single CER result row to a JSONL log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def discover_provider_models(*, provider: str, protocol: str, base_url: str) -> tuple[list[str], str]:
    """Discover provider models, retrying the coding endpoint when quota errors suggest it."""

    api_key = resolve_external_api_key_for_runtime(provider)
    try:
        models = discover_external_models(
            provider=provider,
            protocol=protocol,
            base_url=base_url,
            api_key=api_key,
        )
        return models, base_url
    except ExternalModelDiscoveryError as exc:
        coding_base_url = resolve_coding_base_url(provider, base_url)
        if not coding_base_url or not is_quota_like_api_error(str(exc)):
            raise
        models = discover_external_models(
            provider=provider,
            protocol=protocol,
            base_url=coding_base_url,
            api_key=api_key,
        )
        return models, coding_base_url


def filter_ollama_models_by_size(
    *,
    models: list[str],
    catalog: list[OllamaModelInfo],
    max_model_gb: float,
) -> tuple[list[str], list[tuple[str, int | None]]]:
    """Filter Ollama models to those whose reported size fits the configured limit."""

    if max_model_gb <= 0:
        return list(models), []
    limit_bytes = int(max_model_gb * (1024**3))
    size_by_name = {item.name: item.size_bytes for item in catalog}
    kept: list[str] = []
    skipped: list[tuple[str, int | None]] = []
    for model in models:
        size_bytes = size_by_name.get(model)
        if size_bytes is None:
            skipped.append((model, None))
            continue
        if size_bytes <= limit_bytes:
            kept.append(model)
            continue
        skipped.append((model, size_bytes))
    return kept, skipped


def resolve_default_provider() -> str:
    """Return the current external provider for CLI defaults."""

    return get_active_external_lm_provider()
