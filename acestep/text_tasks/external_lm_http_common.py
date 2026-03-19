"""Common HTTP helpers for external LM provider requests."""

from __future__ import annotations

import json
import os
import socket
from urllib import error, parse, request

from .external_ai_types import ExternalAIClientError
from .external_lm_providers import get_external_provider_profile


def coerce_keep_alive_value(raw_value: str) -> int | str:
    """Coerce keep-alive env values into Ollama-friendly JSON types."""

    normalized = (raw_value or "").strip()
    if not normalized:
        return -1
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    return normalized


def external_base_url(provider: str) -> str:
    """Return the configured provider base URL with provider-specific fallbacks."""

    profile = get_external_provider_profile(provider)
    generic = os.getenv("ACESTEP_EXTERNAL_BASE_URL", "").strip()
    if generic:
        return generic
    if provider == "zai":
        zai_url = os.getenv("ACESTEP_GLM_BASE_URL", "").strip()
        if zai_url:
            return zai_url
    provider_specific_env = {
        "openai": "ACESTEP_OPENAI_BASE_URL",
        "ollama": "ACESTEP_OLLAMA_BASE_URL",
        "claude": "ACESTEP_ANTHROPIC_BASE_URL",
    }.get(provider)
    if provider_specific_env:
        configured = os.getenv(provider_specific_env, "").strip()
        if configured:
            return configured
    return profile.default_base_url


def ollama_native_api_url(*, base_url: str, path: str) -> str:
    """Build a native Ollama API URL from the configured base URL."""

    parsed = parse.urlparse(base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    if not netloc:
        raise ExternalAIClientError("Invalid Ollama base URL.")
    return parse.urlunparse((scheme, netloc, path, "", "", ""))


def ollama_model_is_loaded(*, model: str, base_url: str, timeout_sec: int = 2) -> bool:
    """Return whether the requested Ollama model is currently loaded."""

    ps_url = ollama_native_api_url(base_url=base_url, path="/api/ps")
    req = request.Request(url=ps_url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError, TimeoutError, socket.timeout):
        return False

    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return False
    for item in models:
        if not isinstance(item, dict):
            continue
        active_name = str(item.get("name", "")).strip()
        active_model = str(item.get("model", "")).strip()
        if model in {active_name, active_model}:
            return True
    return False


def post_json(
    *,
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_sec: int,
    model: str,
    provider_base_url: str,
    build_http_error_guidance_fn,
) -> str:
    """POST a JSON payload and return the decoded response body."""

    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            return response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        guidance = build_http_error_guidance_fn(
            detail=detail,
            model=model,
            base_url=provider_base_url,
        )
        raise ExternalAIClientError(f"HTTP {exc.code}: {detail[:240]}{guidance}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ExternalAIClientError(
            "Timed out waiting for the external provider response. "
            "If you are using a local Ollama CPU model, try a smaller model or retry."
        ) from exc
    except error.URLError as exc:
        raise ExternalAIClientError(f"Network error contacting external provider: {exc}") from exc
