"""Warm-up helpers for external LM providers."""

from __future__ import annotations

import json
import os
import socket
from urllib import error, request

from .external_ai_request_helpers import build_http_error_guidance
from .external_ai_types import ExternalAIClientError
from .external_lm_http_common import (
    coerce_keep_alive_value,
    external_base_url,
    ollama_model_is_loaded,
    ollama_native_api_url,
)
from .external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_provider,
    resolve_external_api_key_for_runtime,
)
from .secure_secret_store import SecretStoreError


def warm_up_external_provider(timeout_sec: int | None = None) -> str | None:
    """Best-effort warm-up for the active external provider during service init."""

    provider = get_active_external_lm_provider()
    if provider != "ollama":
        return None

    model = get_active_external_lm_model()
    try:
        api_key = resolve_external_api_key_for_runtime(provider)
    except SecretStoreError as exc:
        raise ExternalAIClientError(str(exc)) from exc

    base_url = external_base_url(provider)
    if ollama_model_is_loaded(model=model, base_url=base_url):
        return f"Ollama model already loaded ({model})"

    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": coerce_keep_alive_value(
            os.getenv("ACESTEP_EXTERNAL_LM_WARMUP_KEEP_ALIVE", "-1")
        ),
        "options": {
            "num_predict": int(os.getenv("ACESTEP_EXTERNAL_LM_WARMUP_MAX_TOKENS", "8")),
            "temperature": 0.0,
        },
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        url=ollama_native_api_url(base_url=base_url, path="/api/generate"),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    effective_timeout = timeout_sec or int(os.getenv("ACESTEP_EXTERNAL_LM_WARMUP_TIMEOUT", "120"))
    try:
        with request.urlopen(req, timeout=effective_timeout) as response:
            response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        guidance = build_http_error_guidance(detail=detail, model=model, base_url=base_url)
        raise ExternalAIClientError(f"HTTP {exc.code}: {detail[:240]}{guidance}") from exc
    except (TimeoutError, socket.timeout) as exc:
        if ollama_model_is_loaded(model=model, base_url=base_url):
            return f"Ollama model finished loading after timeout window ({model})"
        raise ExternalAIClientError(
            "Timed out while warming the Ollama model. The model may still load on first use."
        ) from exc
    except error.URLError as exc:
        raise ExternalAIClientError(f"Network error contacting external provider: {exc}") from exc
    return f"Ollama warm-up complete ({model})"
