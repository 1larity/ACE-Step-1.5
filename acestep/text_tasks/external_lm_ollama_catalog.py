"""Ollama model catalog helpers for CER and model selection flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(frozen=True)
class OllamaModelInfo:
    """Normalized Ollama model metadata from the ``/api/tags`` endpoint."""

    name: str
    size_bytes: int | None


def build_ollama_tags_url(base_url: str) -> str:
    """Build the native Ollama ``/api/tags`` URL from a configured base URL."""

    parsed = parse.urlparse((base_url or "").strip())
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    if not netloc:
        raise RuntimeError("Invalid Ollama base URL.")
    return parse.urlunparse((scheme, netloc, "/api/tags", "", "", ""))


def parse_ollama_tags_payload(payload: Any) -> list[OllamaModelInfo]:
    """Extract Ollama model names and sizes from a tags payload."""

    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    results: list[OllamaModelInfo] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip() or str(item.get("model", "")).strip()
        if not name:
            continue
        raw_size = item.get("size")
        size_bytes: int | None = None
        if isinstance(raw_size, int):
            size_bytes = raw_size
        elif isinstance(raw_size, str) and raw_size.strip().isdigit():
            size_bytes = int(raw_size.strip())
        results.append(OllamaModelInfo(name=name, size_bytes=size_bytes))
    return results


def list_ollama_models(base_url: str, timeout_sec: int = 20) -> list[OllamaModelInfo]:
    """Query Ollama for model metadata, including byte sizes."""

    req = request.Request(url=build_ollama_tags_url(base_url), method="GET")
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"HTTP {exc.code}: {detail[:200]}") from exc
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to query Ollama model catalog: {exc}") from exc
    return parse_ollama_tags_payload(payload)
