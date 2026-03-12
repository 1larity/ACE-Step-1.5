"""LLM client helpers for planning caption/lyrics/metadata text tasks."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request

from loguru import logger

from .external_ai_debug import is_external_ai_debug_enabled, preview_text
from .external_ai_request_helpers import (
    build_http_error_guidance,
    build_planning_messages,
    build_request_for_protocol,
)
from .external_ai_response_parsing import (
    extract_protocol_message_content,
    parse_plan_from_content,
)
from .external_ai_types import ExternalAIClientError, ExternalAIPlan


def _debug_log_external_ai_request(
    *,
    protocol: str,
    model: str,
    base_url: str,
    payload: dict[str, Any],
) -> None:
    """Log sanitized outbound External AI request context when debug mode is enabled."""
    if not is_external_ai_debug_enabled():
        return
    logger.debug(
        "External AI request protocol={} model={} endpoint={} payload={} ",
        protocol,
        model,
        base_url,
        json.dumps(payload, ensure_ascii=False),
    )


def _debug_log_external_ai_response(*, protocol: str, base_url: str, raw_response: str) -> None:
    """Log raw inbound External AI response body when debug mode is enabled."""
    if not is_external_ai_debug_enabled():
        return
    logger.debug(
        "External AI raw response protocol={} endpoint={} body={} ",
        protocol,
        base_url,
        raw_response,
    )


def request_external_ai_plan(
    *,
    api_key: str,
    intent: str,
    model: str,
    base_url: str,
    timeout_sec: int = 120,
    task_focus: str = "all",
    protocol: str = "openai_chat",
) -> ExternalAIPlan:
    """Request structured planning output from external chat-completions endpoints."""
    if not intent.strip():
        raise ExternalAIClientError("Intent cannot be empty.")

    messages = build_planning_messages(intent=intent, task_focus=task_focus)
    payload, headers = build_request_for_protocol(
        protocol=protocol,
        api_key=api_key,
        model=model,
        messages=messages,
        base_url=base_url,
    )
    _debug_log_external_ai_request(
        protocol=protocol,
        model=model,
        base_url=base_url,
        payload=payload,
    )
    req = request.Request(
        url=base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            _debug_log_external_ai_response(protocol=protocol, base_url=base_url, raw_response=raw)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        guidance = build_http_error_guidance(detail=detail, model=model, base_url=base_url)
        request_target = f" | model={model} | endpoint={base_url}"
        raise ExternalAIClientError(
            f"External AI request failed: HTTP {exc.code} {detail[:200]}{request_target}{guidance}"
        ) from exc
    except error.URLError as exc:
        raise ExternalAIClientError(f"External AI request failed: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ExternalAIClientError(f"External AI request timed out after {timeout_sec}s.") from exc

    return parse_external_ai_chat_response(raw, protocol=protocol, task_focus=task_focus)


def parse_external_ai_chat_response(
    raw_response: str,
    protocol: str = "openai_chat",
    task_focus: str = "all",
) -> ExternalAIPlan:
    """Parse external response body into normalized ``ExternalAIPlan``."""
    content = extract_protocol_message_content(raw_response=raw_response, protocol=protocol)
    if is_external_ai_debug_enabled():
        logger.debug(
            "External AI extracted content protocol={} preview={}",
            protocol,
            preview_text(content),
        )
    return parse_plan_from_content(content, task_focus=task_focus)


def build_acestep_generation_payload(plan: ExternalAIPlan) -> dict[str, Any]:
    """Build ACE-Step generation payload that bypasses local LM flows."""
    lyrics = plan.lyrics or ("[Instrumental]" if plan.instrumental else "")
    return {
        "prompt": plan.caption,
        "lyrics": lyrics,
        "bpm": plan.bpm,
        "audio_duration": plan.duration,
        "key_scale": plan.key_scale or None,
        "time_signature": plan.time_signature or None,
        "vocal_language": plan.vocal_language or "unknown",
        "use_format": False,
        "sample_mode": False,
        "use_cot_caption": False,
        "use_cot_language": False,
    }


__all__ = [
    "ExternalAIClientError",
    "ExternalAIPlan",
    "build_acestep_generation_payload",
    "build_planning_messages",
    "parse_external_ai_chat_response",
    "request_external_ai_plan",
]
