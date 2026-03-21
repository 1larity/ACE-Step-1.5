"""Ollama-native planning request helpers for external LM flows."""

from __future__ import annotations

import json

from .external_ai_request_helpers import (
    build_http_error_guidance,
    resolve_max_tokens_for_task_focus,
)
from .external_ai_response_parsing import parse_plan_from_content
from .external_ai_types import ExternalAIClientError
from .external_lm_http_common import ollama_native_api_url, post_json


def _build_safe_debug_payload_view(payload: dict[str, object]) -> dict[str, object]:
    """Return a non-sensitive subset of the Ollama request payload for debug logs."""

    safe_payload = {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "think": payload.get("think"),
    }
    if "format" in payload:
        format_value = payload["format"]
        if isinstance(format_value, dict):
            safe_payload["format"] = {"type": format_value.get("type", "object")}
    if "options" in payload:
        options = payload.get("options") or {}
        if isinstance(options, dict):
            safe_payload["options"] = {
                "temperature": options.get("temperature"),
                "num_predict": options.get("num_predict"),
            }
    return safe_payload


def _build_safe_debug_messages_view(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    """Return a non-sensitive summary of Ollama request messages for debug logs."""

    return [
        {
            "role": str(message.get("role", "")).strip(),
            "content_length": len(str(message.get("content", ""))),
        }
        for message in messages
        if isinstance(message, dict)
    ]


def _build_safe_debug_response_view(raw_response: str) -> dict[str, int]:
    """Return a minimal response summary that avoids logging response content."""

    return {"response_length": len(raw_response or "")}


def request_ollama_plan(
    *,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    task_focus: str,
    debug: bool = False,
):
    """Run an Ollama-native planning request with thinking explicitly disabled."""

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "lyrics": {"type": "string"},
                "bpm": {"type": ["integer", "string", "null"]},
                "duration": {"type": ["integer", "number", "string", "null"]},
                "key_scale": {"type": "string"},
                "time_signature": {"type": "string"},
                "vocal_language": {"type": "string"},
                "instrumental": {"type": "boolean"},
            },
            "required": [
                "caption",
                "lyrics",
                "bpm",
                "duration",
                "key_scale",
                "time_signature",
                "vocal_language",
                "instrumental",
            ],
        },
        "options": {
            "temperature": 0.4,
            "num_predict": resolve_max_tokens_for_task_focus(task_focus),
        },
    }
    if debug:
        from acestep.debug_utils import debug_log_for

        debug_log_for(
            "llm",
            "External LM format request:\n"
            "provider=ollama\n"
            "protocol=ollama_chat\n"
            f"model={model}\n"
            f"base_url={base_url}\n"
            f"messages={json.dumps(_build_safe_debug_messages_view(messages), ensure_ascii=False, indent=2)}\n"
            f"payload={json.dumps(_build_safe_debug_payload_view(payload), ensure_ascii=False, indent=2)}",
        )

    raw_response = post_json(
        url=ollama_native_api_url(base_url=base_url, path="/api/chat"),
        payload=payload,
        headers={"Content-Type": "application/json"},
        timeout_sec=timeout_sec,
        model=model,
        provider_base_url=base_url,
        build_http_error_guidance_fn=build_http_error_guidance,
    )
    if debug:
        from acestep.debug_utils import debug_log_for

        debug_log_for(
            "llm",
            "External LM response summary:\n"
            f"{json.dumps(_build_safe_debug_response_view(raw_response), ensure_ascii=False, indent=2)}",
        )

    outer = json.loads(raw_response)
    message = outer.get("message", {}) if isinstance(outer, dict) else {}
    content = str(message.get("content", "")).strip() if isinstance(message, dict) else ""
    if not content:
        raise ExternalAIClientError(
            "Ollama returned no final content for this formatting request. "
            "Try a non-thinking model or retry with a smaller model."
        )
    return parse_plan_from_content(content, task_focus=task_focus)
