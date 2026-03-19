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
            f"messages={json.dumps(messages, ensure_ascii=False, indent=2)}\n"
            f"payload={json.dumps(payload, ensure_ascii=False, indent=2)}",
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

        debug_log_for("llm", f"External LM raw response:\n{raw_response}")

    outer = json.loads(raw_response)
    message = outer.get("message", {}) if isinstance(outer, dict) else {}
    content = str(message.get("content", "")).strip() if isinstance(message, dict) else ""
    if not content:
        raise ExternalAIClientError(
            "Ollama returned no final content for this formatting request. "
            "Try a non-thinking model or retry with a smaller model."
        )
    return parse_plan_from_content(content, task_focus=task_focus)
