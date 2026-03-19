"""Planning request helpers for non-Ollama external LM providers."""

from __future__ import annotations

import json

from .external_ai_request_helpers import (
    build_http_error_guidance,
    build_planning_messages,
    build_request_for_protocol,
    resolve_max_tokens_for_task_focus,
)
from .external_ai_response_parsing import extract_protocol_message_content, parse_plan_from_content
from .external_ai_types import ExternalAIClientError
from .external_lm_http_common import external_base_url, post_json
from .external_lm_ollama_plan_request import request_ollama_plan
from .external_lm_mode import (
    get_active_external_lm_model,
    get_active_external_lm_protocol,
    get_active_external_lm_provider,
    resolve_external_api_key_for_runtime,
)
from .external_lm_providers import get_external_provider_profile
from .secure_secret_store import SecretStoreError


def request_external_plan(
    *,
    intent: str,
    timeout_sec: int,
    task_focus: str,
    debug: bool = False,
):
    """Run an external planning request and parse its structured response."""

    provider = get_active_external_lm_provider()
    model = get_active_external_lm_model()
    protocol = get_active_external_lm_protocol()
    profile = get_external_provider_profile(provider)

    try:
        api_key = resolve_external_api_key_for_runtime(provider)
    except SecretStoreError as exc:
        raise ExternalAIClientError(str(exc)) from exc

    base_url = external_base_url(provider)
    messages = build_planning_messages(intent=intent, task_focus=task_focus)
    if provider == "ollama":
        return (
            request_ollama_plan(
                model=model,
                base_url=base_url,
                messages=messages,
                timeout_sec=timeout_sec,
                task_focus=task_focus,
                debug=debug,
            ),
            profile,
            model,
        )

    payload, headers = build_request_for_protocol(
        protocol=protocol,
        provider=provider,
        api_key=api_key,
        model=model,
        messages=messages,
        base_url=base_url,
        max_tokens=resolve_max_tokens_for_task_focus(task_focus),
        disable_thinking=(task_focus == "format"),
        require_json_output=(task_focus == "format"),
    )
    if debug:
        from acestep.debug_utils import debug_log_for

        debug_log_for(
            "llm",
            "External LM format request:\n"
            f"provider={provider}\n"
            f"protocol={protocol}\n"
            f"model={model}\n"
            f"base_url={base_url}\n"
            f"messages={json.dumps(messages, ensure_ascii=False, indent=2)}\n"
            f"payload={json.dumps(payload, ensure_ascii=False, indent=2)}",
        )

    raw_response = post_json(
        url=base_url,
        payload=payload,
        headers=headers,
        timeout_sec=timeout_sec,
        model=model,
        provider_base_url=base_url,
        build_http_error_guidance_fn=build_http_error_guidance,
    )
    if debug:
        from acestep.debug_utils import debug_log_for

        debug_log_for("llm", f"External LM raw response:\n{raw_response}")
    content = extract_protocol_message_content(raw_response=raw_response, protocol=protocol)
    return parse_plan_from_content(content, task_focus=task_focus), profile, model
