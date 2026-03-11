"""Request-building helpers for external AI text-task integrations."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import parse

from .external_ai_types import ExternalAIClientError


def build_task_focus_guidance(*, task_focus: str) -> str:
    """Return task-focus specific generation guidance for planning prompts."""
    normalized_focus = (task_focus or "all").strip().lower()
    if normalized_focus == "format":
        return (
            "For format focus: preserve user intent, then improve clarity and musical specificity. "
            "If the caption is sparse, fragmentary, or just a keyword list, expand it into a full "
            "standard ACE-Step caption with global traits first and the linear song narrative after that. "
            "Do not change the core genre/mood unless required for coherence."
        )
    if normalized_focus == "lyrics":
        return (
            "For lyrics focus: keep the caption concept aligned, but write finished singable lyrics. "
            "Do not return placeholder scaffolds, control instructions, or tag-only output unless the user explicitly requests instrumental output."
        )
    return (
        "For all-task focus: produce complete caption/lyrics/metadata that can directly drive "
        "music generation."
    )


def build_planning_messages(intent: str, task_focus: str = "all") -> list[dict[str, str]]:
    """Build chat messages for structured planning output."""
    system = (
        "You generate structured music planning JSON for ACE-Step. "
        "Return only valid JSON with keys: caption, lyrics, bpm, duration, "
        "key_scale, time_signature, vocal_language, instrumental. "
        "Caption must be a narrative production brief (1-2 sentences) that includes: "
        "genre/mood, linear arrangement arc (e.g. intro -> build -> chorus/drop -> outro), "
        "core instrumentation, singer gender and delivery mood/timbre when vocals are present, and "
        "mix/energy trajectory. Keep it concise but specific. "
        "If instrumental=true, set lyrics to [Instrumental]."
    )
    user = (
        f"Task focus: {task_focus}\n"
        f"User intent:\n{intent}\n\n"
        f"{build_task_focus_guidance(task_focus=task_focus)}\n"
        "Output JSON only. No markdown, no commentary."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def resolve_openai_thinking_payload() -> dict[str, str]:
    """Return provider-specific thinking controls for supported OpenAI-like endpoints."""
    configured = os.getenv("ACESTEP_EXTERNAL_AI_THINKING", "disabled").strip().lower()
    thinking_type = "enabled" if configured in {"1", "true", "yes", "on", "enabled"} else "disabled"
    return {"type": thinking_type}


def supports_provider_thinking(*, protocol: str, base_url: str) -> bool:
    """Return whether the configured endpoint supports provider-specific thinking controls."""
    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol != "openai_chat":
        return False

    parsed = parse.urlparse(base_url or "")
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return "z.ai" in host or "/api/paas/" in path or "/api/coding/paas/" in path


def build_request_for_protocol(
    *,
    protocol: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    base_url: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build protocol-specific request payload and HTTP headers."""
    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol == "anthropic_messages":
        if not api_key:
            raise ExternalAIClientError("Missing API key for anthropic_messages protocol.")
        payload = {
            "model": model,
            "max_tokens": int(os.getenv("ACESTEP_ANTHROPIC_MAX_TOKENS", "1024")),
            "temperature": 0.4,
            "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": os.getenv("ACESTEP_ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        }
        return payload, headers

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(os.getenv("ACESTEP_OPENAI_MAX_TOKENS", "3072")),
        "temperature": 0.4,
    }
    if supports_provider_thinking(protocol=normalized_protocol, base_url=base_url):
        payload["thinking"] = resolve_openai_thinking_payload()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return payload, headers


def build_http_error_guidance(*, detail: str, model: str, base_url: str) -> str:
    """Return targeted guidance for common API error payloads."""
    if not detail:
        return ""
    try:
        payload = json.loads(detail)
        err_payload = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(err_payload, dict):
            err = err_payload
        elif err_payload in (None, ""):
            err = {}
        else:
            err = {"message": str(err_payload)}
        code = str(err.get("code", "")).strip()
        error_type = str(err.get("type", "")).strip().lower()
        message = str(err.get("message", "")).strip().lower()
    except json.JSONDecodeError:
        code = ""
        error_type = ""
        message = detail.strip().lower()

    parsed_base_url = parse.urlparse(base_url or "")
    normalized_host = (parsed_base_url.hostname or "").strip().lower()
    normalized_path = (parsed_base_url.path or "").strip().lower()
    is_openai_endpoint = normalized_host == "api.openai.com"
    quota_like_error = (
        code == "1113"
        or code == "insufficient_quota"
        or error_type == "insufficient_quota"
        or "insufficient_quota" in message
        or ("quota" in message and "insufficient" in message)
    )

    if code == "1211":
        return " | Model not found. Try a valid provider model and verify your account has access."
    if code == "1113":
        is_coding_endpoint = normalized_host == "api.z.ai" and normalized_path.startswith(
            "/api/coding/paas/v4/"
        )
        endpoint_hint = ""
        if not is_coding_endpoint:
            endpoint_hint = (
                " Current endpoint looks like general API; coding-plan quota requires "
                "https://api.z.ai/api/coding/paas/v4/chat/completions."
            )
        return (
            " | 1113 means billing quota is unavailable for this request."
            " If you rely on Z.ai coding-plan quota, use the dedicated coding endpoint and a "
            "supported coding-plan model."
            f"{endpoint_hint} Otherwise top up API balance/resource package."
        )
    if is_openai_endpoint and quota_like_error:
        return (
            " | OpenAI API quota is unavailable for this request. ChatGPT/Codex subscription "
            "usage is separate from API billing on platform.openai.com, so ChatGPT or Codex "
            "plan access does not automatically fund API-key calls from this app. This "
            "integration currently uses the OpenAI chat-completions style API; some Codex "
            "models prefer the Responses API, which ACE-Step does not currently target. "
            "Verify the API project's billing/quota, or use a model/endpoint supported by "
            "your funded API project."
        )
    return ""
