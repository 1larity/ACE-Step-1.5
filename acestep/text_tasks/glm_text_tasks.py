"""LLM client helpers for planning caption/lyrics/metadata text tasks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error, request


class GlmClientError(RuntimeError):
    """Raised when external API calls or response parsing fail."""


@dataclass
class GlmPlan:
    """Structured external text-task result for ACE-Step generation inputs."""

    caption: str
    lyrics: str
    bpm: int | None
    duration: float | None
    key_scale: str
    time_signature: str
    vocal_language: str
    instrumental: bool

    def to_dict(self) -> dict[str, Any]:
        """Return plan as a serializable dictionary."""
        return asdict(self)


def build_planning_messages(intent: str, task_focus: str = "all") -> list[dict[str, str]]:
    """Build chat messages for structured planning output."""
    system = (
        "You generate structured music planning JSON for ACE-Step. "
        "Return only valid JSON with keys: caption, lyrics, bpm, duration, "
        "key_scale, time_signature, vocal_language, instrumental. "
        "Caption must be a narrative production brief (1-2 sentences) that includes: "
        "genre/mood, linear arrangement arc (e.g. intro -> build -> chorus/drop -> outro), "
        "core instrumentation, vocalist timbre/delivery when vocals are present, and "
        "mix/energy trajectory. Keep it concise but specific. "
        "If instrumental=true, set lyrics to [Instrumental]."
    )
    focus_guidance = _build_task_focus_guidance(task_focus=task_focus)
    user = (
        f"Task focus: {task_focus}\n"
        f"User intent:\n{intent}\n\n"
        f"{focus_guidance}\n"
        "Output JSON only. No markdown, no commentary."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def request_glm_plan(
    *,
    api_key: str,
    intent: str,
    model: str,
    base_url: str,
    timeout_sec: int = 60,
    task_focus: str = "all",
    protocol: str = "openai_chat",
) -> GlmPlan:
    """Request structured planning output from external chat-completions endpoints."""
    if not intent.strip():
        raise GlmClientError("Intent cannot be empty.")

    messages = build_planning_messages(intent=intent, task_focus=task_focus)
    payload, headers = _build_request_for_protocol(
        protocol=protocol,
        api_key=api_key,
        model=model,
        messages=messages,
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
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        guidance = _build_http_error_guidance(
            detail=detail,
            model=model,
            base_url=base_url,
        )
        request_target = f" | model={model} | endpoint={base_url}"
        raise GlmClientError(
            f"GLM request failed: HTTP {exc.code} {detail[:200]}{request_target}{guidance}"
        ) from exc
    except error.URLError as exc:
        raise GlmClientError(f"GLM request failed: {exc.reason}") from exc

    return parse_glm_chat_response(raw, protocol=protocol)


def parse_glm_chat_response(raw_response: str, protocol: str = "openai_chat") -> GlmPlan:
    """Parse external response body into normalized ``GlmPlan``."""
    content = _extract_protocol_message_content(raw_response=raw_response, protocol=protocol)
    return _parse_plan_from_content(content)


def build_acestep_generation_payload(plan: GlmPlan) -> dict[str, Any]:
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


def _build_request_for_protocol(
    *,
    protocol: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build protocol-specific request payload and HTTP headers."""
    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol == "anthropic_messages":
        if not api_key:
            raise GlmClientError("Missing API key for anthropic_messages protocol.")
        payload = {
            "model": model,
            "max_tokens": int(os.getenv("ACESTEP_ANTHROPIC_MAX_TOKENS", "1024")),
            "temperature": 0.4,
            "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
        }
        return payload, {
            "x-api-key": api_key,
            "anthropic-version": os.getenv("ACESTEP_ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return payload, headers


def _extract_protocol_message_content(*, raw_response: str, protocol: str) -> str:
    """Extract assistant text content from protocol-specific API responses."""
    try:
        outer = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise GlmClientError("Invalid GLM response shape.") from exc

    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol == "anthropic_messages":
        try:
            blocks = outer["content"]
            if isinstance(blocks, list):
                text_chunks = [
                    block.get("text", "")
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(chunk for chunk in text_chunks if chunk)
            if isinstance(blocks, str):
                return blocks
        except (KeyError, TypeError) as exc:
            raise GlmClientError("Invalid GLM response shape.") from exc
        raise GlmClientError("Invalid GLM response shape.")

    try:
        return outer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GlmClientError("Invalid GLM response shape.") from exc


def _parse_plan_from_content(content: str) -> GlmPlan:
    """Parse JSON-plan text content into normalized plan fields."""
    inner_json = _extract_json_block(content)
    try:
        obj = json.loads(inner_json)
    except json.JSONDecodeError as exc:
        raise GlmClientError("GLM content is not valid JSON.") from exc

    caption = str(obj.get("caption") or "").strip()
    lyrics = str(obj.get("lyrics") or "").strip()
    instrumental = _to_bool(obj.get("instrumental"))
    bpm = _to_int(obj.get("bpm"))
    duration = _to_float(obj.get("duration"))
    key_scale = str(obj.get("key_scale") or "").strip()
    time_signature = str(obj.get("time_signature") or "").strip()
    vocal_language = str(obj.get("vocal_language") or "").strip()

    if instrumental and not lyrics:
        lyrics = "[Instrumental]"

    return GlmPlan(
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        duration=duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        instrumental=instrumental,
    )


def _extract_json_block(content: str) -> str:
    """Extract JSON object string from plain or markdown-fenced content."""
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1)

    first = content.find("{")
    last = content.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return content
    return content[first : last + 1]


def _to_bool(value: Any) -> bool:
    """Coerce common bool-like values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _to_int(value: Any) -> int | None:
    """Coerce optional integer field."""
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Coerce optional float field."""
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_http_error_guidance(*, detail: str, model: str, base_url: str) -> str:
    """Return targeted guidance for common API error payloads."""
    if not detail:
        return ""
    try:
        payload = json.loads(detail)
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = str(err.get("code", "")).strip()
    except json.JSONDecodeError:
        code = ""

    if code == "1211":
        return (
            " | Model not found. Try --model glm-4.5-flash "
            "(or set ACESTEP_GLM_MODEL) and verify your account has access."
        )
    if code == "1113":
        is_coding_endpoint = "api/coding/paas/v4" in (base_url or "").lower()
        endpoint_hint = ""
        if not is_coding_endpoint:
            endpoint_hint = (
                " Current endpoint looks like general API; coding-plan quota requires "
                "https://api.z.ai/api/coding/paas/v4/chat/completions."
            )
        return (
            " | 1113 means billing quota is unavailable for this request."
            " If you rely on GLM Coding Plan quota, use the dedicated coding endpoint and a "
            "supported coding-plan model."
            f"{endpoint_hint} Otherwise top up API balance/resource package."
        )
    return ""


def _build_task_focus_guidance(*, task_focus: str) -> str:
    """Return task-focus specific generation guidance for planning prompts."""
    normalized_focus = (task_focus or "all").strip().lower()
    if normalized_focus == "format":
        return (
            "For format focus: preserve user intent, then improve clarity and musical specificity. "
            "Do not change the core genre/mood unless required for coherence."
        )
    return (
        "For all-task focus: produce complete caption/lyrics/metadata that can directly drive "
        "music generation."
    )
