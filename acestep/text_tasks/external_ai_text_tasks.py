"""LLM client helpers for planning caption/lyrics/metadata text tasks."""

from __future__ import annotations

import json
import os
import re
import socket

from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger

from acestep.lm_task_debug import is_lm_task_debug_enabled
from urllib import error, parse, request


class ExternalAIClientError(RuntimeError):
    """Raised when external API calls or response parsing fail."""


def _external_ai_debug_enabled() -> bool:
    """Return ``True`` when verbose LM task logging is enabled."""
    return is_lm_task_debug_enabled()


def _preview_text(text: str, limit: int = 600) -> str:
    """Return a compact single-line preview for logs and error messages."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _debug_log_external_ai_request(*, protocol: str, model: str, base_url: str, payload: dict[str, Any]) -> None:
    """Log sanitized outbound External AI request context when debug mode is enabled."""
    if not _external_ai_debug_enabled():
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
    if not _external_ai_debug_enabled():
        return
    logger.debug(
        "External AI raw response protocol={} endpoint={} body={} ",
        protocol,
        base_url,
        raw_response,
    )


@dataclass
class ExternalAIPlan:
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
        "core instrumentation, singer gender and delivery mood/timbre when vocals are present, and "
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
    payload, headers = _build_request_for_protocol(
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
        guidance = _build_http_error_guidance(
            detail=detail,
            model=model,
            base_url=base_url,
        )
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
    content = _extract_protocol_message_content(raw_response=raw_response, protocol=protocol)
    if _external_ai_debug_enabled():
        logger.debug(
            "External AI extracted content protocol={} preview={}",
            protocol,
            _preview_text(content),
        )
    return _parse_plan_from_content(content, task_focus=task_focus)


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


def _resolve_openai_thinking_payload() -> dict[str, str]:
    """Return provider-specific thinking controls for supported OpenAI-like endpoints."""
    configured = os.getenv("ACESTEP_EXTERNAL_AI_THINKING", "disabled").strip().lower()
    thinking_type = "enabled" if configured in {"1", "true", "yes", "on", "enabled"} else "disabled"
    return {"type": thinking_type}


def _supports_provider_thinking(*, protocol: str, base_url: str) -> bool:
    """Return whether the configured endpoint supports provider-specific thinking controls."""
    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol != "openai_chat":
        return False

    parsed = parse.urlparse(base_url or "")
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return "z.ai" in host or "/api/paas/" in path or "/api/coding/paas/" in path


def _build_request_for_protocol(
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
        return payload, {
            "x-api-key": api_key,
            "anthropic-version": os.getenv("ACESTEP_ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(os.getenv("ACESTEP_OPENAI_MAX_TOKENS", "3072")),
        "temperature": 0.4,
    }
    if _supports_provider_thinking(protocol=normalized_protocol, base_url=base_url):
        payload["thinking"] = _resolve_openai_thinking_payload()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return payload, headers


def _extract_protocol_message_content(*, raw_response: str, protocol: str) -> str:
    """Extract assistant text content from protocol-specific API responses."""
    try:
        outer = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ExternalAIClientError("Invalid External AI response shape.") from exc

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
            raise ExternalAIClientError("Invalid External AI response shape.") from exc
        raise ExternalAIClientError("Invalid External AI response shape.")

    try:
        choice = outer["choices"][0]
        message = choice["message"]
        content = message.get("content", "")
        if isinstance(content, list):
            text_chunks = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(chunk for chunk in text_chunks if chunk)
        if isinstance(content, str) and content.strip():
            return content

        reasoning_content = str(message.get("reasoning_content") or "").strip()
        finish_reason = str(choice.get("finish_reason") or "").strip().lower()
        if reasoning_content:
            hint = ""
            if finish_reason == "length":
                hint = (
                    " Increase ACESTEP_OPENAI_MAX_TOKENS or choose a model/profile that returns final content without exhausting reasoning tokens."
                )
            raise ExternalAIClientError(
                "External AI returned empty content after using its completion budget on reasoning."
                f"{hint} Reasoning preview: {_preview_text(reasoning_content, limit=400)}"
            )
        return str(content or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ExternalAIClientError("Invalid External AI response shape.") from exc


def _parse_plan_from_content(content: str, task_focus: str = "all") -> ExternalAIPlan:
    """Parse provider content into normalized plan fields."""
    normalized_focus = (task_focus or "all").strip().lower()
    if normalized_focus == "lyrics":
        plain_lyrics = _extract_plain_lyrics_content(content)
        if plain_lyrics is not None:
            return ExternalAIPlan(
                caption="",
                lyrics=plain_lyrics,
                bpm=None,
                duration=None,
                key_scale="",
                time_signature="",
                vocal_language="",
                instrumental=plain_lyrics.strip().lower() in {"instrumental", "[instrumental]"},
            )

    obj = _load_plan_json_object(content, task_focus=task_focus)

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

    return ExternalAIPlan(
        caption=caption,
        lyrics=lyrics,
        bpm=bpm,
        duration=duration,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        instrumental=instrumental,
    )


def _extract_plain_lyrics_content(content: str) -> str | None:
    """Return plain lyric text for lyrics-focused providers that ignore JSON instructions."""
    normalized = (content or "").strip()
    if not normalized:
        return None
    if normalized.lower() in {"instrumental", "[instrumental]"}:
        return "[Instrumental]"

    section_pattern = re.compile(
        r"(?im)^\s*(?:\[(?:verse|chorus|bridge|pre-chorus|intro|outro|hook|refrain|drop)[^\]]*\]|(?:verse\s*\d+|chorus|bridge|pre-chorus|intro|outro|hook|refrain|drop)\s*:)",
    )
    if not section_pattern.search(normalized):
        return None
    if "{" in normalized or "}" in normalized:
        return None
    return normalized


def _load_plan_json_object(content: str, task_focus: str = "all") -> dict[str, Any]:
    """Load the best JSON object candidate from provider text content."""
    last_error: json.JSONDecodeError | None = None
    candidates = _iter_json_candidates(content)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed

    content_preview = _preview_text(content)
    if _external_ai_debug_enabled():
        candidate_previews = [_preview_text(candidate, limit=240) for candidate in candidates[:5]]
        logger.debug(
            "External AI JSON parse failure content_preview={} candidates={} last_error={}",
            content_preview,
            candidate_previews,
            last_error,
        )
    normalized_focus = (task_focus or "all").strip().lower()
    strict_hint = ""
    if normalized_focus != "lyrics":
        strict_hint = f" Task focus '{normalized_focus}' requires valid JSON."
    raise ExternalAIClientError(
        f"External AI content is not valid JSON.{strict_hint} Content preview: {content_preview}"
    ) from last_error


def _iter_json_candidates(content: str) -> list[str]:
    """Return de-duplicated JSON candidates from wrapped model content."""
    normalized = _normalize_model_content(content)
    candidates: list[str] = []
    for candidate in [
        _extract_json_block(normalized),
        *_extract_balanced_json_objects(normalized),
    ]:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        candidates.append(cleaned)
        repaired = _repair_json_candidate(cleaned)
        if repaired != cleaned:
            candidates.append(repaired)
    return list(dict.fromkeys(candidates))


def _normalize_model_content(content: str) -> str:
    """Strip common reasoning wrappers before JSON extraction."""
    normalized = (content or "").strip().lstrip("﻿")
    normalized = re.sub(r"<think>.*?</think>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    normalized = re.sub(r"<analysis>.*?</analysis>", " ", normalized, flags=re.DOTALL | re.IGNORECASE)
    return normalized.strip()


def _extract_balanced_json_objects(content: str) -> list[str]:
    """Extract balanced top-level JSON object candidates from free-form text."""
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(content):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(content[start : index + 1])
                start = None

    return objects


def _repair_json_candidate(candidate: str) -> str:
    """Apply small non-destructive repairs for common provider JSON defects."""
    repaired = candidate.strip()
    repaired = repaired.replace("“", '"').replace("”", '"')
    repaired = repaired.replace("‘", "'").replace("’", "'")
    repaired = repaired.replace(" ", " ")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


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
        error_type = str(err.get("type", "")).strip().lower()
        message = str(err.get("message", "")).strip().lower()
    except json.JSONDecodeError:
        code = ""
        error_type = ""
        message = detail.strip().lower()

    normalized_base_url = (base_url or "").lower()
    is_openai_endpoint = "api.openai.com" in normalized_base_url
    quota_like_error = (
        code == "1113"
        or code == "insufficient_quota"
        or error_type == "insufficient_quota"
        or "insufficient_quota" in message
        or ("quota" in message and "insufficient" in message)
    )

    if code == "1211":
        return (
            " | Model not found. Try a valid provider model and verify your account has access."
        )
    if code == "1113":
        is_coding_endpoint = "api/coding/paas/v4" in normalized_base_url
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


def _build_task_focus_guidance(*, task_focus: str) -> str:
    """Return task-focus specific generation guidance for planning prompts."""
    normalized_focus = (task_focus or "all").strip().lower()
    if normalized_focus == "format":
        return (
            "For format focus: preserve user intent, then improve clarity and musical specificity. "
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




