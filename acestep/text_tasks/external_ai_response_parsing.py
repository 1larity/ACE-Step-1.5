"""Response parsing helpers for external AI text-task integrations."""

from __future__ import annotations

import json

from .external_ai_debug import preview_text
from .external_ai_json_parsing import (
    extract_plain_lyrics_content,
    load_plan_json_object,
    to_bool,
    to_float,
    to_int,
)
from .external_ai_types import ExternalAIClientError, ExternalAIPlan


def extract_protocol_message_content(*, raw_response: str, protocol: str) -> str:
    """Extract assistant text content from protocol-specific API responses."""
    try:
        outer = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ExternalAIClientError("Invalid External AI response shape.") from exc

    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol == "anthropic_messages":
        return _extract_anthropic_content(outer)
    return _extract_openai_style_content(outer)


def parse_plan_from_content(content: str, task_focus: str = "all") -> ExternalAIPlan:
    """Parse provider content into normalized plan fields."""
    normalized_focus = (task_focus or "all").strip().lower()
    if normalized_focus == "lyrics":
        plain_lyrics = extract_plain_lyrics_content(content)
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

    obj = load_plan_json_object(content, task_focus=task_focus)
    caption = str(obj.get("caption") or "").strip()
    lyrics = str(obj.get("lyrics") or "").strip()
    instrumental = to_bool(obj.get("instrumental"))
    bpm = to_int(obj.get("bpm"))
    duration = to_float(obj.get("duration"))
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


def _extract_anthropic_content(outer: dict[str, object]) -> str:
    """Return text content from Anthropic-style message responses."""
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


def _extract_openai_style_content(outer: dict[str, object]) -> str:
    """Return text content from OpenAI-compatible chat completion responses."""
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
                    " Increase ACESTEP_OPENAI_MAX_TOKENS or choose a model/profile that returns "
                    "final content without exhausting reasoning tokens."
                )
            raise ExternalAIClientError(
                "External AI returned empty content after using its completion budget on reasoning."
                f"{hint} Reasoning preview: {preview_text(reasoning_content, limit=400)}"
            )
        return str(content or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ExternalAIClientError("Invalid External AI response shape.") from exc
