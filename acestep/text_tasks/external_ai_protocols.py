"""Protocol and intent-signal helpers for external AI request building."""

from __future__ import annotations

from .external_ai_types import ExternalAIClientError

SUPPORTED_REQUEST_PROTOCOLS = frozenset({"anthropic_messages", "openai_chat"})


def extract_intent_signal_text(intent: str) -> str:
    """Return the caption and explicit metadata lines used for intent heuristics."""

    lines = [(line or "").strip() for line in (intent or "").splitlines()]
    signal_lines = [
        line.partition(":")[2].strip()
        for line in lines
        if line.lower().startswith(("caption:", "instrumental:", "vocal_language:"))
    ]
    if signal_lines:
        return "\n".join(signal_lines).lower().strip()
    return (intent or "").strip().lower()


def normalize_request_protocol(protocol: str) -> str:
    """Validate and normalize a supported request protocol identifier."""

    normalized_protocol = (protocol or "openai_chat").strip().lower()
    if normalized_protocol not in SUPPORTED_REQUEST_PROTOCOLS:
        raise ExternalAIClientError(
            f"Unsupported external request protocol: {normalized_protocol or '<empty>'}."
        )
    return normalized_protocol


def require_message_pair(messages: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Return the expected system and user messages for provider request building."""

    if len(messages) < 2:
        raise ExternalAIClientError(
            "External planning request requires both system and user messages."
        )
    return messages[0], messages[1]
