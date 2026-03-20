"""Caption and metadata helpers for external LM formatting flows."""

from __future__ import annotations

from typing import Any


def normalized_caption(text: str) -> str:
    """Return a normalized caption string for retry and equality checks."""

    return " ".join((text or "").strip().lower().split())


def caption_needs_retry(*, original_caption: str, generated_caption: str) -> bool:
    """Return whether the generated caption looks like a non-enhanced echo."""

    normalized_original = normalized_caption(original_caption)
    normalized_generated = normalized_caption(generated_caption)
    if not normalized_generated:
        return True
    if normalized_generated == normalized_original:
        return True
    return len(normalized_generated.split()) < max(6, len(normalized_original.split()) + 3)


def apply_user_metadata_overrides(*, plan: Any, user_metadata: dict[str, Any]) -> Any:
    """Preserve user-supplied metadata over any parsed provider drift."""

    if not user_metadata:
        return plan
    if user_metadata.get("bpm") not in (None, ""):
        try:
            plan.bpm = int(user_metadata["bpm"])
        except (TypeError, ValueError):
            pass
    if user_metadata.get("duration") not in (None, ""):
        try:
            plan.duration = float(user_metadata["duration"])
        except (TypeError, ValueError):
            pass
    if user_metadata.get("keyscale"):
        plan.key_scale = str(user_metadata["keyscale"]).strip()
    if user_metadata.get("timesignature"):
        plan.time_signature = str(user_metadata["timesignature"]).strip()
    if user_metadata.get("language"):
        plan.vocal_language = str(user_metadata["language"]).strip()
    return plan


def build_fallback_caption(*, caption: str, user_metadata: dict[str, Any]) -> str:
    """Build a simple local narrative fallback when the provider keeps echoing input."""

    source = (caption or "music piece").strip().rstrip(".")
    bpm = user_metadata.get("bpm")
    duration = user_metadata.get("duration")
    keyscale = user_metadata.get("keyscale")
    timesignature = user_metadata.get("timesignature")

    parts = [
        f"{source} unfolds as a fuller arranged track with a clear intro, developing verses,",
        "a stronger chorus or drop, and a shaped outro that resolves the energy naturally.",
    ]
    if bpm not in (None, ""):
        parts.append(f"The groove stays anchored around {bpm} BPM.")
    if timesignature:
        parts.append(f"The arrangement holds a {timesignature} pulse throughout.")
    if keyscale:
        parts.append(f"The harmony centers on {keyscale}.")
    if duration not in (None, ""):
        parts.append(f"The structure is paced for roughly {duration} seconds.")
    parts.append(
        "The mix should grow from a more focused opening into a fuller, more energetic peak before easing out."
    )
    return " ".join(parts)


def build_format_request_intent(
    *,
    caption: str,
    lyrics: str,
    user_metadata: dict[str, Any],
) -> str:
    """Build the format-mode user intent string for external provider requests."""

    intent_parts = [
        "Please format and enrich the following for ACE-Step generation.",
        f"Caption: {caption or ''}",
        f"Lyrics: {lyrics or ''}",
    ]
    for key in ("bpm", "duration", "keyscale", "timesignature", "language"):
        value = user_metadata.get(key)
        if value not in (None, "", "unknown"):
            intent_parts.append(f"{key}: {value}")
    return "\n".join(intent_parts)
