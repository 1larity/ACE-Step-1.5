"""Shared types for external AI text-task request and parse flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ExternalAIClientError(RuntimeError):
    """Raised when external API calls or response parsing fail."""


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
