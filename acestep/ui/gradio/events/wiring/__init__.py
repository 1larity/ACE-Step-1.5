"""Wiring helpers for Gradio event registration.

This package provides shared context and list-builder helpers used by the
event wiring facade in ``acestep.ui.gradio.events``.
"""

from .context import (
    GenerationWiringContext,
    TrainingWiringContext,
    build_auto_checkbox_inputs,
    build_auto_checkbox_outputs,
    build_mode_ui_outputs,
)

__all__ = [
    "GenerationWiringContext",
    "TrainingWiringContext",
    "build_auto_checkbox_inputs",
    "build_auto_checkbox_outputs",
    "build_mode_ui_outputs",
]
