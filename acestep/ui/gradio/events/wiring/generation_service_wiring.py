"""Generation service-layer event wiring helpers.

This module contains wiring related to service initialization, LoRA controls,
auto-checkbox controls, and visibility updates for generation components.
"""

from typing import Any

import gradio as gr

from acestep.text_tasks.external_lm_mode import (
    activate_external_lm_mode,
    deactivate_external_lm_mode,
    is_external_lm_model,
)

from .. import generation_handlers as gen_h
from ...i18n import get_i18n
from .context import GenerationWiringContext, build_auto_checkbox_inputs, build_auto_checkbox_outputs
from .generation_service_wiring_registration import (
    register_dataset_handlers,
    register_external_lm_handlers,
    register_lora_handlers,
    register_service_init_handlers,
)
from .generation_service_wiring_ui import register_auto_checkbox_handlers, register_visibility_handlers


def register_generation_service_handlers(
    context: GenerationWiringContext,
) -> tuple[list[Any], list[Any]]:
    """Register generation service/init handlers and return auto-checkbox lists."""
    dataset_section = context.dataset_section
    generation_section = context.generation_section
    results_section = context.results_section
    dit_handler = context.dit_handler
    llm_handler = context.llm_handler
    dataset_handler = context.dataset_handler

    register_dataset_handlers(dataset_section=dataset_section, dataset_handler=dataset_handler)
    register_service_init_handlers(
        generation_section=generation_section,
        gen_h=gen_h,
        dit_handler=dit_handler,
        llm_handler=llm_handler,
    )

    generation_section["language_dropdown"].change(
        fn=lambda language: _apply_runtime_language(language),
        inputs=[generation_section["language_dropdown"]],
        outputs=[generation_section["language_dropdown"]],
    )
    generation_section["lm_model_path"].change(
        fn=_sync_external_lm_mode_from_dropdown,
        inputs=[generation_section["lm_model_path"]],
        outputs=[generation_section["init_llm_checkbox"]],
    )
    register_external_lm_handlers(
        generation_section=generation_section,
        gen_h=gen_h,
        llm_handler=llm_handler,
    )
    generation_section["external_lm_save_btn"].click(
        fn=lambda *args: gen_h.save_external_lm_settings_from_ui(*args, llm_handler=llm_handler),
        inputs=[
            generation_section["external_lm_provider_dropdown"],
            generation_section["external_lm_protocol_dropdown"],
            generation_section["external_lm_model_input"],
            generation_section["external_lm_base_url_input"],
            generation_section["external_lm_api_key_input"],
            generation_section["external_lm_store_passphrase_input"],
            generation_section["external_lm_save_passphrase_checkbox"],
        ],
        outputs=[
            generation_section["external_lm_status"],
            generation_section["external_lm_api_key_input"],
            generation_section["external_lm_store_passphrase_input"],
            generation_section["lm_model_path"],
        ],
    )

    register_lora_handlers(generation_section=generation_section, dit_handler=dit_handler)
    auto_checkbox_outputs = build_auto_checkbox_outputs(context)
    auto_checkbox_inputs = build_auto_checkbox_inputs(context)
    register_auto_checkbox_handlers(
        generation_section=generation_section,
        gen_h=gen_h,
        auto_checkbox_outputs=auto_checkbox_outputs,
    )
    register_visibility_handlers(
        generation_section=generation_section,
        results_section=results_section,
        gen_h=gen_h,
        llm_handler=llm_handler,
        sync_lm_selection_from_init_checkbox=_sync_lm_selection_from_init_checkbox,
    )
    return auto_checkbox_inputs, auto_checkbox_outputs


def _apply_runtime_language(language: str) -> dict[str, Any]:
    """Update global i18n language for runtime-generated messages.

    Args:
        language: Selected UI language code from the language dropdown.

    Returns:
        A ``gr.update`` payload preserving the selected dropdown value.
    """
    get_i18n(language)
    return gr.update(value=language)


def _sync_external_lm_mode_from_dropdown(lm_model_path: str) -> dict[str, Any]:
    """Keep external LM mode in sync with the selected LM dropdown value."""
    if is_external_lm_model(lm_model_path):
        activate_external_lm_mode(lm_model_path)
        return gr.update(value=False)
    deactivate_external_lm_mode()
    return gr.update(value=True)


def _sync_lm_selection_from_init_checkbox(
    init_llm: bool,
    lm_model_path: str,
    llm_handler: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep 5Hz-init checkbox and LM model selection mutually consistent."""
    if not init_llm:
        return gr.update(value=False), gr.update()
    if not is_external_lm_model(lm_model_path):
        return gr.update(value=True), gr.update()

    local_models = llm_handler.get_available_5hz_lm_models() if llm_handler else []
    local_model_choices = [model for model in (local_models or []) if model]
    if not local_model_choices:
        return gr.update(value=False), gr.update()
    deactivate_external_lm_mode()
    return gr.update(value=True), gr.update(value=local_model_choices[0])
