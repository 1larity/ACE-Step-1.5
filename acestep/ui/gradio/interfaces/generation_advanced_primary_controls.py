"""Primary advanced-settings controls for generation UI."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t

from .generation_advanced_lora_tabs import build_lora_tabbed_controls


def build_lora_controls() -> dict[str, Any]:
    """Create tabbed LoRA adapter controls for loading and scaling inference adapters."""
    with gr.Accordion(t("generation.lora_accordion_title"), open=False, elem_classes=["has-info-container"]):
        lora_ui = build_lora_tabbed_controls()

    first_slot = lora_ui["lora_slots"][0]
    return {
        "lora_path": first_slot["lora_path"],
        "load_lora_btn": first_slot["load_lora_btn"],
        "unload_lora_btn": first_slot["unload_lora_btn"],
        "use_lora_checkbox": first_slot["use_lora_checkbox"],
        "lora_scale_slider": first_slot["lora_scale_slider"],
        "lora_status": first_slot["lora_status"],
        "lora_slots": lora_ui["lora_slots"],
        "lora_add_slot_btn": lora_ui["lora_add_slot_btn"],
        "lora_slot_count_state": lora_ui["lora_slot_count_state"],
        "lora_slot_visibility_state": lora_ui["lora_slot_visibility_state"],
        "lora_tabs": lora_ui["lora_tabs"],
        "lora_tab_items": lora_ui["lora_tab_items"],
    }


def build_lm_controls(service_mode: bool) -> dict[str, Any]:
    """Create language-model generation controls for advanced settings.

    Args:
        service_mode: Whether the UI is running in service mode (disables some controls).

    Returns:
        A component map containing LM sampling, CoT, negative prompt, and batch controls.
    """

    with gr.Accordion(t("generation.advanced_lm_section"), open=False, elem_classes=["has-info-container"]):
        with gr.Row():
            lm_temperature = gr.Slider(
                label=t("generation.lm_temperature_label"),
                minimum=0.0,
                maximum=2.0,
                value=0.85,
                step=0.1,
                scale=1,
                info=t("generation.lm_temperature_info"),
                elem_classes=["has-info-container"],
            )
            lm_cfg_scale = gr.Slider(
                label=t("generation.lm_cfg_scale_label"),
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                scale=1,
                info=t("generation.lm_cfg_scale_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            lm_top_k = gr.Slider(
                label=t("generation.lm_top_k_label"),
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                scale=1,
                info=t("generation.lm_top_k_info"),
                elem_classes=["has-info-container"],
            )
            lm_top_p = gr.Slider(
                label=t("generation.lm_top_p_label"),
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.01,
                scale=1,
                info=t("generation.lm_top_p_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Row():
            lm_negative_prompt = gr.Textbox(
                label=t("generation.lm_negative_prompt_label"),
                value="NO USER INPUT",
                placeholder=t("generation.lm_negative_prompt_placeholder"),
                info=t("generation.lm_negative_prompt_info"),
                elem_classes=["has-info-container"],
                lines=2,
            )
        with gr.Row():
            use_cot_metas = gr.Checkbox(
                label=t("generation.cot_metas_label"),
                value=True,
                info=t("generation.cot_metas_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            use_cot_language = gr.Checkbox(
                label=t("generation.cot_language_label"),
                value=True,
                info=t("generation.cot_language_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            constrained_decoding_debug = gr.Checkbox(
                label=t("generation.constrained_debug_label"),
                value=False,
                info=t("generation.constrained_debug_info"),
                scale=1,
                interactive=not service_mode,
            )
        with gr.Row():
            allow_lm_batch = gr.Checkbox(
                label=t("generation.parallel_thinking_label"),
                value=True,
                info=t("generation.parallel_thinking_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )
            use_cot_caption = gr.Checkbox(
                label=t("generation.caption_rewrite_label"),
                value=False,
                info=t("generation.caption_rewrite_info"),
                scale=1,
                elem_classes=["has-info-container"],
            )

    return {
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "use_cot_caption": use_cot_caption,
    }
