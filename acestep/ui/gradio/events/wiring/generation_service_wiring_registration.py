"""Helper registration blocks for generation service wiring."""

from __future__ import annotations

from typing import Any

import gradio as gr


def register_dataset_handlers(*, dataset_section: dict[str, Any], dataset_handler: Any) -> None:
    """Register dataset import handlers for the generation tab."""
    dataset_section["import_dataset_btn"].click(
        fn=dataset_handler.import_dataset,
        inputs=[dataset_section["dataset_type"]],
        outputs=[dataset_section["data_status"]],
    )


def register_external_lm_handlers(*, generation_section: dict[str, Any], gen_h: Any, llm_handler: Any) -> None:
    """Register external LM setup handlers other than the save action."""
    generation_section["external_lm_provider_dropdown"].change(
        fn=lambda provider: gen_h.load_external_lm_provider_defaults_with_lm_dropdown(
            provider,
            llm_handler=llm_handler,
        ),
        inputs=[generation_section["external_lm_provider_dropdown"]],
        outputs=[
            generation_section["external_lm_protocol_dropdown"],
            generation_section["external_lm_model_input"],
            generation_section["external_lm_base_url_input"],
            generation_section["external_lm_status"],
            generation_section["lm_model_path"],
        ],
    )
    generation_section["external_lm_defaults_btn"].click(
        fn=lambda provider: gen_h.load_external_lm_provider_defaults_with_lm_dropdown(
            provider,
            llm_handler=llm_handler,
        ),
        inputs=[generation_section["external_lm_provider_dropdown"]],
        outputs=[
            generation_section["external_lm_protocol_dropdown"],
            generation_section["external_lm_model_input"],
            generation_section["external_lm_base_url_input"],
            generation_section["external_lm_status"],
            generation_section["lm_model_path"],
        ],
    )
    generation_section["external_lm_fetch_models_btn"].click(
        fn=lambda *args: gen_h.fetch_external_lm_models_from_ui_with_lm_dropdown(*args, llm_handler=llm_handler),
        inputs=[
            generation_section["external_lm_provider_dropdown"],
            generation_section["external_lm_protocol_dropdown"],
            generation_section["external_lm_base_url_input"],
            generation_section["external_lm_api_key_input"],
            generation_section["external_lm_model_input"],
        ],
        outputs=[
            generation_section["external_lm_model_input"],
            generation_section["external_lm_status"],
            generation_section["lm_model_path"],
        ],
    )
    generation_section["external_lm_doctor_btn"].click(
        fn=gen_h.check_external_lm_runtime_from_ui,
        inputs=[
            generation_section["external_lm_provider_dropdown"],
            generation_section["external_lm_protocol_dropdown"],
            generation_section["external_lm_model_input"],
            generation_section["external_lm_base_url_input"],
        ],
        outputs=[generation_section["external_lm_status"]],
    )


def register_service_init_handlers(*, generation_section: dict[str, Any], gen_h: Any, dit_handler: Any, llm_handler: Any) -> None:
    """Register checkpoint, init, and tier service handlers."""
    generation_section["refresh_btn"].click(
        fn=lambda: gen_h.refresh_checkpoints(dit_handler),
        outputs=[generation_section["checkpoint_dropdown"]],
    )
    generation_section["config_path"].change(
        fn=gen_h.update_model_type_settings,
        inputs=[generation_section["config_path"], generation_section["generation_mode"]],
        outputs=[
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["use_adg"],
            generation_section["shift"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["task_type"],
            generation_section["generation_mode"],
            generation_section["init_llm_checkbox"],
        ],
    )
    generation_section["tier_dropdown"].change(
        fn=lambda tier: gen_h.on_tier_change(tier, llm_handler),
        inputs=[generation_section["tier_dropdown"]],
        outputs=[
            generation_section["offload_to_cpu_checkbox"],
            generation_section["offload_dit_to_cpu_checkbox"],
            generation_section["compile_model_checkbox"],
            generation_section["quantization_checkbox"],
            generation_section["backend_dropdown"],
            generation_section["lm_model_path"],
            generation_section["init_llm_checkbox"],
            generation_section["batch_size_input"],
            generation_section["audio_duration"],
            generation_section["gpu_info_display"],
        ],
    )
    generation_section["init_btn"].click(
        fn=lambda *args: gen_h.init_service_wrapper(dit_handler, llm_handler, *args),
        inputs=[
            generation_section["checkpoint_dropdown"],
            generation_section["config_path"],
            generation_section["device"],
            generation_section["init_llm_checkbox"],
            generation_section["lm_model_path"],
            generation_section["backend_dropdown"],
            generation_section["use_flash_attention_checkbox"],
            generation_section["offload_to_cpu_checkbox"],
            generation_section["offload_dit_to_cpu_checkbox"],
            generation_section["compile_model_checkbox"],
            generation_section["quantization_checkbox"],
            generation_section["mlx_dit_checkbox"],
            generation_section["generation_mode"],
            generation_section["batch_size_input"],
        ],
        outputs=[
            generation_section["init_status"],
            generation_section["generate_btn"],
            generation_section["service_config_accordion"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["use_adg"],
            generation_section["shift"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["task_type"],
            generation_section["generation_mode"],
            generation_section["init_llm_checkbox"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["think_checkbox"],
        ],
    )


def register_lora_handlers(*, generation_section: dict[str, Any], dit_handler: Any) -> None:
    """Register LoRA load, unload, and scale handlers."""
    generation_section["load_lora_btn"].click(
        fn=dit_handler.load_lora,
        inputs=[generation_section["lora_path"]],
        outputs=[generation_section["lora_status"]],
    ).then(fn=lambda: gr.update(value=True), outputs=[generation_section["use_lora_checkbox"]])
    generation_section["unload_lora_btn"].click(
        fn=dit_handler.unload_lora,
        outputs=[generation_section["lora_status"]],
    ).then(fn=lambda: gr.update(value=False), outputs=[generation_section["use_lora_checkbox"]])
    generation_section["use_lora_checkbox"].change(
        fn=dit_handler.set_use_lora,
        inputs=[generation_section["use_lora_checkbox"]],
        outputs=[generation_section["lora_status"]],
    )
    generation_section["lora_scale_slider"].change(
        fn=dit_handler.set_lora_scale,
        inputs=[generation_section["lora_scale_slider"]],
        outputs=[generation_section["lora_status"]],
    )
