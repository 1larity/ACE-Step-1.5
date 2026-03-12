"""Service initialization and tier management for generation handlers.

Contains functions for initializing the DiT/LLM services, refreshing
checkpoints, and handling GPU tier changes.
"""

from __future__ import annotations

import os

import gradio as gr
import torch

from acestep.gpu_config import get_global_gpu_config
from acestep.text_tasks.external_lm_mode import (
    activate_external_lm_mode,
    deactivate_external_lm_mode,
    get_external_lm_choices,
    is_external_lm_active,
    is_external_lm_model,
)

from .service_init_helpers import (
    build_model_settings,
    build_runtime_limit_updates,
    build_tier_updates,
    project_root_from,
    resolve_local_lm_request,
    resolve_quantization_value,
)


def refresh_checkpoints(dit_handler):
    """Refresh available checkpoints."""
    return gr.update(choices=dit_handler.get_available_checkpoints())


def init_service_wrapper(
    dit_handler,
    llm_handler,
    checkpoint,
    config_path,
    device,
    init_llm,
    lm_model_path,
    backend,
    use_flash_attention,
    offload_to_cpu,
    offload_dit_to_cpu,
    compile_model,
    quantization,
    mlx_dit=True,
    current_mode=None,
    current_batch_size=None,
):
    """Wrapper for service initialization."""
    quantization, quant_value = resolve_quantization_value(quantization=quantization, device=device)
    gpu_config = get_global_gpu_config()
    should_initialize_local_lm, lm_device, backend = resolve_local_lm_request(
        gpu_config=gpu_config,
        init_llm=bool(init_llm and not is_external_lm_model(lm_model_path)),
        lm_model_path=lm_model_path,
        device=device,
        backend=backend,
    )
    project_root = project_root_from(os.path.abspath(__file__))
    status, enable = dit_handler.initialize_service(
        project_root,
        config_path,
        device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
        quantization=quant_value,
        use_mlx_dit=mlx_dit,
    )
    external_selection = activate_external_lm_mode(lm_model_path) if is_external_lm_model(lm_model_path) else None
    if external_selection is None:
        deactivate_external_lm_mode()
    if should_initialize_local_lm:
        lm_status, _ = llm_handler.initialize(
            checkpoint_dir=os.path.join(project_root, "checkpoints"),
            lm_model_path=lm_model_path,
            backend=backend,
            device=lm_device,
            offload_to_cpu=offload_to_cpu,
            dtype=None,
        )
        status += f"\n{lm_status}"
    if external_selection is not None:
        status += f"\nExternal LM provider active: {external_selection.provider}:{external_selection.model}"
    accordion_state = gr.Accordion(open=not (dit_handler.model is not None))
    model_type_settings = build_model_settings(dit_handler=dit_handler, config_path=config_path, current_mode=current_mode)
    duration_update, batch_update, status_suffix, local_lm_initialized = build_runtime_limit_updates(
        gpu_config=get_global_gpu_config(),
        llm_handler=llm_handler,
        current_batch_size=current_batch_size,
    )
    think_interactive = local_lm_initialized or is_external_lm_active()
    return (
        status + status_suffix,
        gr.update(interactive=enable),
        accordion_state,
        *model_type_settings,
        duration_update,
        batch_update,
        gr.update(interactive=think_interactive, value=think_interactive),
    )


def on_tier_change(selected_tier, llm_handler=None):
    """Handle manual tier override from the UI dropdown."""
    all_disk_models = llm_handler.get_available_5hz_lm_models() if llm_handler else []
    merged_models = list(dict.fromkeys((all_disk_models or []) + get_external_lm_choices()))
    return build_tier_updates(selected_tier=selected_tier, all_disk_models=merged_models)
