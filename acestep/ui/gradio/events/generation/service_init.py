"""Service init UI helpers for generation handlers."""

import os
import sys

import gradio as gr
from loguru import logger

from acestep.gpu_config import get_global_gpu_config, is_lm_model_size_allowed

from .model_config import get_model_type_ui_settings, is_pure_base_model
from .service_init_quantization import _select_quantization_value
from .service_init_tier import on_tier_change


def refresh_checkpoints(dit_handler):
    """Refresh available checkpoints."""
    return gr.update(choices=dit_handler.get_available_checkpoints())


def init_service_wrapper(
    dit_handler, llm_handler, checkpoint, config_path, device, init_llm, lm_model_path,
    backend, use_flash_attention, offload_to_cpu, offload_dit_to_cpu, compile_model,
    quantization, mlx_dit=True, current_mode=None, current_batch_size=None,
):
    """Initialize services and return UI updates for the generation tab."""
    quant_value = _select_quantization_value(
        quantization_enabled=quantization,
        device=device,
    )
    gpu_config = get_global_gpu_config()

    if sys.platform == "darwin":
        if compile_model:
            logger.info(
                "macOS detected: torch.compile not supported; compilation will use mx.compile via MLX."
            )
        if quantization:
            logger.info("macOS detected: disabling INT8 quantization (torchao incompatible with MPS)")
            quantization = False
            quant_value = None

    if init_llm:
        if not gpu_config.available_lm_models:
            logger.warning(
                f"GPU tier {gpu_config.tier} ({gpu_config.gpu_memory_gb:.1f}GB) does not support LM on GPU. "
                "Falling back to CPU for LM initialization."
            )
            lm_device = "cpu"
        else:
            lm_device = device

    if init_llm and lm_model_path and gpu_config.available_lm_models:
        if not is_lm_model_size_allowed(lm_model_path, gpu_config.available_lm_models):
            logger.warning(
                f"LM model {lm_model_path} is not in the recommended list for tier {gpu_config.tier} "
                f"(recommended: {gpu_config.available_lm_models}). Proceeding with user selection - "
                "this may cause high VRAM usage or OOM."
            )

    if init_llm and gpu_config.lm_backend_restriction == "pt_mlx_only" and backend == "vllm":
        backend = gpu_config.recommended_backend
        logger.warning(
            f"vllm backend not supported for tier {gpu_config.tier} "
            f"(VRAM too low for KV cache), falling back to {backend}"
        )

    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))))
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

    if init_llm:
        lm_status, _lm_success = llm_handler.initialize(
            checkpoint_dir=os.path.join(project_root, "checkpoints"),
            lm_model_path=lm_model_path,
            backend=backend,
            device=lm_device,
            offload_to_cpu=offload_to_cpu,
            dtype=None,
        )
        status += f"\n{lm_status}"

    is_model_initialized = dit_handler.model is not None
    accordion_state = gr.Accordion(open=not is_model_initialized)
    is_turbo = dit_handler.is_turbo_model()
    is_pure_base = is_pure_base_model((config_path or "").lower())
    model_type_settings = get_model_type_ui_settings(
        is_turbo, current_mode=current_mode, is_pure_base=is_pure_base
    )

    gpu_config = get_global_gpu_config()
    lm_actually_initialized = llm_handler.llm_initialized if llm_handler else False
    max_duration = gpu_config.max_duration_with_lm if lm_actually_initialized else gpu_config.max_duration_without_lm
    max_batch = gpu_config.max_batch_size_with_lm if lm_actually_initialized else gpu_config.max_batch_size_without_lm
    duration_update = gr.update(
        maximum=float(max_duration),
        info=f"Duration in seconds (-1 for auto). Max: {max_duration}s / {max_duration // 60} min.",
        elem_classes=["has-info-container"],
    )

    if current_batch_size is not None:
        try:
            batch_value_int = int(current_batch_size)
            if batch_value_int >= 1:
                batch_value = min(batch_value_int, max_batch)
                if batch_value_int > max_batch:
                    logger.warning(
                        f"Batch size {batch_value_int} exceeds GPU limit {max_batch}, clamping to {batch_value}"
                    )
            else:
                logger.warning(
                    f"Invalid batch size {batch_value_int} (must be >= 1), using default {min(2, max_batch)}"
                )
                batch_value = min(2, max_batch)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid batch size {current_batch_size!r}, using default {min(2, max_batch)}"
            )
            batch_value = min(2, max_batch)
    else:
        batch_value = min(2, max_batch)

    batch_update = gr.update(
        value=batch_value,
        maximum=max_batch,
        info=f"Number of samples to generate (Max: {max_batch}).",
        elem_classes=["has-info-container"],
    )

    status += f"\nGPU Config: tier={gpu_config.tier}, max_duration={max_duration}s, max_batch={max_batch}"
    if gpu_config.available_lm_models:
        status += f", available_lm={gpu_config.available_lm_models}"
    else:
        status += ", LM not available for this GPU tier"

    return (
        status,
        gr.update(interactive=enable),
        accordion_state,
        *model_type_settings,
        duration_update,
        batch_update,
        gr.update(interactive=lm_actually_initialized, value=lm_actually_initialized),
    )
