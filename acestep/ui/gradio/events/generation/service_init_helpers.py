"""Service init helper calculations for generation handlers."""

from __future__ import annotations

import os
import sys
from typing import Any

import gradio as gr
import torch
from loguru import logger

from acestep.gpu_config import find_best_lm_model_on_disk, get_gpu_device_name, get_gpu_config_for_tier, is_lm_model_size_allowed, set_global_gpu_config, GPU_TIER_CONFIGS, GPU_TIER_LABELS
from acestep.ui.gradio.i18n import t
from .model_config import get_model_type_ui_settings, is_pure_base_model


def resolve_quantization_value(*, quantization: bool, device: str) -> tuple[bool, str | None]:
    """Resolve quantization state for the current platform and GPU capability."""
    quant_value = "int8_weight_only" if quantization else None
    if quantization and device in {"auto", "cuda"}:
        try:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                if major < 7:
                    logger.info("Pre-Ampere CUDA detected: using int8_weight_only quantization for stability")
        except Exception:
            pass
    if sys.platform == "darwin":
        if quantization:
            logger.info("macOS detected: disabling INT8 quantization (torchao incompatible with MPS)")
            return False, None
    return quantization, quant_value


def resolve_local_lm_request(*, gpu_config: Any, init_llm: bool, lm_model_path: str | None, device: str, backend: str) -> tuple[bool, str | None, str]:
    """Resolve whether and how the local LM should be initialized."""
    should_initialize_local_lm = bool(init_llm and not str(lm_model_path or "").startswith("external:"))
    lm_device = None
    if should_initialize_local_lm:
        lm_device = "cpu" if not gpu_config.available_lm_models else device
        if not gpu_config.available_lm_models:
            logger.warning(
                f"GPU tier {gpu_config.tier} ({gpu_config.gpu_memory_gb:.1f}GB) does not support LM on GPU. Falling back to CPU for LM initialization."
            )
        elif lm_model_path and not is_lm_model_size_allowed(lm_model_path, gpu_config.available_lm_models):
            logger.warning(
                f"LM model {lm_model_path} is not in the recommended list for tier {gpu_config.tier} (recommended: {gpu_config.available_lm_models}). Proceeding with user selection; this may cause high VRAM usage or OOM."
            )
        if gpu_config.lm_backend_restriction == "pt_mlx_only" and backend == "vllm":
            backend = gpu_config.recommended_backend
            logger.warning(
                f"vllm backend not supported for tier {gpu_config.tier} (VRAM too low for KV cache), falling back to {backend}"
            )
    return should_initialize_local_lm, lm_device, backend


def project_root_from(current_file: str) -> str:
    """Return the ACE-Step project root from this module path."""
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        )
    )


def build_model_settings(*, dit_handler: Any, config_path: str, current_mode: Any) -> tuple[Any, ...]:
    """Build model-type UI settings for the initialized DiT model."""
    is_turbo = dit_handler.is_turbo_model()
    is_pure_base = is_pure_base_model((config_path or "").lower())
    return get_model_type_ui_settings(is_turbo, current_mode=current_mode, is_pure_base=is_pure_base)


def build_runtime_limit_updates(*, gpu_config: Any, llm_handler: Any, current_batch_size: Any) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    """Build GPU-config-aware duration/batch updates and status summary."""
    local_lm_initialized = llm_handler.llm_initialized if llm_handler else False
    max_duration = gpu_config.max_duration_with_lm if local_lm_initialized else gpu_config.max_duration_without_lm
    max_batch = gpu_config.max_batch_size_with_lm if local_lm_initialized else gpu_config.max_batch_size_without_lm
    duration_update = gr.update(
        maximum=float(max_duration),
        info=f"Duration in seconds (-1 for auto). Max: {max_duration}s / {max_duration // 60} min.",
        elem_classes=["has-info-container"],
    )
    try:
        batch_value = min(max(int(current_batch_size), 1), max_batch) if current_batch_size is not None else min(2, max_batch)
    except (TypeError, ValueError):
        batch_value = min(2, max_batch)
    batch_update = gr.update(
        value=batch_value,
        maximum=max_batch,
        info=f"Number of samples to generate (Max: {max_batch}).",
        elem_classes=["has-info-container"],
    )
    status_suffix = f"\nGPU Config: tier={gpu_config.tier}, max_duration={max_duration}s, max_batch={max_batch}"
    status_suffix += f", available_lm={gpu_config.available_lm_models}" if gpu_config.available_lm_models else ", LM not available for this GPU tier"
    return duration_update, batch_update, status_suffix, local_lm_initialized


def build_tier_updates(*, selected_tier: str, all_disk_models: list[str]) -> tuple[Any, ...]:
    """Build UI updates for a manual GPU tier override."""
    if not selected_tier or selected_tier not in GPU_TIER_CONFIGS:
        logger.warning(f"Invalid tier selection: {selected_tier}")
        return (gr.update(),) * 10
    new_config = get_gpu_config_for_tier(selected_tier)
    set_global_gpu_config(new_config)
    logger.info(f"Tier manually changed to {selected_tier} - updating UI defaults")
    available_backends = ["pt", "mlx"] if new_config.lm_backend_restriction == "pt_mlx_only" else ["vllm", "pt", "mlx"]
    recommended_backend = new_config.recommended_backend if new_config.recommended_backend in available_backends else available_backends[0]
    recommended_lm = new_config.recommended_lm_model
    default_lm_model = find_best_lm_model_on_disk(recommended_lm, all_disk_models)
    max_duration = new_config.max_duration_without_lm
    max_batch = new_config.max_batch_size_without_lm
    tier_label = GPU_TIER_LABELS.get(selected_tier, selected_tier)
    gpu_info_text = (
        f"**{get_gpu_device_name()}** - {new_config.gpu_memory_gb:.1f} GB VRAM "
        f"- {t('service.gpu_auto_tier')}: **{tier_label}**"
    )
    return (
        gr.update(
            value=new_config.offload_to_cpu_default,
            info=t("service.offload_cpu_info") + (" (recommended for this tier)" if new_config.offload_to_cpu_default else ""),
            elem_classes=["has-info-container"],
        ),
        gr.update(
            value=new_config.offload_dit_to_cpu_default,
            info=t("service.offload_dit_cpu_info") + (" (recommended for this tier)" if new_config.offload_dit_to_cpu_default else ""),
            elem_classes=["has-info-container"],
        ),
        gr.update(value=new_config.compile_model_default),
        gr.update(
            value=new_config.quantization_default,
            info=t("service.quantization_info") + (" (recommended for this tier)" if new_config.quantization_default else ""),
            elem_classes=["has-info-container"],
        ),
        gr.update(choices=available_backends, value=recommended_backend, elem_classes=["has-info-container"]),
        gr.update(
            choices=all_disk_models,
            value=default_lm_model,
            info=t("service.lm_model_path_info") + (f" (Recommended: {recommended_lm})" if recommended_lm else " (LM not available for this GPU tier)."),
            elem_classes=["has-info-container"],
        ),
        gr.update(value=new_config.init_lm_default, elem_classes=["has-info-container"]),
        gr.update(
            value=min(2, max_batch),
            maximum=max_batch,
            info=f"Number of samples to generate (Max: {max_batch}).",
            elem_classes=["has-info-container"],
        ),
        gr.update(
            maximum=float(max_duration),
            info=f"Duration in seconds (-1 for auto). Max: {max_duration}s / {max_duration // 60} min.",
            elem_classes=["has-info-container"],
        ),
        gr.update(value=gpu_info_text),
    )
