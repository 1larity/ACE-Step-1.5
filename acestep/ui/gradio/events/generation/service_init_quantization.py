"""Quantization selection helpers for generation service initialization."""

from loguru import logger


def _select_quantization_value(
    *,
    quantization_enabled: bool,
    device: str,
) -> str | None:
    """Return the DiT quantization mode selected for the current UI state."""
    quant_value = "int8_weight_only" if quantization_enabled else None
    if not quantization_enabled or device not in {"auto", "cuda"}:
        return quant_value

    try:
        import torch
    except ImportError:
        return quant_value

    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                logger.info(
                    "Pre-Ampere CUDA detected: using w8a8_dynamic quantization for stability"
                )
                return "w8a8_dynamic"
    except Exception:
        return quant_value
    return quant_value
