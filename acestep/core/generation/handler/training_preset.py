"""Training-preset switching helpers for handler decomposition."""

from typing import Tuple


class TrainingPresetMixin:
    """Helpers for switching runtime initialization to training-safe settings."""

    def _switch_quantization_preset(self, target_quantization: str | None, preset_label: str) -> Tuple[str, bool]:
        """Reinitialize service with a target quantization preset using cached init params."""
        if not self.last_init_params:
            return "Cannot switch preset automatically: no previous init parameters found.", False

        params = dict(self.last_init_params)
        status, ok = self.initialize_service(
            project_root=params["project_root"],
            config_path=params["config_path"],
            device=params["device"],
            use_flash_attention=params["use_flash_attention"],
            compile_model=params["compile_model"],
            offload_to_cpu=params["offload_to_cpu"],
            offload_dit_to_cpu=params["offload_dit_to_cpu"],
            quantization=target_quantization,
            prefer_source=params.get("prefer_source"),
            use_mlx_dit=params.get("use_mlx_dit", True),
        )
        if ok:
            return f"Switched to {preset_label}.\n{status}", True
        return f"Failed to switch to {preset_label}.\n{status}", False

    def switch_to_training_preset(self) -> Tuple[str, bool]:
        """Reinitialize with quantization disabled using the last successful init parameters.

        Returns:
            Tuple[str, bool]:
                - A human-readable status message for UI/API consumers.
                - ``True`` when the preset is already safe or reinitialization succeeds,
                  otherwise ``False``.
        """
        if self.quantization is None:
            return "Already in training-safe preset (quantization disabled).", True

        return self._switch_quantization_preset(None, "training preset (quantization disabled)")

    def switch_to_stable_quantized_preset(self) -> Tuple[str, bool]:
        """Reinitialize with int8 weight-only quantization for better numerical stability."""
        if self.quantization == "int8_weight_only":
            return "Already in stable quantized preset (int8_weight_only).", True
        return self._switch_quantization_preset(
            "int8_weight_only",
            "stable quantized preset (int8_weight_only)",
        )

    def switch_to_cpu_stability_preset(self) -> Tuple[str, bool]:
        """Reinitialize on CPU without quantization as a last-resort stability fallback."""
        if not self.last_init_params:
            return "Cannot switch preset automatically: no previous init parameters found.", False

        params = dict(self.last_init_params)
        status, ok = self.initialize_service(
            project_root=params["project_root"],
            config_path=params["config_path"],
            device="cpu",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
            prefer_source=params.get("prefer_source"),
            use_mlx_dit=False,
        )
        if ok:
            return f"Switched to CPU stability preset (quantization disabled).\n{status}", True
        return f"Failed to switch to CPU stability preset.\n{status}", False
