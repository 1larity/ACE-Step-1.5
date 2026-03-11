"""Tests for extracted ``generate_music`` orchestration behavior.

The module loads ``acestep.core.generation.handler.generate_music`` directly
from file to avoid package import side effects and validates orchestration
ordering, readiness short-circuiting, and failure payload handling.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import torch


def _load_generate_music_module():
    """Load ``generate_music.py`` from disk for isolated mixin tests.

    Returns:
        types.ModuleType: Loaded module object for
        ``acestep.core.generation.handler.generate_music``.

    Raises:
        FileNotFoundError: If the target module file is missing.
        ImportError: If module loading fails.
    """
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_paths = {
        "acestep": repo_root / "acestep",
        "acestep.core": repo_root / "acestep" / "core",
        "acestep.core.generation": repo_root / "acestep" / "core" / "generation",
        "acestep.core.generation.handler": repo_root / "acestep" / "core" / "generation" / "handler",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        sys.modules[package_name] = package_module
    module_path = Path(__file__).with_name("generate_music.py")
    spec = importlib.util.spec_from_file_location(
        "acestep.core.generation.handler.generate_music",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GENERATE_MUSIC_MODULE = _load_generate_music_module()
GenerateMusicMixin = GENERATE_MUSIC_MODULE.GenerateMusicMixin


class _Host(GenerateMusicMixin):
    """Minimal host implementing ``generate_music`` helper dependencies.

    The host captures helper calls in ``self.calls`` and returns deterministic
    payloads so tests can assert orchestration sequencing and return behavior.
    """

    def __init__(self, offload_to_cpu: bool = False):
        """Initialize deterministic state and stub payloads for orchestration tests."""
        self.model = object()
        self.vae = object()
        self.text_tokenizer = object()
        self.text_encoder = object()
        self.offload_to_cpu = offload_to_cpu
        self.calls: Dict[str, Any] = {}
        self.service_run_calls = []
        self.device = "cpu"
        self.quantization = None
        self.switch_calls = 0
        self.quantized_switch_calls = 0
        self.cpu_switch_calls = 0
        self._final_payload = {"audios": [{"tensor": torch.zeros(1, 4), "sample_rate": 48000}], "success": True}
        self._readiness_error = {
            "audios": [],
            "status_message": "not ready",
            "extra_outputs": {},
            "success": False,
            "error": "Model not fully initialized",
        }

    def _resolve_generate_music_progress(self, progress):
        """Return provided callback or deterministic no-op callback."""
        self.calls["_resolve_generate_music_progress"] = bool(progress)
        if progress is not None:
            return progress

        def _noop(*_args, **_kwargs):
            """Ignore progress updates in tests."""
            return None

        return _noop

    def _validate_generate_music_readiness(self):
        """Return deterministic readiness error payload."""
        self.calls["_validate_generate_music_readiness"] = True
        return self._readiness_error

    def _resolve_generate_music_task(self, **kwargs):
        """Capture task resolution args and return deterministic task/instruction."""
        self.calls["_resolve_generate_music_task"] = kwargs
        return kwargs["task_type"], kwargs["instruction"]

    def _prepare_generate_music_runtime(self, **kwargs):
        """Capture runtime args and return deterministic runtime state."""
        self.calls["_prepare_generate_music_runtime"] = kwargs
        return {
            "actual_batch_size": 1,
            "actual_seed_list": [77],
            "seed_value_for_ui": 77,
            "audio_duration": kwargs["audio_duration"],
            "repainting_end": kwargs["repainting_end"],
        }

    def _prepare_reference_and_source_audio(self, **kwargs):
        """Capture audio-prepare args and return deterministic prepared state."""
        self.calls["_prepare_reference_and_source_audio"] = kwargs
        return [[torch.zeros(2, 10)]], None, None

    def _prepare_generate_music_service_inputs(self, **kwargs):
        """Capture service-input args and return deterministic payload."""
        self.calls["_prepare_generate_music_service_inputs"] = kwargs
        return {"should_return_intermediate": True}

    def _run_generate_music_service_with_progress(self, **kwargs):
        """Capture service execution args and return deterministic model outputs."""
        self.calls["_run_generate_music_service_with_progress"] = kwargs
        self.service_run_calls.append(kwargs)
        return {
            "outputs": {
                "target_latents": torch.ones(1, 4, 3),
                "time_costs": {"total_time_cost": 1.0, "diffusion_per_step_time_cost": 0.1},
            },
            "infer_steps_for_progress": 8,
        }

    def _prepare_generate_music_decode_state(self, **kwargs):
        """Capture decode-state args and return deterministic latents/costs."""
        self.calls["_prepare_generate_music_decode_state"] = kwargs
        return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

    def _decode_generate_music_pred_latents(self, **kwargs):
        """Capture decode args and return deterministic decode outputs."""
        self.calls["_decode_generate_music_pred_latents"] = kwargs
        return torch.ones(1, 2, 8), torch.ones(1, 4, 3), {"total_time_cost": 2.0}

    def _build_generate_music_success_payload(self, **kwargs):
        """Capture payload-builder args and return deterministic success payload."""
        self.calls["_build_generate_music_success_payload"] = kwargs
        return self._final_payload

    def switch_to_training_preset(self):
        """Simulate successful preset switch in fallback tests."""
        self.switch_calls += 1
        self.quantization = None
        return "switched", True

    def switch_to_stable_quantized_preset(self):
        """Simulate successful quantized-mode switch used by stability fallback."""
        self.quantized_switch_calls += 1
        self.quantization = "int8_weight_only"
        return "switched to int8", True

    def switch_to_cpu_stability_preset(self):
        """Simulate successful CPU stability fallback switch."""
        self.cpu_switch_calls += 1
        self.quantization = None
        self.device = "cpu"
        return "switched to cpu", True


class GenerateMusicMixinTests(unittest.TestCase):
    """Verify top-level ``generate_music`` orchestration behavior."""

    def test_generate_music_returns_success_payload_from_builder(self):
        """It executes helper stages and returns the payload builder result."""
        host = _Host()
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=6.5,
            use_random_seed=False,
            seed=77,
            task_type="text2music",
        )
        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.calls["_prepare_generate_music_runtime"]["seed"], 77)
        self.assertEqual(host.calls["_run_generate_music_service_with_progress"]["guidance_scale"], 6.5)
        self.assertEqual(host.calls["_prepare_generate_music_decode_state"]["infer_steps_for_progress"], 8)

    def test_generate_music_returns_readiness_error_when_components_missing(self):
        """It short-circuits with readiness payload when required models are missing."""
        host = _Host()
        host.model = None
        out = host.generate_music(captions="cap", lyrics="lyr")
        self.assertEqual(out, host._readiness_error)
        self.assertTrue(host.calls["_validate_generate_music_readiness"])
        self.assertNotIn("_prepare_generate_music_runtime", host.calls)

    def test_generate_music_returns_error_payload_on_exception(self):
        """It catches orchestration errors and returns standardized failure payload."""
        host = _Host()

        def _raise_error(**_kwargs):
            """Raise deterministic runtime failure for exception-path validation."""
            raise RuntimeError("boom")

        host._prepare_reference_and_source_audio = _raise_error
        out = host.generate_music(captions="cap", lyrics="lyr")
        self.assertFalse(out["success"])
        self.assertEqual(out["error"], "boom")
        self.assertIn("Error: boom", out["status_message"])

    def test_generate_music_retries_once_with_safer_diffusion_after_non_finite_latents(self):
        """It retries generation once with safer settings after non-finite-latent runtime error."""
        host = _Host()
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**kwargs):
            """Raise once, then return valid decode inputs."""
            decode_calls["count"] += 1
            if decode_calls["count"] == 1:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=7.0,
            use_adg=True,
        )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(len(host.service_run_calls), 2)
        self.assertEqual(host.service_run_calls[0]["guidance_scale"], 7.0)
        self.assertEqual(host.service_run_calls[1]["guidance_scale"], 1.0)
        self.assertFalse(host.service_run_calls[1]["use_adg"])
        self.assertIsNone(host.service_run_calls[1]["progress"])

    def test_generate_music_runs_final_stability_retry_for_quantized_non_finite_failure(self):
        """It performs a third retry with reduced steps when quantized retries still fail."""
        host = _Host()
        host.quantization = "w8a8_dynamic"
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**kwargs):
            """Raise twice, then succeed so fallback path is asserted."""
            decode_calls["count"] += 1
            if decode_calls["count"] <= 2:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            inference_steps=8,
            guidance_scale=7.0,
            use_adg=True,
        )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(len(host.service_run_calls), 3)
        self.assertEqual(host.service_run_calls[2]["inference_steps"], 6)
        self.assertEqual(host.service_run_calls[2]["infer_method"], "sde")
        self.assertIsNone(host.service_run_calls[2]["actual_seed_list"])
        self.assertIsNone(host.service_run_calls[2]["timesteps"])

    def test_generate_music_switches_to_non_quantized_preset_after_all_retries_fail(self):
        """It auto-switches preset and re-enters generation once when quantized path stays unstable."""
        host = _Host()
        host.quantization = "w8a8_dynamic"
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**kwargs):
            """Fail first three decode-state validations, then succeed after preset switch."""
            decode_calls["count"] += 1
            if decode_calls["count"] <= 3:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        with patch.object(host, "_can_attempt_non_quantized_fallback", return_value=True):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.switch_calls, 1)
        self.assertEqual(len(host.service_run_calls), 4)

    def test_generate_music_skips_non_quantized_fallback_on_low_vram(self):
        """It avoids non-quantized fallback on low VRAM and returns failure payload."""
        host = _Host()
        host.quantization = "w8a8_dynamic"

        def _decode_state_side_effect(**_kwargs):
            """Always raise non-finite-latents error to force fallback decision."""
            raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        with patch.object(host, "_can_attempt_non_quantized_fallback", return_value=False), patch.object(
            host, "_should_preflight_cpu_stability_mode", return_value=False
        ):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
                _allow_quantized_mode_fallback=False,
            )

        self.assertFalse(out["success"])
        self.assertIn("too many NaN or Inf latents", out["error"])
        self.assertEqual(host.switch_calls, 0)
        self.assertEqual(host.quantized_switch_calls, 0)
        self.assertEqual(len(host.service_run_calls), 3)

    def test_generate_music_switches_to_stable_quantized_preset_on_low_vram(self):
        """It switches w8a8 runtime to int8 weight-only before failing low-VRAM fallback."""
        host = _Host()
        host.quantization = "w8a8_dynamic"
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**_kwargs):
            """Fail three times, then succeed after quantization mode switch."""
            decode_calls["count"] += 1
            if decode_calls["count"] <= 3:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        with patch.object(host, "_can_attempt_non_quantized_fallback", return_value=False), patch.object(
            host, "_should_preflight_cpu_stability_mode", return_value=False
        ):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.quantized_switch_calls, 1)
        self.assertEqual(host.switch_calls, 0)
        self.assertEqual(len(host.service_run_calls), 4)

    def test_generate_music_switches_to_cpu_stability_preset_when_int8_remains_unstable(self):
        """It switches to CPU stability mode when low-VRAM int8 retries remain non-finite."""
        host = _Host()
        host.quantization = "int8_weight_only"
        host.device = "cuda"
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**_kwargs):
            """Fail three times, then succeed after CPU stability fallback switch."""
            decode_calls["count"] += 1
            if decode_calls["count"] <= 3:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        with patch.object(host, "_can_attempt_non_quantized_fallback", return_value=False), patch.object(
            host, "_should_preflight_cpu_stability_mode", return_value=False
        ):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
                _allow_quantized_mode_fallback=False,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.cpu_switch_calls, 1)
        self.assertEqual(host.quantized_switch_calls, 0)
        self.assertEqual(host.switch_calls, 0)
        self.assertEqual(len(host.service_run_calls), 4)

    def test_generate_music_preflights_to_cpu_stability_before_cuda_attempt(self):
        """It switches to CPU stability mode before any diffusion on known-risky profile."""
        host = _Host()
        host.quantization = "int8_weight_only"
        host.device = "cuda"

        with patch.object(host, "_should_preflight_cpu_stability_mode", return_value=True):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.cpu_switch_calls, 1)
        self.assertEqual(len(host.service_run_calls), 1)


class NonQuantizedFallbackGateTests(unittest.TestCase):
    """Verify non-quantized fallback gating behavior on CUDA memory detection."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    @patch.object(_GM_MOD, "get_gpu_memory_gb", return_value=3.94)
    @patch.object(_GM_MOD, "torch")
    def test_fallback_gate_blocks_when_cuda_vram_is_below_threshold(
        self, mock_torch, _mock_gpu_memory
    ):
        """It returns False when detected CUDA memory is below minimum threshold."""
        mock_torch.cuda.is_available.return_value = True
        self.assertFalse(_Host()._can_attempt_non_quantized_fallback())

    @patch.object(_GM_MOD, "get_gpu_memory_gb", side_effect=RuntimeError("probe failed"))
    @patch.object(_GM_MOD, "torch")
    def test_fallback_gate_fails_closed_when_memory_detection_errors(
        self, mock_torch, _mock_gpu_memory
    ):
        """It returns False when memory probing fails to avoid risky fallback attempts."""
        mock_torch.cuda.is_available.return_value = True
        self.assertFalse(_Host()._can_attempt_non_quantized_fallback())

    @patch.object(_GM_MOD, "get_gpu_memory_gb", return_value=3.94)
    @patch.object(_GM_MOD, "torch")
    def test_cpu_preflight_can_be_overridden_by_env_opt_out(self, mock_torch, _mock_gpu_memory):
        """It disables CPU preflight when user explicitly opts into risky CUDA attempts."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        host = _Host()
        host.quantization = "int8_weight_only"
        host.device = "cuda"

        with patch.dict(os.environ, {"ACESTEP_ALLOW_RISKY_QUANTIZED_CUDA": "1"}, clear=False):
            self.assertFalse(host._should_preflight_cpu_stability_mode())


class VramPreflightCheckTests(unittest.TestCase):
    """Verify ``_vram_preflight_check`` respects CPU offload mode."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    @patch.object(_GM_MOD, "torch")
    def test_preflight_skips_when_offload_to_cpu_enabled(self, mock_torch):
        """It returns None (pass) when offload_to_cpu is True, regardless of free VRAM."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=True)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=3.4)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_blocks_when_offload_disabled_and_vram_low(
        self, mock_torch, _mock_free_vram
    ):
        """It returns error payload when offload is off and free VRAM is insufficient."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertIn("Insufficient free VRAM", result["error"])

    @patch.object(_GM_MOD, "get_effective_free_vram_gb", return_value=24.0)
    @patch.object(_GM_MOD, "torch")
    def test_preflight_passes_when_offload_disabled_and_vram_sufficient(
        self, mock_torch, _mock_free_vram
    ):
        """It returns None when offload is off but free VRAM exceeds estimate."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)

    @patch.object(_GM_MOD, "torch")
    def test_preflight_passes_on_non_cuda_device(self, mock_torch):
        """It returns None when CUDA is not available (CPU/MPS/XPU)."""
        mock_torch.cuda.is_available.return_value = False
        host = _Host(offload_to_cpu=False)
        result = host._vram_preflight_check(
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
