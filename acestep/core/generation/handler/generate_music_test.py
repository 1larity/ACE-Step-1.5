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

    def _vram_preflight_check(self, **_kwargs):
        """Disable hardware-dependent VRAM checks in unit tests."""
        return None

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

    def _empty_cache(self):
        """Ignore cache-clearing hooks in orchestration unit tests."""
        self.calls["_empty_cache"] = True

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

    def test_generate_music_accepts_repaint_controls_from_public_api(self):
        """It accepts repaint controls exposed by the UI/API contract."""
        host = _Host()
        out = host.generate_music(
            captions="cap",
            lyrics="lyr",
            repaint_latent_crossfade_frames=12,
            repaint_wav_crossfade_sec=0.2,
            repaint_mode="aggressive",
            repaint_strength=0.3,
        )

        self.assertEqual(out, host._final_payload)
        self.assertIn("_run_generate_music_service_with_progress", host.calls)
        self.assertEqual(host.calls["_run_generate_music_service_with_progress"]["repaint_crossfade_frames"], 12)
        self.assertEqual(host.calls["_run_generate_music_service_with_progress"]["repaint_injection_ratio"], 0.0)

    def test_generate_music_maps_balanced_repaint_strength_into_service_controls(self):
        """It resolves balanced repaint mode into model repaint parameters."""
        host = _Host()
        host.generate_music(
            captions="cap",
            lyrics="lyr",
            repaint_mode="balanced",
            repaint_strength=0.25,
        )

        run_kwargs = host.calls["_run_generate_music_service_with_progress"]
        self.assertEqual(run_kwargs["repaint_crossfade_frames"], 19)
        self.assertEqual(run_kwargs["repaint_injection_ratio"], 0.75)

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

    def test_generate_music_preserves_global_caption_on_recursive_fallback(self):
        """It forwards ``global_caption`` unchanged when a fallback re-enters generate_music."""

        class _RecordingHost(_Host):
            def __init__(self):
                super().__init__()
                self.generate_music_kwargs = []

            def generate_music(self, *args, **kwargs):
                self.generate_music_kwargs.append(dict(kwargs))
                if len(self.generate_music_kwargs) > 1:
                    return self._final_payload
                return super().generate_music(*args, **kwargs)

        host = _RecordingHost()
        host.quantization = "int8_weight_only"
        host.device = "cuda"

        with patch.object(host, "_should_preflight_cpu_stability_mode", return_value=True):
            out = host.generate_music(
                captions="cap",
                global_caption="global song framing",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual("global song framing", host.generate_music_kwargs[1]["global_caption"])

    def test_generate_music_cuda_canary_switches_to_cpu_before_full_run(self):
        """It uses canary probe and switches to CPU before full diffusion when canary is non-finite."""
        host = _Host()
        host.quantization = "int8_weight_only"
        host.device = "cuda"
        decode_calls = {"count": 0}

        def _decode_state_side_effect(**_kwargs):
            """Fail canary decode once, then allow decode after CPU switch."""
            decode_calls["count"] += 1
            if decode_calls["count"] == 1:
                raise RuntimeError("Generation produced too many NaN or Inf latents (100.00%).")
            return torch.ones(1, 4, 3), {"total_time_cost": 1.0}

        host._prepare_generate_music_decode_state = _decode_state_side_effect
        with patch.dict(os.environ, {"ACESTEP_ALLOW_RISKY_QUANTIZED_CUDA": "1"}, clear=False), patch.object(
            host,
            "_should_run_cuda_stability_canary",
            side_effect=lambda: host.device == "cuda",
        ):
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        self.assertEqual(host.cpu_switch_calls, 1)
        self.assertEqual(len(host.service_run_calls), 2)

    def test_generate_music_releases_canary_cuda_memory_before_full_run(self):
        """It clears CUDA cache after the canary probe before starting full diffusion."""
        host = _Host()
        host.quantization = "int8_weight_only"
        host.device = "cuda"

        with patch.dict(os.environ, {"ACESTEP_ALLOW_RISKY_QUANTIZED_CUDA": "1"}, clear=False), \
                patch.object(host, "_should_run_cuda_stability_canary", return_value=True), \
                patch.object(GENERATE_MUSIC_MODULE.torch.cuda, "is_available", return_value=True), \
                patch.object(GENERATE_MUSIC_MODULE.torch.cuda, "empty_cache") as empty_cache_mock:
            out = host.generate_music(
                captions="cap",
                lyrics="lyr",
                inference_steps=8,
                guidance_scale=7.0,
                use_adg=True,
            )

        self.assertEqual(out, host._final_payload)
        empty_cache_mock.assert_called()

    def test_generate_music_uses_profiled_phase_ranges_for_progress_mapping(self):
        """It forwards profiled phase ranges to service/decode progress mapping."""
        host = _Host()
        host._get_progress_phase_ranges = lambda: {
            "service_start": 0.20,
            "encoding_end": 0.32,
            "diffusion_end": 0.74,
        }
        progress_values = []

        def _progress(value, desc=None):
            """Capture progress checkpoints emitted by orchestration."""
            progress_values.append((value, desc))

        out = host.generate_music(captions="cap", lyrics="lyr", progress=_progress)
        self.assertEqual(out, host._final_payload)
        self.assertEqual(
            host.calls["_run_generate_music_service_with_progress"]["phase_ranges"]["service_start"],
            0.20,
        )
        self.assertEqual(
            host.calls["_decode_generate_music_pred_latents"]["decode_progress_start"],
            0.75,
        )
        self.assertTrue(any(abs(value - 0.20) < 1e-6 for value, _ in progress_values))


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

    @patch.object(_GM_MOD, "torch")
    def test_cuda_stability_canary_disabled_without_risky_opt_in(self, mock_torch):
        """It keeps canary disabled unless risky CUDA mode is explicitly enabled."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host()
        host.device = "cuda"
        host.quantization = "int8_weight_only"
        self.assertFalse(host._should_run_cuda_stability_canary())

    @patch.object(_GM_MOD, "torch")
    def test_cuda_stability_canary_enabled_with_risky_opt_in(self, mock_torch):
        """It enables canary on quantized CUDA when risky mode is explicitly enabled."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host()
        host.device = "cuda"
        host.quantization = "int8_weight_only"
        with patch.dict(os.environ, {"ACESTEP_ALLOW_RISKY_QUANTIZED_CUDA": "1"}, clear=False):
            self.assertTrue(host._should_run_cuda_stability_canary())


class ProgressProfileLoggingTests(unittest.TestCase):
    """Verify progress-profile timing math for stable, intuitive percentages."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    def test_progress_profile_uses_stage_fallbacks_and_partitioned_percentages(self):
        """It derives setup/service/decode from fallback keys and reports consistent percentages."""
        host = _Host()
        stage_timings = {
            "total_orchestration_sec": 5.0,
            "runtime_prep_sec": 0.2,
            "reference_audio_stage_sec": 0.3,
            "service_input_stage_sec": 0.4,
            "preflight_stage_sec": 0.1,
            "canary_service_generate_sec": 0.5,
        }
        time_costs = {
            "total_time_cost": 3.0,
            "vae_decode_time_cost": 1.0,
            "diffusion_time_cost": 1.5,
        }

        with patch.object(self._GM_MOD.logger, "info") as info_mock:
            host._log_generation_progress_profile(stage_timings=stage_timings, time_costs=time_costs)

        self.assertTrue(info_mock.called)
        call_args = info_mock.call_args[0]
        self.assertIn("diffusion", call_args[0])
        self.assertAlmostEqual(call_args[1], 5.0, places=6)
        self.assertAlmostEqual(call_args[2], 1.5, places=6)
        self.assertAlmostEqual(call_args[3], 30.0, places=6)
        self.assertAlmostEqual(call_args[4], 2.0, places=6)
        self.assertAlmostEqual(call_args[5], 40.0, places=6)
        self.assertAlmostEqual(call_args[6], 1.0, places=6)
        self.assertAlmostEqual(call_args[7], 20.0, places=6)
        self.assertAlmostEqual(call_args[8], 1.5, places=6)
        self.assertAlmostEqual(call_args[9], 75.0, places=6)
        self.assertAlmostEqual(call_args[10], 30.0, places=6)

    def test_progress_profile_skips_log_when_total_is_not_positive(self):
        """It does not emit profile logs when total orchestration time is missing or zero."""
        host = _Host()
        with patch.object(self._GM_MOD.logger, "info") as info_mock:
            host._log_generation_progress_profile(
                stage_timings={"total_orchestration_sec": 0.0},
                time_costs={"diffusion_time_cost": 1.0},
            )
        info_mock.assert_not_called()


class InternalFallbackControlsTests(unittest.TestCase):
    """Verify internal fallback controls and tunable fallback-step resolution."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    def test_generate_music_rejects_unknown_keyword_arguments(self):
        """It raises TypeError for unexpected kwargs instead of silently ignoring them."""
        host = _Host()
        with self.assertRaises(TypeError):
            host.generate_music(captions="cap", lyrics="lyr", unknown_option=True)

    def test_extract_internal_fallback_flags_defaults_and_overrides(self):
        """It reads internal fallback guard kwargs and rejects unexpected internal keys."""
        host = _Host()
        internal_kwargs = {"_allow_cpu_device_fallback": False}
        flags = host._extract_internal_fallback_flags(internal_kwargs)
        self.assertEqual(flags, (True, True, False))
        self.assertEqual(internal_kwargs, {})

        with self.assertRaises(TypeError):
            host._extract_internal_fallback_flags({"_bad_internal_key": True})

    def test_resolve_quantized_stability_fallback_steps_uses_default_tuning(self):
        """It preserves the legacy default fallback-step tuning when no env override is set."""
        host = _Host()
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(8), 6)
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(5), 5)
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(3), 4)

    def test_resolve_quantized_stability_fallback_steps_accepts_valid_env_override(self):
        """It honors valid fallback-step overrides and clamps values above inference steps."""
        host = _Host()
        with patch.dict(os.environ, {"ACESTEP_STABILITY_FALLBACK_STEPS": "5"}, clear=False):
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(8), 5)
        with patch.dict(os.environ, {"ACESTEP_STABILITY_FALLBACK_STEPS": "10"}, clear=False):
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(8), 8)

    def test_resolve_quantized_stability_fallback_steps_falls_back_on_invalid_override(self):
        """It ignores invalid fallback-step overrides and keeps legacy default behavior."""
        host = _Host()
        with patch.dict(os.environ, {"ACESTEP_STABILITY_FALLBACK_STEPS": "abc"}, clear=False):
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(8), 6)
        with patch.dict(os.environ, {"ACESTEP_STABILITY_FALLBACK_STEPS": "0"}, clear=False):
            self.assertEqual(host._resolve_quantized_stability_fallback_steps(8), 6)


class VramPreflightCheckTests(unittest.TestCase):
    """Verify ``_vram_preflight_check`` respects CPU offload mode."""

    _GM_MOD = GENERATE_MUSIC_MODULE

    @patch.object(_GM_MOD, "torch")
    def test_preflight_skips_when_offload_to_cpu_enabled(self, mock_torch):
        """It returns None (pass) when offload_to_cpu is True, regardless of free VRAM."""
        mock_torch.cuda.is_available.return_value = True
        host = _Host(offload_to_cpu=True)
        result = GenerateMusicMixin._vram_preflight_check(
            host,
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
        result = GenerateMusicMixin._vram_preflight_check(
            host,
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
        result = GenerateMusicMixin._vram_preflight_check(
            host,
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
        result = GenerateMusicMixin._vram_preflight_check(
            host,
            actual_batch_size=2,
            audio_duration=246.0,
            guidance_scale=7.0,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
