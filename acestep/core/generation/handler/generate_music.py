"""Top-level ``generate_music`` orchestration mixin.

This module provides the public ``generate_music`` entry point extracted from
``AceStepHandler`` so orchestration stays separate from lower-level helpers.
"""

import gc
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION
from acestep.gpu_config import (
    DIT_INFERENCE_VRAM_PER_BATCH,
    VRAM_SAFETY_MARGIN_GB,
    get_effective_free_vram_gb,
    get_gpu_memory_gb,
)

_DEFAULT_PROGRESS_PHASE_RANGES = {
    "service_start": 0.30,
    "encoding_end": 0.45,
    "diffusion_end": 0.79,
}

_MAX_REPAINT_CROSSFADE_FRAMES = 25
_MAX_REPAINT_WAV_CROSSFADE_SEC = 0.05


def _resolve_repaint_config(
    mode: str = "balanced",
    strength: float = 0.5,
) -> Tuple[float, int, float]:
    """Resolve repaint preservation controls into model/runtime parameters.

    Args:
        mode: User-facing repaint mode.
        strength: Balanced-mode preservation slider.

    Returns:
        Tuple of ``(repaint_injection_ratio, repaint_crossfade_frames,
        repaint_wav_crossfade_sec)``.
    """
    normalized_mode = (mode or "balanced").strip().lower()
    clamped_strength = max(0.0, min(1.0, float(strength)))

    if normalized_mode == "aggressive":
        preserve_ratio = 0.0
    elif normalized_mode == "conservative":
        preserve_ratio = 1.0
    else:
        preserve_ratio = 1.0 - clamped_strength

    repaint_crossfade_frames = int(round(_MAX_REPAINT_CROSSFADE_FRAMES * preserve_ratio))
    repaint_wav_crossfade_sec = _MAX_REPAINT_WAV_CROSSFADE_SEC * preserve_ratio
    return preserve_ratio, repaint_crossfade_frames, repaint_wav_crossfade_sec


class GenerateMusicMixin:
    """Coordinate request prep, service execution, decode, and payload assembly.

    The host class is expected to implement helper methods invoked by this
    orchestration flow.
    """

    def _vram_preflight_check(
        self,
        actual_batch_size: int,
        audio_duration: Optional[float],
        guidance_scale: float,
    ) -> Optional[Dict[str, Any]]:
        """Check free VRAM headroom before attempting service_generate.

        Model weights are already resident in GPU memory at this point.  We
        only need to verify there is enough room for the diffusion-pass
        activations (intermediate attention maps, FFN buffers, noise tensors)
        plus a project-standard safety margin.

        Args:
            actual_batch_size: Number of samples being generated.
            audio_duration: Requested audio length in seconds, or None for default.
            guidance_scale: CFG guidance value; values > 1.0 indicate CFG is active
                and the DiT runs two forward passes per step (doubling activation memory).

        Returns:
            An error payload dict when VRAM is insufficient, or None when the
            check passes or no CUDA device is present (CPU/MPS/XPU fall through).
        """
        if not torch.cuda.is_available():
            return None

        if getattr(self, "offload_to_cpu", False):
            logger.debug(
                "[generate_music] VRAM pre-flight: skipping check "
                "(offload_to_cpu=True, models loaded one-at-a-time)"
            )
            return None

        duration_s = audio_duration or 60.0
        # CFG doubles forward-pass memory: two DiT evaluations per step.
        dit_key = "base" if guidance_scale > 1.0 else "turbo"
        per_batch_gb = DIT_INFERENCE_VRAM_PER_BATCH.get(dit_key, 0.6)
        # Longer audio = more latent frames (5 Hz rate) = more memory.
        duration_factor = max(1.0, duration_s / 60.0)
        needed_gb = per_batch_gb * actual_batch_size * duration_factor + VRAM_SAFETY_MARGIN_GB

        free_gb = get_effective_free_vram_gb()
        logger.info(
            "[generate_music] VRAM pre-flight: {:.2f} GB free, ~{:.2f} GB needed "
            "(batch={}, duration={:.0f}s, mode={}).",
            free_gb, needed_gb, actual_batch_size, duration_s, dit_key,
        )

        if free_gb >= needed_gb:
            return None

        msg = (
            f"Insufficient free VRAM: need ~{needed_gb:.1f} GB, "
            f"only {free_gb:.1f} GB available. "
            f"Reduce batch size (currently {actual_batch_size}) "
            f"or audio duration (currently {duration_s:.0f}s)."
        )
        logger.warning("[generate_music] VRAM pre-flight failed: {}", msg)
        return {
            "audios": [],
            "status_message": f"Error: {msg}",
            "extra_outputs": {},
            "success": False,
            "error": msg,
        }

    @staticmethod
    def _should_retry_after_non_finite_latents(
        exc: RuntimeError,
        guidance_scale: float,
        use_adg: bool,
    ) -> bool:
        """Return ``True`` when a safer one-time diffusion retry is warranted."""
        msg = str(exc)
        non_finite_error = "too many NaN or Inf latents" in msg
        return non_finite_error and (guidance_scale > 1.0 or use_adg)

    @staticmethod
    def _is_non_finite_latents_error(exc: RuntimeError) -> bool:
        """Return ``True`` when error matches non-finite latent validation failure."""
        return "too many NaN or Inf latents" in str(exc)

    @staticmethod
    def _can_attempt_non_quantized_fallback() -> bool:
        """Return whether non-quantized fallback is viable on current hardware."""
        if not torch.cuda.is_available():
            return False
        try:
            total_mem_gb = float(get_gpu_memory_gb())
        except Exception:
            total_mem_gb = 0.0
        return total_mem_gb >= 6.0

    @staticmethod
    def _is_generation_progress_profiling_enabled() -> bool:
        """Return whether optional generation progress profiling is enabled."""
        return os.environ.get("ACESTEP_PROFILE_GENERATION_PROGRESS", "").lower() in ("1", "true", "yes")

    @staticmethod
    def _allow_risky_quantized_cuda_attempts() -> bool:
        """Return whether the user explicitly allows risky quantized CUDA attempts."""
        return os.environ.get("ACESTEP_ALLOW_RISKY_QUANTIZED_CUDA", "").lower() in ("1", "true", "yes")

    def _should_preflight_cpu_stability_mode(self) -> bool:
        """Return whether generation should skip risky CUDA retries and start in CPU stability mode.

        This guard targets legacy low-VRAM pre-Ampere CUDA setups where quantized DiT
        runs are prone to producing fully non-finite latent outputs.
        """
        if self._allow_risky_quantized_cuda_attempts():
            return False
        if getattr(self, "device", None) != "cuda":
            return False
        quantization = getattr(self, "quantization", None)
        if quantization not in {"int8_weight_only", "w8a8_dynamic", "fp8_weight_only"}:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            major, _ = torch.cuda.get_device_capability(0)
            total_mem_gb = float(get_gpu_memory_gb())
        except Exception:
            return False
        return major < 7 and total_mem_gb < 6.0

    def _should_run_cuda_stability_canary(self) -> bool:
        """Return whether a short CUDA canary run should probe latent stability.

        The canary is only enabled when users explicitly opt into risky quantized
        CUDA attempts; otherwise we either use normal CUDA flow or static CPU
        preflight fallback for known-unstable profiles.
        """
        if not self._allow_risky_quantized_cuda_attempts():
            return False
        if os.environ.get("ACESTEP_DISABLE_CUDA_STABILITY_CANARY", "").lower() in ("1", "true", "yes"):
            return False
        if getattr(self, "device", None) != "cuda":
            return False
        if getattr(self, "quantization", None) is None:
            return False
        return torch.cuda.is_available()

    @staticmethod
    def _extract_internal_fallback_flags(internal_kwargs: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Extract internal recursion-guard flags from kwargs.

        These guards are intentionally treated as internal controls and should
        not appear as explicit public API parameters.

        Args:
            internal_kwargs: Extra kwargs passed into ``generate_music``.

        Returns:
            Tuple containing ``(allow_dequant, allow_quantized_mode, allow_cpu_fallback)``.

        Raises:
            TypeError: If unexpected kwargs are provided.
        """
        allow_dequant = bool(internal_kwargs.pop("_allow_dequant_fallback", True))
        allow_quantized_mode = bool(internal_kwargs.pop("_allow_quantized_mode_fallback", True))
        allow_cpu_fallback = bool(internal_kwargs.pop("_allow_cpu_device_fallback", True))
        if internal_kwargs:
            unexpected = ", ".join(sorted(internal_kwargs))
            raise TypeError(f"generate_music() got unexpected keyword argument(s): {unexpected}")
        return allow_dequant, allow_quantized_mode, allow_cpu_fallback

    @staticmethod
    def _resolve_quantized_stability_fallback_steps(inference_steps: int) -> int:
        """Resolve fallback diffusion steps for quantized stability retries.

        Default behavior preserves legacy tuning (clamped into ``[4, 6]`` and
        bounded by requested ``inference_steps``). An explicit env override can
        be used for experimentation on edge hardware.
        """
        default_steps = max(4, min(inference_steps, 6))
        raw_override = os.environ.get("ACESTEP_STABILITY_FALLBACK_STEPS", "").strip()
        if not raw_override:
            return default_steps
        try:
            requested = int(raw_override)
        except ValueError:
            logger.warning(
                "[generate_music] Invalid ACESTEP_STABILITY_FALLBACK_STEPS='{}'; using default {}.",
                raw_override,
                default_steps,
            )
            return default_steps

        if requested < 1:
            logger.warning(
                "[generate_music] ACESTEP_STABILITY_FALLBACK_STEPS must be >=1 (got {}); "
                "using default {}.",
                requested,
                default_steps,
            )
            return default_steps
        resolved = min(requested, max(1, inference_steps))
        if resolved != requested:
            logger.info(
                "[generate_music] Clamped ACESTEP_STABILITY_FALLBACK_STEPS from {} to {} "
                "(requested inference_steps={}).",
                requested,
                resolved,
                inference_steps,
            )
        return resolved

    def _log_generation_progress_profile(
        self,
        *,
        stage_timings: Dict[str, float],
        time_costs: Optional[Dict[str, Any]],
    ) -> None:
        """Log optional stage timings used to calibrate progress tracking."""
        def _to_float(value: Any) -> float:
            """Convert numeric-like values to float while failing closed to 0.0."""
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        total = _to_float(stage_timings.get("total_orchestration_sec", 0.0))
        if total <= 0:
            return
        setup = _to_float(stage_timings.get("setup_before_service_sec", 0.0))
        if setup <= 0:
            setup = (
                _to_float(stage_timings.get("runtime_prep_sec", 0.0))
                + _to_float(stage_timings.get("reference_audio_stage_sec", 0.0))
                + _to_float(stage_timings.get("service_input_stage_sec", 0.0))
                + _to_float(stage_timings.get("preflight_stage_sec", 0.0))
                + _to_float(stage_timings.get("canary_service_generate_sec", 0.0))
            )
        service = _to_float(stage_timings.get("service_generate_sec", 0.0))
        decode = _to_float(stage_timings.get("decode_total_sec", 0.0))
        diffusion = 0.0
        if isinstance(time_costs, dict):
            diffusion = _to_float(time_costs.get("diffusion_time_cost", 0.0))
            if service <= 0:
                total_cost = _to_float(time_costs.get("total_time_cost", 0.0))
                decode_cost = _to_float(time_costs.get("vae_decode_time_cost", 0.0))
                service = max(0.0, total_cost - decode_cost)
            if decode <= 0:
                decode = _to_float(time_costs.get("vae_decode_time_cost", 0.0))

        partition_total = setup + service + decode
        denom = max(total, partition_total, 1e-9)
        service_denom = max(service, 1e-9)
        setup_pct = 100.0 * setup / denom
        service_pct = 100.0 * service / denom
        decode_pct = 100.0 * decode / denom
        diffusion_service_pct = min(100.0, 100.0 * diffusion / service_denom)
        diffusion_total_pct = min(100.0, 100.0 * diffusion / denom)

        logger.info(
            "[progress_profile] total={:.2f}s | setup={:.2f}s ({:.1f}%) | "
            "service={:.2f}s ({:.1f}%) | decode={:.2f}s ({:.1f}%) | "
            "diffusion={:.2f}s ({:.1f}% of service, {:.1f}% of total)",
            total,
            setup,
            setup_pct,
            service,
            service_pct,
            decode,
            decode_pct,
            diffusion,
            diffusion_service_pct,
            diffusion_total_pct,
        )

    def _resolve_generation_phase_ranges(self) -> Dict[str, float]:
        """Resolve progress phase ranges from persisted machine profile or defaults."""
        resolver = getattr(self, "_get_progress_phase_ranges", None)
        if callable(resolver):
            resolved = resolver()
            if isinstance(resolved, dict):
                return resolved
        return dict(_DEFAULT_PROGRESS_PHASE_RANGES)

    @staticmethod
    def _scale_setup_progress_checkpoint(service_start: float, ratio: float) -> float:
        """Map setup checkpoint ``ratio`` into ``[0.05, service_start]`` progress range."""
        ratio = max(0.0, min(1.0, float(ratio)))
        return 0.05 + ((float(service_start) - 0.05) * ratio)

    @staticmethod
    def _resolve_decode_progress_start(phase_ranges: Dict[str, float]) -> float:
        """Return decode stage start progress just after diffusion range end."""
        diffusion_end = float(phase_ranges.get("diffusion_end", 0.79))
        return max(diffusion_end, min(0.97, diffusion_end + 0.01))

    def generate_music(
        self,
        captions: str,
        global_caption: str = "",
        lyrics: str = "",
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        vocal_language: str = "en",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        use_random_seed: bool = True,
        seed: Optional[Union[str, float, int]] = -1,
        reference_audio=None,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        src_audio=None,
        audio_code_string: Union[str, List[str]] = "",
        repainting_start: float = 0.0,
        repainting_end: Optional[float] = None,
        instruction: str = DEFAULT_DIT_INSTRUCTION,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        task_type: str = "text2music",
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        infer_method: str = "ode",
        use_tiled_decode: bool = True,
        timesteps: Optional[List[float]] = None,
        latent_shift: float = 0.0,
        latent_rescale: float = 1.0,
        chunk_mask_mode: str = "auto",
        repaint_latent_crossfade_frames: int = 10,
        repaint_wav_crossfade_sec: float = 0.0,
        repaint_mode: str = "balanced",
        repaint_strength: float = 0.5,
        progress=None,
        **internal_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate audio from text/reference inputs and return response payload.

        Args:
            captions: Text prompt describing requested music.
            lyrics: Lyric text used for conditioning.
            reference_audio: Optional reference-audio payload.
            src_audio: Optional source audio for repaint/cover.
            inference_steps: Diffusion step count.
            guidance_scale: CFG guidance value.
            seed: Optional explicit seed from caller/UI.
            infer_method: Diffusion method name.
            timesteps: Optional custom timestep schedule.
            use_tiled_decode: Whether tiled VAE decode is used.
            latent_shift: Additive latent post-processing value.
            latent_rescale: Multiplicative latent post-processing value.
            progress: Optional callback taking ``(ratio, desc=...)``.

        Returns:
            Dict[str, Any]: Standard payload with generated audio tensors, status,
            intermediate outputs, success flag, and optional error text.

        Raises:
            No exceptions are re-raised. Runtime failures are converted into the
            returned error payload.
        """
        progress = self._resolve_generate_music_progress(progress)
        (
            allow_dequant_fallback,
            allow_quantized_mode_fallback,
            allow_cpu_device_fallback,
        ) = self._extract_internal_fallback_flags(internal_kwargs)
        profile_progress = self._is_generation_progress_profiling_enabled()
        stage_timings: Dict[str, float] = {}
        last_time_costs: Optional[Dict[str, Any]] = None
        generation_start = time.perf_counter()
        phase_ranges = self._resolve_generation_phase_ranges()
        service_start_progress = float(phase_ranges.get("service_start", 0.30))
        diffusion_end_progress = float(phase_ranges.get("diffusion_end", 0.79))
        decode_start_progress = self._resolve_decode_progress_start(phase_ranges)
        if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
            readiness_error = self._validate_generate_music_readiness()
            return readiness_error

        task_type, instruction = self._resolve_generate_music_task(
            task_type=task_type,
            audio_code_string=audio_code_string,
            instruction=instruction,
        )
        repaint_injection_ratio, resolved_repaint_crossfade_frames, _ = _resolve_repaint_config(
            repaint_mode,
            repaint_strength,
        )
        if repaint_latent_crossfade_frames != 10:
            resolved_repaint_crossfade_frames = max(0, int(repaint_latent_crossfade_frames))

        if (
            allow_cpu_device_fallback
            and self._should_preflight_cpu_stability_mode()
            and hasattr(self, "switch_to_cpu_stability_preset")
        ):
            logger.warning(
                "[generate_music] Preflight detected unstable quantized CUDA profile; "
                "switching to CPU stability preset before diffusion."
            )
            switch_status, switch_ok = self.switch_to_cpu_stability_preset()
            if switch_ok:
                logger.info("[generate_music] {}", switch_status)
                return self.generate_music(
                    captions=captions,
                    global_caption=global_caption,
                    lyrics=lyrics,
                    bpm=bpm,
                    key_scale=key_scale,
                    time_signature=time_signature,
                    vocal_language=vocal_language,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    use_random_seed=use_random_seed,
                    seed=seed,
                    reference_audio=reference_audio,
                    audio_duration=audio_duration,
                    batch_size=batch_size,
                    src_audio=src_audio,
                    audio_code_string=audio_code_string,
                    repainting_start=repainting_start,
                    repainting_end=repainting_end,
                    instruction=instruction,
                    audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    task_type=task_type,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    shift=shift,
                    infer_method=infer_method,
                    use_tiled_decode=use_tiled_decode,
                    timesteps=timesteps,
                    latent_shift=latent_shift,
                    latent_rescale=latent_rescale,
                    repaint_latent_crossfade_frames=repaint_latent_crossfade_frames,
                    repaint_wav_crossfade_sec=repaint_wav_crossfade_sec,
                    repaint_mode=repaint_mode,
                    repaint_strength=repaint_strength,
                    progress=progress,
                    _allow_dequant_fallback=False,
                    _allow_quantized_mode_fallback=False,
                    _allow_cpu_device_fallback=False,
                )
            logger.warning(
                "[generate_music] Preflight CPU stability switch failed; continuing with current runtime: {}",
                switch_status,
            )

        logger.info("[generate_music] Starting generation...")
        if progress:
            progress(0.05, desc="Preparing inputs...")
        logger.info("[generate_music] Preparing inputs...")

        runtime_start = time.perf_counter()
        runtime = self._prepare_generate_music_runtime(
            batch_size=batch_size,
            audio_duration=audio_duration,
            repainting_end=repainting_end,
            seed=seed,
            use_random_seed=use_random_seed,
        )
        stage_timings["runtime_prep_sec"] = time.perf_counter() - runtime_start
        actual_batch_size = runtime["actual_batch_size"]
        actual_seed_list = runtime["actual_seed_list"]
        seed_value_for_ui = runtime["seed_value_for_ui"]
        audio_duration = runtime["audio_duration"]
        repainting_end = runtime["repainting_end"]
        if progress:
            progress(
                self._scale_setup_progress_checkpoint(service_start_progress, 0.20),
                desc="Preparing reference/source audio...",
            )

        try:
            reference_stage_start = time.perf_counter()
            refer_audios, processed_src_audio, audio_error = self._prepare_reference_and_source_audio(
                reference_audio=reference_audio,
                src_audio=src_audio,
                audio_code_string=audio_code_string,
                actual_batch_size=actual_batch_size,
                task_type=task_type,
            )
            stage_timings["reference_audio_stage_sec"] = time.perf_counter() - reference_stage_start
            if audio_error is not None:
                return audio_error
            if progress:
                progress(
                    self._scale_setup_progress_checkpoint(service_start_progress, 0.52),
                    desc="Building conditioning inputs...",
                )

            service_input_start = time.perf_counter()
            service_inputs = self._prepare_generate_music_service_inputs(
                actual_batch_size=actual_batch_size,
                processed_src_audio=processed_src_audio,
                audio_duration=audio_duration,
                captions=captions,
                global_caption=global_caption,
                lyrics=lyrics,
                vocal_language=vocal_language,
                instruction=instruction,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                task_type=task_type,
                audio_code_string=audio_code_string,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
                chunk_mask_mode=chunk_mask_mode,
            )
            stage_timings["service_input_stage_sec"] = time.perf_counter() - service_input_start
            if progress:
                progress(
                    self._scale_setup_progress_checkpoint(service_start_progress, 0.76),
                    desc="Running VRAM and setup checks...",
                )

            preflight_start = time.perf_counter()
            vram_error = self._vram_preflight_check(
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                guidance_scale=guidance_scale,
            )
            stage_timings["preflight_stage_sec"] = time.perf_counter() - preflight_start
            if vram_error is not None:
                return vram_error
            stage_timings["setup_before_service_sec"] = (
                stage_timings.get("runtime_prep_sec", 0.0)
                + stage_timings.get("reference_audio_stage_sec", 0.0)
                + stage_timings.get("service_input_stage_sec", 0.0)
                + stage_timings.get("preflight_stage_sec", 0.0)
            )
            if self._should_run_cuda_stability_canary():
                logger.info(
                    "[generate_music] Running CUDA stability canary "
                    "(1-step diffusion probe before full generation)."
                )
                if progress:
                    progress(
                        self._scale_setup_progress_checkpoint(service_start_progress, 0.88),
                        desc="Running CUDA stability preflight...",
                    )
                canary_start = time.perf_counter()
                canary_run = self._run_generate_music_service_with_progress(
                    progress=None,
                    actual_batch_size=actual_batch_size,
                    audio_duration=audio_duration,
                    inference_steps=1,
                    timesteps=None,
                    service_inputs=service_inputs,
                    refer_audios=refer_audios,
                    guidance_scale=guidance_scale,
                    actual_seed_list=actual_seed_list,
                    audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    shift=shift,
                    infer_method=infer_method,
                    repaint_crossfade_frames=resolved_repaint_crossfade_frames,
                    repaint_injection_ratio=repaint_injection_ratio,
                    phase_ranges=phase_ranges,
                )
                stage_timings["canary_service_generate_sec"] = time.perf_counter() - canary_start
                canary_outputs = canary_run["outputs"]
                canary_infer_steps = canary_run["infer_steps_for_progress"]
                try:
                    self._prepare_generate_music_decode_state(
                        outputs=canary_outputs,
                        infer_steps_for_progress=canary_infer_steps,
                        actual_batch_size=actual_batch_size,
                        audio_duration=audio_duration,
                        latent_shift=latent_shift,
                        latent_rescale=latent_rescale,
                    )
                except RuntimeError as canary_exc:
                    if self._is_non_finite_latents_error(canary_exc):
                        logger.warning(
                            "[generate_music] CUDA stability canary detected non-finite latents; "
                            "switching to CPU stability preset before full generation."
                        )
                        canary_cpu_fallback = (
                            allow_cpu_device_fallback
                            and hasattr(self, "switch_to_cpu_stability_preset")
                        )
                        if canary_cpu_fallback:
                            switch_status, switch_ok = self.switch_to_cpu_stability_preset()
                            if switch_ok:
                                logger.info("[generate_music] {}", switch_status)
                                return self.generate_music(
                                    captions=captions,
                                    global_caption=global_caption,
                                    lyrics=lyrics,
                                    bpm=bpm,
                                    key_scale=key_scale,
                                    time_signature=time_signature,
                                    vocal_language=vocal_language,
                                    inference_steps=inference_steps,
                                    guidance_scale=guidance_scale,
                                    use_random_seed=use_random_seed,
                                    seed=seed,
                                    reference_audio=reference_audio,
                                    audio_duration=audio_duration,
                                    batch_size=batch_size,
                                    src_audio=src_audio,
                                    audio_code_string=audio_code_string,
                                    repainting_start=repainting_start,
                                    repainting_end=repainting_end,
                                    instruction=instruction,
                                    audio_cover_strength=audio_cover_strength,
                                    cover_noise_strength=cover_noise_strength,
                                    task_type=task_type,
                                    use_adg=use_adg,
                                    cfg_interval_start=cfg_interval_start,
                                    cfg_interval_end=cfg_interval_end,
                                    shift=shift,
                                    infer_method=infer_method,
                                    use_tiled_decode=use_tiled_decode,
                                    timesteps=timesteps,
                                    latent_shift=latent_shift,
                                    latent_rescale=latent_rescale,
                                    repaint_latent_crossfade_frames=repaint_latent_crossfade_frames,
                                    repaint_wav_crossfade_sec=repaint_wav_crossfade_sec,
                                    repaint_mode=repaint_mode,
                                    repaint_strength=repaint_strength,
                                    progress=progress,
                                    _allow_dequant_fallback=False,
                                    _allow_quantized_mode_fallback=False,
                                    _allow_cpu_device_fallback=False,
                                )
                            logger.warning(
                                "[generate_music] Failed to switch to CPU stability preset after canary: {}",
                                switch_status,
                            )
                    else:
                        logger.warning(
                            "[generate_music] CUDA stability canary failed with non-latent error "
                            "({}); continuing with normal generation.",
                            canary_exc,
                        )
                finally:
                    canary_outputs = None
                    canary_run = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if progress:
                progress(service_start_progress, desc="Starting diffusion...")

            service_start = time.perf_counter()
            service_run = self._run_generate_music_service_with_progress(
                progress=progress,
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                inference_steps=inference_steps,
                timesteps=timesteps,
                service_inputs=service_inputs,
                refer_audios=refer_audios,
                guidance_scale=guidance_scale,
                actual_seed_list=actual_seed_list,
                audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    shift=shift,
                    infer_method=infer_method,
                    repaint_crossfade_frames=resolved_repaint_crossfade_frames,
                    repaint_injection_ratio=repaint_injection_ratio,
                    phase_ranges=phase_ranges,
                )
            stage_timings["service_generate_sec"] = time.perf_counter() - service_start
            outputs = service_run["outputs"]
            if isinstance(outputs, dict):
                maybe_time_costs = outputs.get("time_costs")
                if isinstance(maybe_time_costs, dict):
                    last_time_costs = maybe_time_costs
            infer_steps_for_progress = service_run["infer_steps_for_progress"]

            try:
                decode_state_start = time.perf_counter()
                pred_latents, time_costs = self._prepare_generate_music_decode_state(
                    outputs=outputs,
                    infer_steps_for_progress=infer_steps_for_progress,
                    actual_batch_size=actual_batch_size,
                    audio_duration=audio_duration,
                    latent_shift=latent_shift,
                    latent_rescale=latent_rescale,
                )
                stage_timings["decode_state_stage_sec"] = time.perf_counter() - decode_state_start
            except RuntimeError as decode_exc:
                if not self._should_retry_after_non_finite_latents(
                    decode_exc,
                    guidance_scale=guidance_scale,
                    use_adg=use_adg,
                ):
                    raise

                logger.warning(
                    "[generate_music] Non-finite latents detected; retrying diffusion once "
                    "with safer settings (guidance_scale=1.0, use_adg=False)."
                )
                if progress:
                    progress(diffusion_end_progress, desc="Retrying diffusion with safer settings...")
                retry_service_start = time.perf_counter()
                service_run = self._run_generate_music_service_with_progress(
                    progress=None,
                    actual_batch_size=actual_batch_size,
                    audio_duration=audio_duration,
                    inference_steps=inference_steps,
                    timesteps=timesteps,
                    service_inputs=service_inputs,
                    refer_audios=refer_audios,
                    guidance_scale=1.0,
                    actual_seed_list=actual_seed_list,
                    audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    use_adg=False,
                    cfg_interval_start=0.0,
                    cfg_interval_end=1.0,
                    shift=shift,
                    infer_method=infer_method,
                    repaint_crossfade_frames=resolved_repaint_crossfade_frames,
                    repaint_injection_ratio=repaint_injection_ratio,
                    phase_ranges=phase_ranges,
                )
                stage_timings["service_generate_sec"] = (
                    stage_timings.get("service_generate_sec", 0.0)
                    + (time.perf_counter() - retry_service_start)
                )
                outputs = service_run["outputs"]
                if isinstance(outputs, dict):
                    maybe_time_costs = outputs.get("time_costs")
                    if isinstance(maybe_time_costs, dict):
                        last_time_costs = maybe_time_costs
                infer_steps_for_progress = service_run["infer_steps_for_progress"]
                decode_state_start = time.perf_counter()
                try:
                    pred_latents, time_costs = self._prepare_generate_music_decode_state(
                        outputs=outputs,
                        infer_steps_for_progress=infer_steps_for_progress,
                        actual_batch_size=actual_batch_size,
                        audio_duration=audio_duration,
                        latent_shift=latent_shift,
                        latent_rescale=latent_rescale,
                    )
                except RuntimeError as second_decode_exc:
                    quantized_runtime = getattr(self, "quantization", None) is not None
                    if not (self._is_non_finite_latents_error(second_decode_exc) and quantized_runtime):
                        raise
                    fallback_steps = self._resolve_quantized_stability_fallback_steps(inference_steps)
                    fallback_infer_method = "sde" if infer_method != "sde" else infer_method
                    logger.warning(
                        "[generate_music] Safe retry still produced non-finite latents on quantized runtime; "
                        "retrying once more with fresh seed, reduced steps ({}) and infer_method={} for stability.",
                        fallback_steps,
                        fallback_infer_method,
                    )
                    if progress:
                        progress(diffusion_end_progress, desc="Retrying with fallback stability settings...")
                    final_retry_start = time.perf_counter()
                    service_run = self._run_generate_music_service_with_progress(
                        progress=None,
                        actual_batch_size=actual_batch_size,
                        audio_duration=audio_duration,
                        inference_steps=fallback_steps,
                        timesteps=None,
                        service_inputs=service_inputs,
                        refer_audios=refer_audios,
                        guidance_scale=1.0,
                        actual_seed_list=None,
                        audio_cover_strength=audio_cover_strength,
                        cover_noise_strength=cover_noise_strength,
                        use_adg=False,
                        cfg_interval_start=0.0,
                        cfg_interval_end=1.0,
                        shift=shift,
                        infer_method=fallback_infer_method,
                        repaint_crossfade_frames=resolved_repaint_crossfade_frames,
                        repaint_injection_ratio=repaint_injection_ratio,
                        phase_ranges=phase_ranges,
                    )
                    stage_timings["service_generate_sec"] = (
                        stage_timings.get("service_generate_sec", 0.0)
                        + (time.perf_counter() - final_retry_start)
                    )
                    outputs = service_run["outputs"]
                    if isinstance(outputs, dict):
                        maybe_time_costs = outputs.get("time_costs")
                        if isinstance(maybe_time_costs, dict):
                            last_time_costs = maybe_time_costs
                    infer_steps_for_progress = service_run["infer_steps_for_progress"]
                    try:
                        pred_latents, time_costs = self._prepare_generate_music_decode_state(
                            outputs=outputs,
                            infer_steps_for_progress=infer_steps_for_progress,
                            actual_batch_size=actual_batch_size,
                            audio_duration=audio_duration,
                            latent_shift=latent_shift,
                            latent_rescale=latent_rescale,
                        )
                    except RuntimeError as final_decode_exc:
                        can_dequant_fallback_prereqs = (
                            allow_dequant_fallback
                            and self._is_non_finite_latents_error(final_decode_exc)
                            and getattr(self, "quantization", None) is not None
                            and hasattr(self, "switch_to_training_preset")
                        )
                        can_dequant_fallback = (
                            can_dequant_fallback_prereqs
                            and self._can_attempt_non_quantized_fallback()
                        )
                        can_quantized_mode_fallback = (
                            allow_quantized_mode_fallback
                            and self._is_non_finite_latents_error(final_decode_exc)
                            and getattr(self, "quantization", None) == "w8a8_dynamic"
                            and hasattr(self, "switch_to_stable_quantized_preset")
                            and not can_dequant_fallback
                        )
                        if can_quantized_mode_fallback:
                            logger.warning(
                                "[generate_music] w8a8_dynamic remained unstable after retries; "
                                "switching to int8_weight_only quantization and retrying once."
                            )
                            if progress:
                                progress(
                                    diffusion_end_progress,
                                    desc="Switching to stable quantized mode and retrying...",
                                )
                            switch_status, switch_ok = self.switch_to_stable_quantized_preset()
                            if switch_ok:
                                logger.info("[generate_music] {}", switch_status)
                                return self.generate_music(
                                    captions=captions,
                                    global_caption=global_caption,
                                    lyrics=lyrics,
                                    bpm=bpm,
                                    key_scale=key_scale,
                                    time_signature=time_signature,
                                    vocal_language=vocal_language,
                                    inference_steps=inference_steps,
                                    guidance_scale=guidance_scale,
                                    use_random_seed=use_random_seed,
                                    seed=seed,
                                    reference_audio=reference_audio,
                                    audio_duration=audio_duration,
                                    batch_size=batch_size,
                                    src_audio=src_audio,
                                    audio_code_string=audio_code_string,
                                    repainting_start=repainting_start,
                                    repainting_end=repainting_end,
                                    instruction=instruction,
                                    audio_cover_strength=audio_cover_strength,
                                    cover_noise_strength=cover_noise_strength,
                                    task_type=task_type,
                                    use_adg=use_adg,
                                    cfg_interval_start=cfg_interval_start,
                                    cfg_interval_end=cfg_interval_end,
                                    shift=shift,
                                    infer_method=infer_method,
                                    use_tiled_decode=use_tiled_decode,
                                    timesteps=timesteps,
                                    latent_shift=latent_shift,
                                    latent_rescale=latent_rescale,
                                    repaint_latent_crossfade_frames=repaint_latent_crossfade_frames,
                                    repaint_wav_crossfade_sec=repaint_wav_crossfade_sec,
                                    repaint_mode=repaint_mode,
                                    repaint_strength=repaint_strength,
                                    progress=progress,
                                    _allow_dequant_fallback=allow_dequant_fallback,
                                    _allow_quantized_mode_fallback=False,
                                    _allow_cpu_device_fallback=allow_cpu_device_fallback,
                                )
                            logger.warning(
                                "[generate_music] Failed to switch quantization mode automatically: {}",
                                switch_status,
                            )
                        can_cpu_device_fallback = (
                            allow_cpu_device_fallback
                            and self._is_non_finite_latents_error(final_decode_exc)
                            and getattr(self, "device", None) == "cuda"
                            and getattr(self, "quantization", None) == "int8_weight_only"
                            and hasattr(self, "switch_to_cpu_stability_preset")
                            and not can_dequant_fallback
                        )
                        if can_cpu_device_fallback:
                            logger.warning(
                                "[generate_music] Quantized low-VRAM CUDA runtime remained unstable after retries; "
                                "switching to CPU stability preset and retrying once."
                            )
                            if progress:
                                progress(
                                    diffusion_end_progress,
                                    desc="Switching to CPU stability mode and retrying...",
                                )
                            switch_status, switch_ok = self.switch_to_cpu_stability_preset()
                            if switch_ok:
                                logger.info("[generate_music] {}", switch_status)
                                return self.generate_music(
                                    captions=captions,
                                    global_caption=global_caption,
                                    lyrics=lyrics,
                                    bpm=bpm,
                                    key_scale=key_scale,
                                    time_signature=time_signature,
                                    vocal_language=vocal_language,
                                    inference_steps=inference_steps,
                                    guidance_scale=guidance_scale,
                                    use_random_seed=use_random_seed,
                                    seed=seed,
                                    reference_audio=reference_audio,
                                    audio_duration=audio_duration,
                                    batch_size=batch_size,
                                    src_audio=src_audio,
                                    audio_code_string=audio_code_string,
                                    repainting_start=repainting_start,
                                    repainting_end=repainting_end,
                                    instruction=instruction,
                                    audio_cover_strength=audio_cover_strength,
                                    cover_noise_strength=cover_noise_strength,
                                    task_type=task_type,
                                    use_adg=use_adg,
                                    cfg_interval_start=cfg_interval_start,
                                    cfg_interval_end=cfg_interval_end,
                                    shift=shift,
                                    infer_method=infer_method,
                                    use_tiled_decode=use_tiled_decode,
                                    timesteps=timesteps,
                                    latent_shift=latent_shift,
                                    latent_rescale=latent_rescale,
                                    repaint_latent_crossfade_frames=repaint_latent_crossfade_frames,
                                    repaint_wav_crossfade_sec=repaint_wav_crossfade_sec,
                                    repaint_mode=repaint_mode,
                                    repaint_strength=repaint_strength,
                                    progress=progress,
                                    _allow_dequant_fallback=False,
                                    _allow_quantized_mode_fallback=False,
                                    _allow_cpu_device_fallback=False,
                                )
                            logger.warning(
                                "[generate_music] Failed to switch to CPU stability preset automatically: {}",
                                switch_status,
                            )
                        if not can_dequant_fallback:
                            if (
                                can_dequant_fallback_prereqs
                                and not can_dequant_fallback
                            ):
                                logger.warning(
                                    "[generate_music] Skipping non-quantized fallback on low-VRAM GPU; "
                                    "model is unlikely to fit without quantization."
                                )
                            raise
                        logger.warning(
                            "[generate_music] Quantized runtime remained unstable after retries; "
                            "switching to non-quantized preset and retrying generation once."
                        )
                        if progress:
                            progress(
                                diffusion_end_progress,
                                desc="Switching to non-quantized mode and retrying...",
                            )
                        switch_status, switch_ok = self.switch_to_training_preset()
                        if not switch_ok:
                            raise RuntimeError(
                                "Failed to auto-switch to non-quantized preset after unstable diffusion. "
                                f"Details: {switch_status}"
                            ) from final_decode_exc
                        logger.info("[generate_music] {}", switch_status)
                        return self.generate_music(
                            captions=captions,
                            global_caption=global_caption,
                            lyrics=lyrics,
                            bpm=bpm,
                            key_scale=key_scale,
                            time_signature=time_signature,
                            vocal_language=vocal_language,
                            inference_steps=inference_steps,
                            guidance_scale=guidance_scale,
                            use_random_seed=use_random_seed,
                            seed=seed,
                            reference_audio=reference_audio,
                            audio_duration=audio_duration,
                            batch_size=batch_size,
                            src_audio=src_audio,
                            audio_code_string=audio_code_string,
                            repainting_start=repainting_start,
                            repainting_end=repainting_end,
                            instruction=instruction,
                            audio_cover_strength=audio_cover_strength,
                            cover_noise_strength=cover_noise_strength,
                            task_type=task_type,
                            use_adg=use_adg,
                            cfg_interval_start=cfg_interval_start,
                            cfg_interval_end=cfg_interval_end,
                            shift=shift,
                            infer_method=infer_method,
                            use_tiled_decode=use_tiled_decode,
                            timesteps=timesteps,
                            latent_shift=latent_shift,
                            latent_rescale=latent_rescale,
                            repaint_latent_crossfade_frames=repaint_latent_crossfade_frames,
                            repaint_wav_crossfade_sec=repaint_wav_crossfade_sec,
                            repaint_mode=repaint_mode,
                            repaint_strength=repaint_strength,
                            progress=progress,
                            _allow_dequant_fallback=False,
                            _allow_quantized_mode_fallback=False,
                            _allow_cpu_device_fallback=allow_cpu_device_fallback,
                        )
                stage_timings["decode_state_stage_sec"] = time.perf_counter() - decode_state_start
            decode_start = time.perf_counter()
            pred_wavs, pred_latents_cpu, time_costs = self._decode_generate_music_pred_latents(
                pred_latents=pred_latents,
                progress=progress,
                use_tiled_decode=use_tiled_decode,
                time_costs=time_costs,
                decode_progress_start=decode_start_progress,
            )
            stage_timings["decode_stage_sec"] = time.perf_counter() - decode_start
            stage_timings["decode_total_sec"] = (
                stage_timings.get("decode_state_stage_sec", 0.0)
                + stage_timings.get("decode_stage_sec", 0.0)
            )
            stage_timings["total_orchestration_sec"] = time.perf_counter() - generation_start
            time_costs["setup_time_cost"] = stage_timings.get("setup_before_service_sec", 0.0)
            time_costs["service_generate_time_cost"] = stage_timings.get("service_generate_sec", 0.0)
            phase_profile_updater = getattr(self, "_update_progress_phase_profile", None)
            if callable(phase_profile_updater):
                phase_profile_updater(stage_timings=stage_timings, time_costs=time_costs)
            if profile_progress:
                self._log_generation_progress_profile(stage_timings=stage_timings, time_costs=time_costs)
            result = self._build_generate_music_success_payload(
                outputs=outputs,
                pred_wavs=pred_wavs,
                pred_latents_cpu=pred_latents_cpu,
                time_costs=time_costs,
                seed_value_for_ui=seed_value_for_ui,
                actual_batch_size=actual_batch_size,
                progress=progress,
            )
            # Clear GPU tensor references from the mutable outputs dict so
            # accelerator memory is reclaimable before the next generation.
            _gpu_keys = (
                "src_latents", "target_latents_input", "chunk_masks",
                "latent_masks", "encoder_hidden_states",
                "encoder_attention_mask", "context_latents",
                "lyric_token_idss",
            )
            for _k in _gpu_keys:
                outputs.pop(_k, None)
            del outputs, pred_wavs, pred_latents_cpu
            gc.collect()
            self._empty_cache()
            return result
        except Exception as exc:
            if profile_progress:
                stage_timings["total_orchestration_sec"] = time.perf_counter() - generation_start
                self._log_generation_progress_profile(stage_timings=stage_timings, time_costs=last_time_costs)
            error_msg = f"Error: {exc!s}\n{traceback.format_exc()}"
            logger.exception("[generate_music] Generation failed")
            return {
                "audios": [],
                "status_message": error_msg,
                "extra_outputs": {},
                "success": False,
                "error": f"{exc!s}",
            }
