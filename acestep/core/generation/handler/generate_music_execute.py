"""Execution helper for ``generate_music`` service invocation with progress tracking."""

import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

# Maximum wall-clock seconds to wait for service_generate before declaring a hang.
# Generous default: most generations finish in 30-120s, but large batches on slow
# GPUs can take several minutes.  Override via ACESTEP_GENERATION_TIMEOUT env var.
_DEFAULT_GENERATION_TIMEOUT = int(os.environ.get("ACESTEP_GENERATION_TIMEOUT", "600"))


class GenerateMusicExecuteMixin:
    """Run service generation with timeout and thread-safe progress forwarding."""

    def _run_generate_music_service_with_progress(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
    ) -> Dict[str, Any]:
        """Invoke ``service_generate`` while relaying progress to the request thread.

        Wraps the synchronous CUDA call in a monitored thread so that a hung
        diffusion loop becomes a recoverable ``TimeoutError`` instead of a
        permanent UI freeze.
        """
        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress_desc = f"Generating music (batch size: {actual_batch_size})..."
        max_progress = 0.0
        max_progress_lock = threading.Lock()

        def _report_progress(value: float, desc: Optional[str] = None) -> None:
            """Emit monotonic progress updates to avoid bar regressions."""
            nonlocal max_progress
            if progress is None:
                return
            with max_progress_lock:
                next_value = max(max_progress, float(value))
                max_progress = next_value
            progress(next_value, desc=desc or progress_desc)

        _report_progress(0.10, desc=progress_desc)

        stage_ranges = {
            "encoding": (0.10, 0.30),
            "diffusion": (0.30, 0.79),
        }
        runtime_progress_setter = getattr(self, "_set_runtime_progress_callback", None)

        def _service_progress_callback(stage: str, current: int, total: int, desc: str) -> None:
            """Map service stage progress into the main Gradio progress range."""
            if stage not in stage_ranges or total <= 0:
                return
            start, end = stage_ranges[stage]
            ratio = min(1.0, max(0.0, float(current) / float(total)))
            progress_events.put((start + (end - start) * ratio, desc))

        progress_events: "queue.Queue[tuple[float, str]]" = queue.Queue()
        start_wait_ts = time.monotonic()
        per_step_estimate = None
        estimate_fn = getattr(self, "_estimate_diffusion_per_step", None)
        if callable(estimate_fn):
            per_step_estimate = estimate_fn(
                infer_steps=infer_steps_for_progress,
                batch_size=actual_batch_size,
                duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
            ) or getattr(self, "_last_diffusion_per_step_sec", None)
        if not per_step_estimate:
            per_step_estimate = 2.5 * max(1, actual_batch_size)
        expected_sec = float(per_step_estimate) * max(1, infer_steps_for_progress)

        def _drain_progress_events() -> None:
            """Flush queued service progress events on the request thread."""
            while True:
                try:
                    value, desc = progress_events.get_nowait()
                except queue.Empty:
                    break
                _report_progress(value, desc=desc)

        # --- Timeout-wrapped service_generate ---
        # Run the actual CUDA work in a child thread so we can join() with a
        # deadline.  If it exceeds the timeout the calling thread unblocks and
        # raises TimeoutError, which propagates to generate_music()'s
        # try/except and becomes a clean error payload for the UI.
        _result: Dict[str, Any] = {}
        _error: Dict[str, BaseException] = {}

        def _service_target():
            try:
                _result["outputs"] = self.service_generate(
                    captions=service_inputs["captions_batch"],
                    global_captions=service_inputs.get("global_captions_batch"),
                    lyrics=service_inputs["lyrics_batch"],
                    metas=service_inputs["metas_batch"],
                    vocal_languages=service_inputs["vocal_languages_batch"],
                    refer_audios=refer_audios,
                    target_wavs=service_inputs["target_wavs_tensor"],
                    infer_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=actual_seed_list,
                    repainting_start=service_inputs["repainting_start_batch"],
                    repainting_end=service_inputs["repainting_end_batch"],
                    instructions=service_inputs["instructions_batch"],
                    audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    shift=shift,
                    infer_method=infer_method,
                    audio_code_hints=service_inputs["audio_code_hints_batch"],
                    return_intermediate=service_inputs["should_return_intermediate"],
                    timesteps=timesteps,
                    chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
                )
            except Exception as exc:
                _error["exc"] = exc

        try:
            if callable(runtime_progress_setter):
                runtime_progress_setter(_service_progress_callback)

            gen_thread = threading.Thread(
                target=_service_target,
                name="service-generate",
                daemon=True,
            )
            gen_thread.start()
            deadline = start_wait_ts + _DEFAULT_GENERATION_TIMEOUT
            while gen_thread.is_alive():
                gen_thread.join(timeout=0.1)
                _drain_progress_events()
                elapsed = time.monotonic() - start_wait_ts
                est_frac = min(0.999, elapsed / expected_sec) if expected_sec > 0 else 0.0
                _report_progress(0.30 + (0.79 - 0.30) * est_frac, desc=progress_desc)
                if time.monotonic() >= deadline and gen_thread.is_alive():
                    logger.error(
                        f"[generate_music] service_generate exceeded {_DEFAULT_GENERATION_TIMEOUT}s "
                        f"timeout (batch={actual_batch_size}, steps={inference_steps}, "
                        f"duration={audio_duration}s).  The CUDA operation may still be "
                        f"running in the background."
                    )
                    raise TimeoutError(
                        f"Music generation timed out after {_DEFAULT_GENERATION_TIMEOUT} seconds.  "
                        f"This usually means the GPU ran out of VRAM or the diffusion loop "
                        f"stalled.  Try reducing batch size, duration, or inference steps."
                    )
            _drain_progress_events()
            if "exc" in _error:
                raise _error["exc"]

        finally:
            if callable(runtime_progress_setter):
                runtime_progress_setter(None)

        return {"outputs": _result["outputs"], "infer_steps_for_progress": infer_steps_for_progress}
