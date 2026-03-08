"""Progress estimation mixin for AceStepHandler."""

import json
import os
import threading
import time
from typing import Any, Callable, Optional

from loguru import logger
import torch

# Conservative per-step estimate used when no historical timing data exists
# (i.e., first-ever generation on this machine).  2.5s/step is deliberately
# slow so the progress bar undershoots rather than overshoots — reaching 79%
# early and pausing is far less alarming than freezing at 52% with zero
# movement.  The estimate self-corrects after the first successful generation.
_FALLBACK_PER_STEP_SEC = 2.5
_DEFAULT_PHASE_RANGES = {
    "service_start": 0.30,
    "encoding_end": 0.45,
    "diffusion_end": 0.79,
}
_DEFAULT_MIN_PHASE_PROFILE_SAMPLES = 8


class ProgressMixin:
    def _set_runtime_progress_callback(
        self,
        callback: Optional[Callable[..., Any]],
    ) -> None:
        """Set per-generation runtime progress callback used by inner loops."""
        self._runtime_progress_callback = callback

    def _emit_runtime_progress(
        self,
        stage: str,
        current: int,
        total: int,
        desc: str,
    ) -> None:
        """Emit a runtime progress event to the active callback when available."""
        callback = getattr(self, "_runtime_progress_callback", None)
        if not callable(callback):
            return
        try:
            callback(stage=stage, current=current, total=total, desc=desc)
        except Exception:
            # Progress reporting is best-effort; generation should continue.
            pass

    def _get_project_root(self) -> str:
        """Get project root directory path.

        Returns the directory set by the ``ACESTEP_PROJECT_ROOT`` environment
        variable when present, otherwise the current working directory.  Using
        the working directory (rather than ``__file__``) keeps generated cache
        files and the checkpoints folder next to where the user launched the
        process, regardless of whether the package was installed via
        ``pip install .`` or run from source.
        """
        env_root = os.environ.get("ACESTEP_PROJECT_ROOT")
        if env_root:
            return os.path.abspath(env_root)
        return os.getcwd()

    def _load_progress_estimates(self) -> None:
        """Load persisted diffusion progress estimates if available."""
        try:
            if os.path.exists(self._progress_estimates_path):
                with open(self._progress_estimates_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        records = data.get("records")
                        phase_profiles = data.get("phase_profiles")
                        self._progress_estimates = {
                            "records": records if isinstance(records, list) else [],
                            "phase_profiles": phase_profiles if isinstance(phase_profiles, dict) else {},
                        }
                        if "updated_at" in data:
                            self._progress_estimates["updated_at"] = data["updated_at"]
        except Exception:
            # Ignore corrupted cache; it will be overwritten on next save.
            self._progress_estimates = {"records": [], "phase_profiles": {}}

    def _save_progress_estimates(self) -> None:
        """Persist diffusion progress estimates."""
        try:
            os.makedirs(os.path.dirname(self._progress_estimates_path), exist_ok=True)
            with open(self._progress_estimates_path, "w", encoding="utf-8") as f:
                json.dump(self._progress_estimates, f)
        except Exception:
            pass

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp ``value`` into inclusive ``[minimum, maximum]`` bounds."""
        return max(minimum, min(maximum, value))

    def _get_progress_profile_machine_key(self) -> str:
        """Build a stable machine/runtime key for persisted progress phase profiles."""
        device = str(getattr(self, "device", "unknown"))
        quantization = str(getattr(self, "quantization", "none"))
        offload = "1" if bool(getattr(self, "offload_to_cpu", False)) else "0"
        offload_dit = "1" if bool(getattr(self, "offload_dit_to_cpu", False)) else "0"
        dtype = str(getattr(self, "dtype", "unknown"))
        capability = "na"
        gpu_name = "na"
        if device == "cuda" and torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                capability = f"{major}.{minor}"
            except Exception:
                capability = "unknown"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "unknown"
        return "|".join(
            [
                f"device:{device}",
                f"gpu:{gpu_name}",
                f"cc:{capability}",
                f"dtype:{dtype}",
                f"quant:{quantization}",
                f"offload:{offload}",
                f"offload_dit:{offload_dit}",
            ]
        )

    @staticmethod
    def _is_progress_phase_auto_profile_enabled() -> bool:
        """Return whether automatic phase profiling is enabled."""
        return os.environ.get("ACESTEP_DISABLE_PROGRESS_AUTO_PROFILE", "").lower() not in (
            "1",
            "true",
            "yes",
        )

    @staticmethod
    def _is_progress_phase_force_train_enabled() -> bool:
        """Return whether updates should continue even after profile freeze."""
        return os.environ.get("ACESTEP_PROGRESS_PROFILE_FORCE_TRAIN", "").lower() in ("1", "true", "yes")

    @staticmethod
    def _is_progress_phase_autofreeze_enabled() -> bool:
        """Return whether profile auto-freeze should occur after minimum samples."""
        return os.environ.get("ACESTEP_PROGRESS_PROFILE_AUTOFREEZE", "1").lower() in ("1", "true", "yes")

    @staticmethod
    def _get_progress_phase_profile_min_samples() -> int:
        """Return minimum sample count before phase profile auto-freeze."""
        raw = os.environ.get("ACESTEP_PROGRESS_PROFILE_MIN_SAMPLES", "").strip()
        if not raw:
            return _DEFAULT_MIN_PHASE_PROFILE_SAMPLES
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_MIN_PHASE_PROFILE_SAMPLES
        return max(1, value)

    def _normalize_progress_phase_ranges(self, ranges: Optional[dict[str, float]]) -> dict[str, float]:
        """Normalize phase ranges so service->encoding->diffusion boundaries are monotonic."""
        resolved = dict(_DEFAULT_PHASE_RANGES)
        if isinstance(ranges, dict):
            for key in ("service_start", "encoding_end", "diffusion_end"):
                value = ranges.get(key)
                if isinstance(value, (int, float)):
                    resolved[key] = float(value)

        service_start = self._clamp(resolved["service_start"], 0.05, 0.55)
        encoding_end = self._clamp(resolved["encoding_end"], service_start + 0.04, 0.88)
        diffusion_end = self._clamp(resolved["diffusion_end"], encoding_end + 0.08, 0.95)
        return {
            "service_start": service_start,
            "encoding_end": encoding_end,
            "diffusion_end": diffusion_end,
        }

    def _phase_ranges_from_profile(
        self,
        setup_ratio: float,
        decode_ratio: float,
        diffusion_ratio_of_service: float,
    ) -> dict[str, float]:
        """Convert observed stage ratios into progress-bar phase boundaries."""
        service_start = self._clamp(setup_ratio, 0.08, 0.35)
        decode_span = self._clamp(decode_ratio, 0.18, 0.45)
        diffusion_end = 1.0 - decode_span
        if diffusion_end < service_start + 0.20:
            diffusion_end = service_start + 0.20
        diffusion_end = self._clamp(diffusion_end, service_start + 0.20, 0.92)
        service_span = diffusion_end - service_start
        encoding_ratio = self._clamp(1.0 - diffusion_ratio_of_service, 0.12, 0.55)
        encoding_end = service_start + (service_span * encoding_ratio)
        return self._normalize_progress_phase_ranges(
            {
                "service_start": service_start,
                "encoding_end": encoding_end,
                "diffusion_end": diffusion_end,
            }
        )

    def _get_progress_phase_ranges(self) -> dict[str, float]:
        """Return persisted machine-specific phase ranges or safe defaults."""
        machine_key = self._get_progress_profile_machine_key()
        with self._progress_estimates_lock:
            phase_profiles = self._progress_estimates.get("phase_profiles", {})
            profile = phase_profiles.get(machine_key, {}) if isinstance(phase_profiles, dict) else {}
        ranges = profile.get("phase_ranges") if isinstance(profile, dict) else None
        return self._normalize_progress_phase_ranges(ranges)

    def _update_progress_phase_profile(
        self,
        *,
        stage_timings: dict[str, float],
        time_costs: Optional[dict[str, Any]],
    ) -> None:
        """Update persistent machine-specific phase profile from one generation run."""
        if not self._is_progress_phase_auto_profile_enabled():
            return

        def _to_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        setup = _to_float(stage_timings.get("setup_before_service_sec", 0.0))
        service = _to_float(stage_timings.get("service_generate_sec", 0.0))
        decode = _to_float(stage_timings.get("decode_total_sec", 0.0))
        diffusion = 0.0
        if isinstance(time_costs, dict):
            diffusion = _to_float(time_costs.get("diffusion_time_cost", 0.0))
            if decode <= 0:
                decode = _to_float(time_costs.get("vae_decode_time_cost", 0.0))
            if service <= 0:
                total_cost = _to_float(time_costs.get("total_time_cost", 0.0))
                service = max(0.0, total_cost - decode)
        observed_total = setup + service + decode
        if observed_total <= 0:
            return

        setup_ratio = self._clamp(setup / observed_total, 0.0, 0.70)
        decode_ratio = self._clamp(decode / observed_total, 0.0, 0.80)
        diffusion_ratio_of_service = self._clamp(
            (diffusion / service) if service > 0 else 0.75,
            0.0,
            1.0,
        )

        machine_key = self._get_progress_profile_machine_key()
        min_samples = self._get_progress_phase_profile_min_samples()
        allow_force_train = self._is_progress_phase_force_train_enabled()
        with self._progress_estimates_lock:
            phase_profiles = self._progress_estimates.setdefault("phase_profiles", {})
            profile = phase_profiles.get(machine_key, {})
            sample_count = int(profile.get("sample_count", 0) or 0)
            frozen = bool(profile.get("frozen", False))
            if frozen and not allow_force_train:
                return

            ratios = profile.get("ratios") if isinstance(profile.get("ratios"), dict) else {}
            old_setup = float(ratios.get("setup_ratio", setup_ratio))
            old_decode = float(ratios.get("decode_ratio", decode_ratio))
            old_diff_service = float(
                ratios.get("diffusion_ratio_of_service", diffusion_ratio_of_service)
            )

            new_count = sample_count + 1
            avg_setup = ((old_setup * sample_count) + setup_ratio) / new_count
            avg_decode = ((old_decode * sample_count) + decode_ratio) / new_count
            avg_diff_service = ((old_diff_service * sample_count) + diffusion_ratio_of_service) / new_count
            phase_ranges = self._phase_ranges_from_profile(
                setup_ratio=avg_setup,
                decode_ratio=avg_decode,
                diffusion_ratio_of_service=avg_diff_service,
            )

            frozen = (
                self._is_progress_phase_autofreeze_enabled()
                and (new_count >= min_samples)
                and not allow_force_train
            )
            phase_profiles[machine_key] = {
                "sample_count": new_count,
                "frozen": frozen,
                "ratios": {
                    "setup_ratio": avg_setup,
                    "decode_ratio": avg_decode,
                    "diffusion_ratio_of_service": avg_diff_service,
                },
                "phase_ranges": phase_ranges,
                "updated_at": time.time(),
            }
            self._progress_estimates["phase_profiles"] = phase_profiles
            self._progress_estimates["updated_at"] = time.time()
            self._save_progress_estimates()

    def _duration_bucket(self, duration_sec: Optional[float]) -> str:
        if duration_sec is None or duration_sec <= 0:
            return "unknown"
        if duration_sec <= 60:
            return "short"
        if duration_sec <= 180:
            return "medium"
        if duration_sec <= 360:
            return "long"
        return "xlong"

    def _update_progress_estimate(
        self,
        per_step_sec: float,
        infer_steps: int,
        batch_size: int,
        duration_sec: Optional[float],
    ) -> None:
        if per_step_sec <= 0 or infer_steps <= 0:
            return
        record = {
            "device": self.device,
            "infer_steps": int(infer_steps),
            "batch_size": int(batch_size),
            "duration_sec": float(duration_sec) if duration_sec and duration_sec > 0 else None,
            "duration_bucket": self._duration_bucket(duration_sec),
            "per_step_sec": float(per_step_sec),
            "updated_at": time.time(),
        }
        with self._progress_estimates_lock:
            records = self._progress_estimates.get("records", [])
            records.append(record)
            # Keep recent 100 records
            records = records[-100:]
            self._progress_estimates["records"] = records
            self._progress_estimates["updated_at"] = time.time()
            self._save_progress_estimates()

    def _estimate_diffusion_per_step(
        self,
        infer_steps: int,
        batch_size: int,
        duration_sec: Optional[float],
    ) -> Optional[float]:
        # Prefer most recent exact-ish record
        target_bucket = self._duration_bucket(duration_sec)
        with self._progress_estimates_lock:
            records = list(self._progress_estimates.get("records", []))
        if not records:
            return None

        # Filter by device first
        device_records = [r for r in records if r.get("device") == self.device] or records

        # Exact match by steps/batch/bucket
        for r in reversed(device_records):
            if (
                r.get("infer_steps") == infer_steps
                and r.get("batch_size") == batch_size
                and r.get("duration_bucket") == target_bucket
            ):
                return r.get("per_step_sec")

        # Same steps + bucket, scale by batch and duration when possible
        for r in reversed(device_records):
            if r.get("infer_steps") == infer_steps and r.get("duration_bucket") == target_bucket:
                base = r.get("per_step_sec")
                base_batch = r.get("batch_size", batch_size)
                base_dur = r.get("duration_sec")
                if base and base_batch:
                    est = base * (batch_size / base_batch)
                    if duration_sec and base_dur:
                        est *= (duration_sec / base_dur)
                    return est

        # Same steps, scale by batch and duration ratio if available
        for r in reversed(device_records):
            if r.get("infer_steps") == infer_steps:
                base = r.get("per_step_sec")
                base_batch = r.get("batch_size", batch_size)
                base_dur = r.get("duration_sec")
                if base and base_batch:
                    est = base * (batch_size / base_batch)
                    if duration_sec and base_dur:
                        est *= (duration_sec / base_dur)
                    return est

        # Fallback to global median
        per_steps = [r.get("per_step_sec") for r in device_records if r.get("per_step_sec")]
        if per_steps:
            per_steps.sort()
            return per_steps[len(per_steps) // 2]
        return None

    def _start_diffusion_progress_estimator(
        self,
        progress,
        start: float,
        end: float,
        infer_steps: int,
        batch_size: int,
        duration_sec: Optional[float],
        desc: str,
    ):
        """Best-effort progress updates during diffusion using previous step timing.

        Falls back to a conservative default estimate when no historical data
        exists (first-ever generation).  This ensures the progress bar always
        moves during Phase 2 instead of freezing at 52%.
        """
        if progress is None or infer_steps <= 0:
            return None, None
        per_step = self._estimate_diffusion_per_step(
            infer_steps=infer_steps,
            batch_size=batch_size,
            duration_sec=duration_sec,
        ) or self._last_diffusion_per_step_sec

        if not per_step or per_step <= 0:
            # No history at all — use conservative fallback so progress bar
            # still moves on first run.  Scale by batch size for a rough
            # approximation.
            per_step = _FALLBACK_PER_STEP_SEC * max(1, batch_size)
            logger.info(
                f"[progress] No timing history — using fallback estimate "
                f"({per_step:.1f}s/step for batch_size={batch_size}).  "
                f"This will self-calibrate after the first generation."
            )

        expected = per_step * infer_steps
        if expected <= 0:
            return None, None
        stop_event = threading.Event()

        def _runner():
            start_time = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                frac = min(0.999, elapsed / expected)
                value = start + (end - start) * frac
                try:
                    progress(value, desc=desc)
                except Exception:
                    pass
                stop_event.wait(0.5)

        thread = threading.Thread(target=_runner, name="diffusion-progress", daemon=True)
        thread.start()
        return stop_event, thread
