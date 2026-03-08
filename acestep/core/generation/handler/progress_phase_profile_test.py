"""Unit tests for persisted machine-specific progress phase profiling."""

import importlib.util
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _load_progress_mixin():
    """Load ``ProgressMixin`` directly from source for isolated unit tests."""
    spec = importlib.util.spec_from_file_location(
        "progress",
        os.path.join(os.path.dirname(__file__), "progress.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ProgressMixin


ProgressMixin = _load_progress_mixin()


class _Host(ProgressMixin):
    """Minimal host exposing ``ProgressMixin`` state required by tests."""

    def __init__(self, estimates_path: str):
        """Initialize deterministic in-memory and persisted progress state."""
        self.device = "cpu"
        self.quantization = None
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.dtype = "torch.float32"
        self._last_diffusion_per_step_sec = None
        self._progress_estimates_lock = threading.Lock()
        self._progress_estimates = {"records": [], "phase_profiles": {}}
        self._progress_estimates_path = estimates_path


class ProgressPhaseProfileTests(unittest.TestCase):
    """Verify profile persistence, compatibility, and auto-freeze behavior."""

    def _sample_stage_timings(self, setup: float, service: float, decode: float) -> dict[str, float]:
        """Build stage timing payload used by profile update tests."""
        return {
            "setup_before_service_sec": setup,
            "service_generate_sec": service,
            "decode_total_sec": decode,
        }

    def _sample_time_costs(self, diffusion: float, decode: float) -> dict[str, float]:
        """Build ``time_costs`` payload used by profile update tests."""
        return {
            "diffusion_time_cost": diffusion,
            "vae_decode_time_cost": decode,
            "total_time_cost": diffusion + decode + 1.0,
        }

    def test_load_progress_estimates_supports_legacy_schema(self):
        """Loading legacy cache files should initialize missing phase-profile maps."""
        with TemporaryDirectory() as tmp_dir:
            estimates_path = str(Path(tmp_dir) / "progress_estimates.json")
            with open(estimates_path, "w", encoding="utf-8") as file_obj:
                file_obj.write('{"records": [{"device": "cpu"}]}')
            host = _Host(estimates_path)
            host._load_progress_estimates()
            self.assertIn("records", host._progress_estimates)
            self.assertIn("phase_profiles", host._progress_estimates)
            self.assertIsInstance(host._progress_estimates["phase_profiles"], dict)

    def test_phase_profile_persists_and_reloads_across_sessions(self):
        """Updated phase ranges should persist and load for the same machine key."""
        with TemporaryDirectory() as tmp_dir:
            estimates_path = str(Path(tmp_dir) / "progress_estimates.json")
            host = _Host(estimates_path)
            host._update_progress_phase_profile(
                stage_timings=self._sample_stage_timings(setup=10.0, service=55.0, decode=35.0),
                time_costs=self._sample_time_costs(diffusion=40.0, decode=35.0),
            )

            machine_key = host._get_progress_profile_machine_key()
            persisted = host._progress_estimates["phase_profiles"][machine_key]["phase_ranges"]
            self.assertGreater(persisted["service_start"], 0.0)
            self.assertLess(persisted["diffusion_end"], 1.0)

            reloaded = _Host(estimates_path)
            reloaded._load_progress_estimates()
            loaded_ranges = reloaded._get_progress_phase_ranges()
            self.assertAlmostEqual(loaded_ranges["service_start"], persisted["service_start"], places=6)
            self.assertAlmostEqual(loaded_ranges["encoding_end"], persisted["encoding_end"], places=6)
            self.assertAlmostEqual(loaded_ranges["diffusion_end"], persisted["diffusion_end"], places=6)

    def test_phase_profile_freezes_after_min_samples(self):
        """Profile should auto-freeze and stop updating once minimum sample count is reached."""
        with TemporaryDirectory() as tmp_dir:
            estimates_path = str(Path(tmp_dir) / "progress_estimates.json")
            host = _Host(estimates_path)
            with patch.dict(
                os.environ,
                {
                    "ACESTEP_PROGRESS_PROFILE_MIN_SAMPLES": "2",
                    "ACESTEP_PROGRESS_PROFILE_AUTOFREEZE": "1",
                },
                clear=False,
            ):
                host._update_progress_phase_profile(
                    stage_timings=self._sample_stage_timings(setup=5.0, service=45.0, decode=50.0),
                    time_costs=self._sample_time_costs(diffusion=35.0, decode=50.0),
                )
                host._update_progress_phase_profile(
                    stage_timings=self._sample_stage_timings(setup=6.0, service=46.0, decode=48.0),
                    time_costs=self._sample_time_costs(diffusion=36.0, decode=48.0),
                )

                machine_key = host._get_progress_profile_machine_key()
                profile_before = host._progress_estimates["phase_profiles"][machine_key]
                self.assertEqual(profile_before["sample_count"], 2)
                self.assertTrue(profile_before["frozen"])

                frozen_ranges = dict(profile_before["phase_ranges"])
                host._update_progress_phase_profile(
                    stage_timings=self._sample_stage_timings(setup=30.0, service=60.0, decode=10.0),
                    time_costs=self._sample_time_costs(diffusion=58.0, decode=10.0),
                )
                profile_after = host._progress_estimates["phase_profiles"][machine_key]
                self.assertEqual(profile_after["sample_count"], 2)
                self.assertEqual(profile_after["phase_ranges"], frozen_ranges)


if __name__ == "__main__":
    unittest.main()
