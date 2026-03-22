"""Unit tests for ``generate_music`` execution helper mixin."""

import unittest

import torch

from acestep.core.generation.handler.generate_music_execute import GenerateMusicExecuteMixin
from acestep.core.generation.handler.service_generate_execute import ServiceGenerateExecuteMixin


class _Host(GenerateMusicExecuteMixin):
    """Minimal host implementing progress/service stubs for execute helper tests."""

    def __init__(self):
        """Capture calls for assertions."""
        self.service_calls = 0
        self._runtime_progress_callback = None

    def _set_runtime_progress_callback(self, callback):
        """Store active runtime callback used by service thread events."""
        self._runtime_progress_callback = callback

    def service_generate(self, **kwargs):
        """Record service invocation and return minimal output payload."""
        _ = kwargs
        self.service_calls += 1
        if callable(self._runtime_progress_callback):
            self._runtime_progress_callback(
                stage="encoding",
                current=1,
                total=2,
                desc="Text encoding",
            )
            self._runtime_progress_callback(
                stage="diffusion",
                current=4,
                total=8,
                desc="DiT diffusion steps",
            )
        return {"target_latents": "ok"}


class GenerateMusicExecuteMixinTests(unittest.TestCase):
    """Verify progress lifecycle and service forwarding behavior."""

    def test_run_service_with_progress_invokes_service_once(self):
        """Helper should call service once and return outputs."""
        host = _Host()
        out = host._run_generate_music_service_with_progress(
            progress=lambda *args, **kwargs: None,
            actual_batch_size=1,
            audio_duration=10.0,
            inference_steps=8,
            timesteps=None,
            service_inputs={
                "captions_batch": ["c"],
                "lyrics_batch": ["l"],
                "metas_batch": ["m"],
                "vocal_languages_batch": ["en"],
                "target_wavs_tensor": None,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "instructions_batch": ["i"],
                "audio_code_hints_batch": None,
                "should_return_intermediate": True,
            },
            refer_audios=None,
            guidance_scale=7.0,
            actual_seed_list=[1],
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            use_adg=False,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            infer_method="ode",
        )
        self.assertEqual(host.service_calls, 1)
        self.assertEqual(out["outputs"]["target_latents"], "ok")

    def test_run_service_with_progress_drains_runtime_events(self):
        """Service-thread runtime events should be forwarded to the progress callback."""
        host = _Host()
        progress_values = []

        def _progress(value, desc=None):
            """Capture forwarded progress updates."""
            progress_values.append((value, desc))

        host._run_generate_music_service_with_progress(
            progress=_progress,
            actual_batch_size=1,
            audio_duration=10.0,
            inference_steps=8,
            timesteps=None,
            service_inputs={
                "captions_batch": ["c"],
                "lyrics_batch": ["l"],
                "metas_batch": ["m"],
                "vocal_languages_batch": ["en"],
                "target_wavs_tensor": None,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "instructions_batch": ["i"],
                "audio_code_hints_batch": None,
                "should_return_intermediate": True,
            },
            refer_audios=None,
            guidance_scale=7.0,
            actual_seed_list=[1],
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            use_adg=False,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            infer_method="ode",
        )

        self.assertTrue(progress_values)
        self.assertTrue(any(value > 0.5 for value, _ in progress_values))

    def test_run_service_with_progress_honors_custom_phase_ranges(self):
        """Runtime event mapping should honor custom phase boundaries."""
        host = _Host()
        progress_values = []

        def _progress(value, desc=None):
            """Capture forwarded progress updates."""
            progress_values.append((value, desc))

        host._run_generate_music_service_with_progress(
            progress=_progress,
            actual_batch_size=1,
            audio_duration=10.0,
            inference_steps=8,
            timesteps=None,
            service_inputs={
                "captions_batch": ["c"],
                "lyrics_batch": ["l"],
                "metas_batch": ["m"],
                "vocal_languages_batch": ["en"],
                "target_wavs_tensor": None,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "instructions_batch": ["i"],
                "audio_code_hints_batch": None,
                "should_return_intermediate": True,
            },
            refer_audios=None,
            guidance_scale=7.0,
            actual_seed_list=[1],
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            use_adg=False,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            infer_method="ode",
            phase_ranges={"service_start": 0.20, "encoding_end": 0.34, "diffusion_end": 0.70},
        )

        self.assertTrue(progress_values)
        values_only = [value for value, _ in progress_values]
        self.assertTrue(any(abs(value - 0.27) < 0.05 for value in values_only))
        self.assertTrue(any(abs(value - 0.52) < 0.08 for value in values_only))

class _ServiceHost(ServiceGenerateExecuteMixin):
    """Minimal host for service processed-data unpacking tests."""


class ServiceGenerateExecuteMixinTests(unittest.TestCase):
    """Verify processed batch unpacking for service generation."""

    def test_unpack_service_processed_data_accepts_repaint_mask(self):
        """It unpacks the repaint mask added by batch preprocessing."""
        host = _ServiceHost()
        processed_data = (
            ["k1"],
            ["text"],
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4),
            torch.tensor([0], dtype=torch.long),
            torch.ones(1, 4, 4),
            [("full", 0, 4)],
            torch.tensor([True]),
            None,
            torch.ones(1, 2, dtype=torch.long),
            None,
            None,
            None,
            torch.tensor([[True, False, True, False]], dtype=torch.bool),
        )

        payload = host._unpack_service_processed_data(processed_data)

        self.assertIn("repaint_mask", payload)
        self.assertTrue(torch.equal(payload["repaint_mask"], processed_data[-1]))


if __name__ == "__main__":
    unittest.main()
