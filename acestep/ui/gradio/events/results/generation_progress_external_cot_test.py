"""Tests for external-CoT remapping behavior in generation_progress."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.text_tasks.external_lm_tasks import ExternalAIClientError
from acestep.ui.gradio.events.results import generation_progress


def _build_call_kwargs() -> dict:
    """Build default kwargs for ``generate_with_progress`` test invocations."""
    defaults = {
        "captions": "input caption",
        "lyrics": "",
        "bpm": None,
        "key_scale": "",
        "time_signature": "",
        "vocal_language": "unknown",
        "inference_steps": 8,
        "guidance_scale": 7.0,
        "random_seed_checkbox": True,
        "seed": "-1",
        "reference_audio": None,
        "audio_duration": -1.0,
        "batch_size_input": 1,
        "src_audio": None,
        "text2music_audio_code_string": "",
        "repainting_start": 0.0,
        "repainting_end": -1.0,
        "instruction_display_gen": "",
        "audio_cover_strength": 1.0,
        "cover_noise_strength": 0.0,
        "task_type": "text2music",
        "use_adg": False,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "shift": 1.0,
        "infer_method": "ode",
        "custom_timesteps": "",
        "audio_format": "flac",
        "lm_temperature": 0.85,
        "think_checkbox": True,
        "lm_cfg_scale": 2.0,
        "lm_top_k": 0,
        "lm_top_p": 0.9,
        "lm_negative_prompt": "NO USER INPUT",
        "use_cot_metas": True,
        "use_cot_caption": True,
        "use_cot_language": True,
        "is_format_caption": False,
        "constrained_decoding_debug": False,
        "allow_lm_batch": False,
        "auto_score": False,
        "auto_lrc": False,
        "score_scale": 1.0,
        "lm_batch_chunk_size": 8,
        "enable_normalization": True,
        "normalization_db": -1.0,
        "fade_in_duration": 0.0,
        "fade_out_duration": 0.0,
        "latent_shift": 0.0,
        "latent_rescale": 1.0,
    }

    kwargs = {}
    for name in list(inspect.signature(generation_progress.generate_with_progress).parameters)[2:]:
        if name == "progress":
            continue
        kwargs[name] = defaults[name]
    return kwargs


class GenerationProgressExternalCotTests(unittest.TestCase):
    """Validate external CoT bridge behavior in generate_with_progress."""

    def _run_once(self, llm_initialized: bool, **overrides):
        """Run generator once and return (outputs, generate_music_call_kwargs)."""
        kwargs = _build_call_kwargs()
        kwargs.update(overrides)

        llm_handler = SimpleNamespace(llm_initialized=llm_initialized)
        gpu_config = SimpleNamespace(
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=4,
        )
        failed_result = SimpleNamespace(
            success=False,
            status_message="Generation failed",
            audios=[],
            extra_outputs={},
        )

        with patch.object(generation_progress, "get_global_gpu_config", return_value=gpu_config), patch.object(
            generation_progress,
            "check_duration_limit",
            return_value=(True, ""),
        ), patch.object(
            generation_progress,
            "check_batch_size_limit",
            return_value=(True, ""),
        ), patch.object(
            generation_progress,
            "parse_and_validate_timesteps",
            return_value=(None, False, ""),
        ), patch.object(
            generation_progress,
            "generate_music",
            return_value=failed_result,
        ) as generate_music_mock:
            outputs = list(generation_progress.generate_with_progress(None, llm_handler, **kwargs))
        return outputs, generate_music_mock.call_args.kwargs

    @patch.object(generation_progress, "is_external_lm_active", return_value=True)
    @patch.object(generation_progress, "format_sample_with_external_provider")
    def test_external_cot_remaps_think_and_cot_flags_when_local_lm_missing(
        self,
        external_format_mock,
        _external_active_mock,
    ):
        """External mode should precompute text metadata and disable local-LM-only flags."""
        external_format_mock.return_value = SimpleNamespace(
            success=True,
            caption="external caption",
            lyrics="external lyrics",
            bpm=110,
            duration=25.0,
            keyscale="D minor",
            language="en",
            timesignature="4/4",
            status_message="External AI format completed (glm-4.5-flash)",
        )

        outputs, call_kwargs = self._run_once(llm_initialized=False)
        params = call_kwargs["params"]

        self.assertFalse(params.thinking)
        self.assertFalse(params.use_cot_metas)
        self.assertFalse(params.use_cot_caption)
        self.assertFalse(params.use_cot_language)
        self.assertEqual(
            "Lead vocals stay present from the opening section onward. Core instrumentation is established from the opening section and stays central throughout. external caption",
            params.caption,
        )
        self.assertEqual(110, params.bpm)
        self.assertEqual("D minor", params.keyscale)
        self.assertEqual("en", params.vocal_language)
        self.assertEqual(1, len(outputs))
        self.assertIn("External AI", outputs[0][10])

    @patch.object(generation_progress, "is_external_lm_active", return_value=True)
    @patch.object(generation_progress, "format_sample_with_external_provider")
    def test_local_lm_initialized_skips_external_cot_bridge(
        self,
        external_format_mock,
        _external_active_mock,
    ):
        """When local 5Hz LM exists, generate_with_progress should keep native think/cot path."""
        outputs, call_kwargs = self._run_once(llm_initialized=True)
        params = call_kwargs["params"]
        self.assertTrue(params.thinking)
        self.assertTrue(params.use_cot_metas)
        self.assertTrue(params.use_cot_caption)
        self.assertTrue(params.use_cot_language)
        external_format_mock.assert_not_called()
        self.assertEqual(1, len(outputs))

    @patch.object(generation_progress, "is_external_lm_active", return_value=True)
    @patch.object(
        generation_progress,
        "format_sample_with_external_provider",
        side_effect=ExternalAIClientError("temporary external error"),
    )
    def test_external_cot_error_disables_local_lm_flags_and_continues(
        self,
        _external_format_mock,
        _external_active_mock,
    ):
        """External CoT failures should warn and continue with think/cot flags disabled."""
        outputs, call_kwargs = self._run_once(llm_initialized=False)
        params = call_kwargs["params"]
        self.assertFalse(params.thinking)
        self.assertFalse(params.use_cot_metas)
        self.assertFalse(params.use_cot_caption)
        self.assertFalse(params.use_cot_language)
        self.assertEqual(1, len(outputs))
        self.assertIn("External AI CoT warning", outputs[0][10])


if __name__ == "__main__":
    unittest.main()
