"""Unit tests for generation input event handlers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

try:
    from acestep.ui.gradio.events import generation_handlers
    from acestep.ui.gradio.i18n import t as _t
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependency guard
    generation_handlers = None
    _t = None
    _IMPORT_ERROR = exc


class _FakeDitHandler:
    """Minimal DiT handler stub for analyze-src-audio tests."""

    def __init__(self, convert_result):
        """Store a configurable conversion return value for test scenarios."""
        self._convert_result = convert_result
        self.model = MagicMock()  # Required by analyze_src_audio guard

    def convert_src_audio_to_codes(self, _src_audio):
        """Return configured conversion output."""
        return self._convert_result


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class GenerationHandlersTests(unittest.TestCase):
    """Tests for source-audio analysis validation behavior."""

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_rejects_non_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Reject conversion output that has no serialized audio-code tokens."""
        dit_handler = _FakeDitHandler("ERROR: not an audio file")
        llm_handler = SimpleNamespace(llm_initialized=True)

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="fake.mp3",
            constrained_decoding_debug=False,
        )

        # When codes_string has no audio-code tokens, the function returns
        # (codes_string, warning_message, "", "", None, None, "", "", "", False)
        from acestep.ui.gradio.i18n import t
        self.assertEqual(
            result,
            ("ERROR: not an audio file", t("messages.no_audio_codes_generated"),
             "", "", None, None, "", "", "", False),
        )
        understand_music_mock.assert_not_called()
        warning_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_allows_valid_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Pass valid audio codes through to LM understanding."""
        dit_handler = _FakeDitHandler("<|audio_code_123|><|audio_code_456|>")
        llm_handler = SimpleNamespace(llm_initialized=True)
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="ok",
            caption="caption",
            lyrics="lyrics",
            bpm=100,
            duration=10.0,
            keyscale="D minor",
            language="en",
            timesignature="4",
        )

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="fake.mp3",
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "<|audio_code_123|><|audio_code_456|>")
        self.assertEqual(result[1], "ok")
        self.assertEqual(result[2], "caption")
        self.assertEqual(result[3], "lyrics")
        self.assertEqual(result[4], 100)
        self.assertEqual(result[5], 10.0)
        self.assertEqual(result[6], "D minor")
        self.assertEqual(result[7], "en")
        self.assertEqual(result[8], "4")
        self.assertTrue(result[9])
        understand_music_mock.assert_called_once()
        warning_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    def test_handle_format_lyrics_preserves_existing_section_directives(
        self,
        format_sample_mock,
        info_mock,
    ):
        """Lyrics with existing section tags should be preserved as-is."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Verse 1]\nalready structured",
            bpm=100,
            duration=10.0,
            keyscale="D minor",
            language="en",
            timesignature="4",
            status_message="formatted",
        )

        result = generation_handlers.handle_format_lyrics(
            llm_handler=llm_handler,
            caption="caption",
            lyrics="lyrics",
            bpm=100,
            audio_duration=10.0,
            key_scale="D minor",
            time_signature="4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "[Verse 1]\nalready structured")
        self.assertEqual(result[-1], "formatted")
        format_sample_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    def test_handle_format_sample_preserves_output_contract(self, format_sample_mock, info_mock):
        """Full sample formatting should return 9 outputs with language at index 5."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="caption out",
            lyrics="lyrics out",
            bpm=112,
            duration=12.0,
            keyscale="G major",
            language="en",
            timesignature="3/4",
            status_message="formatted",
        )

        result = generation_handlers.handle_format_sample(
            llm_handler=llm_handler,
            caption="caption in",
            lyrics="lyrics in",
            bpm=112,
            audio_duration=12.0,
            key_scale="G major",
            time_signature="3/4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(len(result), 9)
        self.assertEqual(result[5], "en")
        self.assertEqual(result[8], "formatted")
        format_sample_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.llm_format_actions.format_sample_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active")
    def test_handle_format_sample_uses_external_provider_when_active(
        self,
        external_active_mock,
        format_sample_mock,
        external_format_mock,
        info_mock,
    ):
        """Format sample should use external provider when external mode is active."""
        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_format_mock.return_value = SimpleNamespace(
            success=True,
            caption="caption out",
            lyrics="lyrics out",
            bpm=112,
            duration=12.0,
            keyscale="G major",
            language="en",
            timesignature="3/4",
            status_message="external formatted",
        )

        result = generation_handlers.handle_format_sample(
            llm_handler=llm_handler,
            caption="caption in",
            lyrics="lyrics in",
            bpm=112,
            audio_duration=12.0,
            key_scale="G major",
            time_signature="3/4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(len(result), 9)
        self.assertEqual(result[5], "en")
        self.assertEqual(result[8], "external formatted")
        format_sample_mock.assert_not_called()
        external_format_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Warning")
    @patch(
        "acestep.ui.gradio.events.generation.llm_format_actions.format_sample_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active")
    def test_handle_format_caption_external_credential_error_returns_status(
        self,
        external_active_mock,
        external_format_mock,
        warning_mock,
    ):
        """External provider credential errors should be returned as UI status, not exceptions."""
        from acestep.text_tasks.external_lm_tasks import ExternalAIClientError

        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_format_mock.side_effect = ExternalAIClientError("Missing External AI credentials")

        result = generation_handlers.handle_format_caption(
            llm_handler=llm_handler,
            caption="caption",
            lyrics="lyrics",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[-1], "Missing External AI credentials")
        warning_mock.assert_called_once_with("Missing External AI credentials")

    @patch("acestep.ui.gradio.events.generation.llm_sample_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.llm_sample_actions.create_sample_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_sample_actions.is_external_lm_active")
    def test_handle_create_sample_external_strips_wrapped_caption_quotes(
        self,
        external_active_mock,
        external_create_mock,
        info_mock,
    ):
        """External sample creation should not leave wrapped quote characters in caption UI text."""
        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_create_mock.return_value = SimpleNamespace(
            success=True,
            caption='"future garage over rain-soaked streets"',
            lyrics="[Instrumental]",
            bpm=120,
            duration=30.0,
            keyscale="C Major",
            language="English",
            timesignature="4/4",
            instrumental=True,
            status_message="External Ollama sample created (smutmonger:latest)",
        )

        result = generation_handlers.handle_create_sample(
            llm_handler=llm_handler,
            query="future garage",
            instrumental=True,
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual("future garage over rain-soaked streets", result[0])
        self.assertEqual("[Instrumental]", result[1])
        self.assertIn("External Ollama sample created", result[13])
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    def test_handle_format_caption_adds_global_vocal_presence_when_missing(
        self,
        format_sample_mock,
        info_mock,
    ):
        """Formatted vocal captions should gain an early vocal-presence sentence when missing."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption=(
                "A neon pop track grows into a midnight chorus. "
                "It later reveals a soulful female vocalist with dynamic delivery."
            ),
            lyrics="[Verse 1]\nMeet me in the static light",
            bpm=118,
            duration=30.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="formatted",
        )

        result = generation_handlers.handle_format_caption(
            llm_handler=llm_handler,
            caption="draft caption",
            lyrics="draft lyrics",
            bpm=118,
            audio_duration=30.0,
            key_scale="A minor",
            time_signature="4/4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertTrue(
            result[0].startswith(
                "Lead vocals stay present from the opening section onward. The lead singer is a soulful and dynamic female vocalist. Core instrumentation is established from the opening section and stays central throughout."
            )
        )
        self.assertEqual("formatted", result[-1])
        format_sample_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Warning")
    @patch(
        "acestep.ui.gradio.events.generation.llm_format_actions.generate_lyrics_from_caption_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active")
    def test_handle_generate_lyrics_from_caption_external_timeout_uses_fallback(
        self,
        external_active_mock,
        create_sample_mock,
        external_generate_mock,
        warning_mock,
    ):
        """External timeout should return fallback lyrics instead of raising callback errors."""
        from acestep.text_tasks.external_lm_tasks import ExternalAIClientError

        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_generate_mock.side_effect = ExternalAIClientError("External AI request timed out after 60s.")

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="anthemic electro-pop",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4/4",
            vocal_language="ja",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertIn("[Verse 1]", result[0])
        self.assertIn("[Chorus]", result[0])
        self.assertNotIn("I carry the pulse of", result[0])
        self.assertNotIn("Deliver these lyrics", result[0])
        self.assertIn("LM timeout", result[-1])
        create_sample_mock.assert_not_called()
        external_generate_mock.assert_called_once()
        warning_mock.assert_called_once_with("External AI request timed out after 60s.")

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.get_i18n")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.build_duration_aware_fallback_lyrics")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions._execute_generate_lyrics")
    def test_handle_generate_lyrics_from_caption_uses_resolved_language_for_final_fallback(
        self,
        execute_generate_mock,
        fallback_mock,
        get_i18n_mock,
    ):
        """Final scaffold fallback should keep the resolved vocal language, not raw `unknown`."""
        get_i18n_mock.return_value = type("I18n", (), {"current_language": "ja"})()
        execute_generate_mock.side_effect = [
            (type("Result", (), {"lyrics": "[Verse 1]", "bpm": 120, "keyscale": "C major", "language": "", "timesignature": "4/4"})(), 30.0, "first"),
            (type("Result", (), {"lyrics": "[Verse 1]", "bpm": 120, "keyscale": "C major", "language": "", "timesignature": "4/4"})(), 30.0, "retry"),
        ]
        fallback_mock.return_value = "fallback lyrics"

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=SimpleNamespace(llm_initialized=True),
            caption="anthemic electro-pop",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4/4",
            vocal_language="unknown",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual("fallback lyrics", result[0])
        self.assertEqual("ja", fallback_mock.call_args.kwargs["vocal_language"])

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.llm_format_actions.generate_lyrics_from_caption_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample_with_external_provider")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active")
    def test_handle_generate_lyrics_from_caption_uses_dedicated_external_path(
        self,
        external_active_mock,
        external_format_mock,
        external_generate_mock,
        info_mock,
    ):
        """External lyric generation should bypass the generic format adapter path."""
        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_generate_mock.return_value = SimpleNamespace(
            success=True,
            caption="arcade rush",
            lyrics="pixel hearts in the midnight glow\ncoins of light keep falling slow",
            bpm=142,
            duration=26.0,
            keyscale="E minor",
            language="en",
            timesignature="4/4",
            status_message="External Z.ai lyrics generated (glm-5)",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="energetic chiptune track",
            bpm=142,
            audio_duration=26.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\npixel hearts in the midnight glow\ncoins of light keep falling slow",
            result[0],
        )
        self.assertIn("External Z.ai lyrics generated", result[-1])
        external_generate_mock.assert_called_once_with(
            caption="energetic chiptune track",
            bpm=142,
            audio_duration=26.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
            retry=False,
        )
        external_format_mock.assert_not_called()
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.llm_format_actions.generate_lyrics_from_caption_with_external_provider"
    )
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active")
    def test_handle_generate_lyrics_from_caption_retries_external_invalid_output(
        self,
        external_active_mock,
        external_generate_mock,
        info_mock,
    ):
        """External lyric generation should retry with retry=True before falling back."""
        llm_handler = SimpleNamespace(llm_initialized=False)
        external_active_mock.return_value = True
        external_generate_mock.side_effect = [
            SimpleNamespace(
                success=True,
                caption="arcade rush",
                lyrics="[Vocalise]",
                bpm=142,
                duration=26.0,
                keyscale="E minor",
                language="en",
                timesignature="4/4",
                status_message="External first pass",
            ),
            SimpleNamespace(
                success=True,
                caption="arcade rush",
                lyrics="pixel hearts in the midnight glow\ncoins of light keep falling slow",
                bpm=142,
                duration=26.0,
                keyscale="E minor",
                language="en",
                timesignature="4/4",
                status_message="External retry pass",
            ),
        ]

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="energetic chiptune track",
            bpm=142,
            audio_duration=26.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\npixel hearts in the midnight glow\ncoins of light keep falling slow",
            result[0],
        )
        self.assertEqual(2, external_generate_mock.call_count)
        self.assertFalse(external_generate_mock.call_args_list[0].kwargs["retry"])
        self.assertTrue(external_generate_mock.call_args_list[1].kwargs["retry"])
        self.assertIn("External retry pass", result[-1])
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_applies_vocal_language(
        self,
        create_sample_mock,
        info_mock,
    ):
        """Generate-lyrics action should use the create-sample flow with language guidance."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="\u6771\u4eac\u306e\u5149\u3092\u8ffd\u3044\u304b\u3051\u3066\n\u591c\u3092\u8d8a\u3048\u3066\u541b\u3068\u6b4c\u3046",
            bpm=100,
            duration=20.0,
            keyscale="D minor",
            language="ja",
            timesignature="4/4",
            status_message="created",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="dreamy city-pop at night",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="ja",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\n\u6771\u4eac\u306e\u5149\u3092\u8ffd\u3044\u304b\u3051\u3066\n\u591c\u3092\u8d8a\u3048\u3066\u541b\u3068\u6b4c\u3046",
            result[0],
        )
        self.assertIn("Lyrics generated from caption.", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        called_kwargs = create_sample_mock.call_args.kwargs
        self.assertEqual("ja", called_kwargs["vocal_language"])
        self.assertFalse(called_kwargs["instrumental"])
        self.assertIn("This is a lyric-writing task for a vocal song with lead vocals.", called_kwargs["query"])
        self.assertIn("Do not return [Instrumental]", called_kwargs["query"])
        self.assertIn("20 syllables or fewer", called_kwargs["query"])
        self.assertIn("break it into a new lyric line", called_kwargs["query"])
        self.assertIn("Preferred vocal language: ja", called_kwargs["query"])
        self.assertIn("tempo=96 bpm", called_kwargs["query"])
        self.assertIn("duration=24 seconds", called_kwargs["query"])
        self.assertIn("Preferred key: A minor", called_kwargs["query"])
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.get_i18n")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_uses_ui_language_when_input_is_instrumental(
        self,
        create_sample_mock,
        info_mock,
        get_i18n_mock,
    ):
        """Lyric generation should fall back to the current UI language for bad instrumental defaults."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        get_i18n_mock.return_value = SimpleNamespace(current_language="ja")
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="city lights keep pulling me near\nwe sing through the midnight air",
            bpm=100,
            duration=20.0,
            keyscale="D minor",
            language="",
            timesignature="4/4",
            status_message="created",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="dreamy city-pop at night",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="instrumental",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        called_kwargs = create_sample_mock.call_args.kwargs
        self.assertEqual("ja", called_kwargs["vocal_language"])
        self.assertIn("Preferred vocal language: ja", called_kwargs["query"])
        self.assertEqual("ja", result[4])
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.get_i18n")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_falls_back_to_english_for_unknown_ui_language(
        self,
        create_sample_mock,
        info_mock,
        get_i18n_mock,
    ):
        """Lyric generation should fall back to English when the UI language is unsupported."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        get_i18n_mock.return_value = SimpleNamespace(current_language="xx")
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="follow the lights into the dawn",
            bpm=100,
            duration=20.0,
            keyscale="D minor",
            language="",
            timesignature="4/4",
            status_message="created",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="dreamy city-pop at night",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="instrumental",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        called_kwargs = create_sample_mock.call_args.kwargs
        self.assertEqual("en", called_kwargs["vocal_language"])
        self.assertIn("Preferred vocal language: en", called_kwargs["query"])
        self.assertEqual("en", result[4])
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_overrides_instrumental_caption(
        self,
        create_sample_mock,
        info_mock,
    ):
        """Instrumental-looking captions should still build a forced-vocal lyric query."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="we turn the ivory keys into a voice tonight",
            bpm=190,
            duration=160.0,
            keyscale="E minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )

        generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="A contemplative and virtuosic solo piano piece with a neoclassical and video game soundtrack feel.",
            bpm=190,
            audio_duration=160.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="unknown",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        query = create_sample_mock.call_args.kwargs["query"]
        self.assertIn("Reinterpret it as a vocal version", query)
        self.assertIn("Do not return [Instrumental]", query)
        self.assertIn("lead vocals", query)
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_retries_on_tag_only_output(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Generate-lyrics action should retry via format flow when first result is tag-only."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Vocalise]",
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="we rise with the tide\nwe glow through the night",
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="synth-pop lift",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\nwe rise with the tide\nwe glow through the night",
            result[0],
        )
        self.assertIn("formatted retry", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        retry_caption = format_sample_mock.call_args.kwargs["caption"]
        retry_seed = format_sample_mock.call_args.kwargs["lyrics"]
        self.assertIn("Convert this concept into a vocal version", retry_caption)
        self.assertIn("Do not return [Instrumental]", retry_caption)
        self.assertIn("different hook and imagery", retry_seed)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_retries_on_instruction_leakage(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Prompt-leakage text should trigger format-flow retry before accepting lyrics."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Intro]\n"
                "[Spoken word sample: Must include clear sung words with lead vocals in en.]"
            ),
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="we rise with the tide\nwe glow through the night",
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="synth-pop lift",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\nwe rise with the tide\nwe glow through the night",
            result[0],
        )
        self.assertIn("formatted retry", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_retries_on_placeholder_scaffold(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Placeholder scaffold outputs should be rejected and retried via format flow."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Verse 1]\n"
                "(2-4 sung lines in the requested language; start the narrative)\n\n"
                "[Chorus]\n"
                "(2-4 sung lines; memorable hook)\n\n"
                "[Verse 2]\n"
                "(2-4 sung lines; develop the narrative)"
            ),
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="we rise with the tide\nwe glow through the night",
            bpm=96,
            duration=24.0,
            keyscale="A minor",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="synth-pop lift",
            bpm=96,
            audio_duration=24.0,
            key_scale="A minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\nwe rise with the tide\nwe glow through the night",
            result[0],
        )
        self.assertIn("formatted retry", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_retry_prompt_overrides_instrumental_caption(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Retry prompt should explicitly override instrumental-only captions."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Instrumental]",
            bpm=40,
            duration=184.0,
            keyscale="E minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="we light the room and start the fire tonight",
            bpm=40,
            duration=184.0,
            keyscale="E minor",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="An upbeat instrumental funk track with strong acid jazz influences.",
            bpm=40,
            audio_duration=184.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        retry_caption = format_sample_mock.call_args.kwargs["caption"]
        self.assertIn("Convert this concept into a vocal version", retry_caption)
        self.assertIn("Override that and write a vocal topline", retry_caption)
        self.assertIn("Do not return [Instrumental]", retry_caption)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_extracts_spoken_word_sample_content(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Retry output should keep spoken-word content instead of falling back."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Verse 1 - Instrumental]",
            bpm=90,
            duration=141.0,
            keyscale="F# major",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Intro]\n"
                "[Spoken word sample: I'm on the beat, I'm on the beat, you know, can't be]\n\n"
                "[Beat Drop]\n\n"
                "[Outro]"
            ),
            bpm=90,
            duration=141.0,
            keyscale="F# major",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="instrumental hip-hop concept",
            bpm=90,
            audio_duration=141.0,
            key_scale="F# major",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Intro]\nI'm on the beat, I'm on the beat, you know, can't be\n\n[Outro]",
            result[0],
        )
        self.assertNotIn("fallback scaffold", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_strips_placeholder_lines_before_accepting(
        self,
        create_sample_mock,
        info_mock,
    ):
        """Valid lyrics should survive even when one scaffold line leaks into the output."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Verse 1]\n"
                "(2-4 sung lines; memorable hook)\n"
                "pixel hearts race through the midnight glow\n\n"
                "[Chorus]\n"
                "coins of light keep falling slow"
            ),
            bpm=142,
            duration=26.0,
            keyscale="E minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="energetic chiptune track",
            bpm=142,
            audio_duration=26.0,
            key_scale="E minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(
            "[Verse 1]\npixel hearts race through the midnight glow\n\n[Chorus]\ncoins of light keep falling slow",
            result[0],
        )
        self.assertNotIn("fallback scaffold", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        info_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_retries_on_inconsistent_verse_lengths(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Repeated verse sections with different line counts should trigger retry."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Verse 1]\n"
                "Every small heartbeat keeps pulling me west\n"
                "All of my doubt starts to loosen and fall\n"
                "Streetlight reflections are writing our names\n\n"
                "[Chorus]\n"
                "Sing it till the skyline opens wide\n"
                "Hold me in the light, we can outrun the night\n"
                "When the kick drum lands, our shadows come alive\n"
                "We keep the fire bright until the morning tide\n\n"
                "[Verse 2]\n"
                "Rain on the windows keeps time with the snare\n"
                "One final chorus and then we let go\n"
                "Leaving a trail of electric glow\n"
                "Breath on the downbeat, we lean into sound"
            ),
            bpm=118,
            duration=34.0,
            keyscale="D minor",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics=(
                "[Verse 1]\n"
                "Every small heartbeat keeps pulling me west\n"
                "All of my doubt starts to loosen and fall\n"
                "Streetlight reflections are writing our names\n\n"
                "[Chorus]\n"
                "Sing it till the skyline opens wide\n"
                "Hold me in the light, we can outrun the night\n"
                "When the kick drum lands, our shadows come alive\n"
                "We keep the fire bright until the morning tide\n\n"
                "[Verse 2]\n"
                "Rain on the windows keeps time with the snare\n"
                "One final chorus and then we let go\n"
                "Leaving a trail of electric glow"
            ),
            bpm=118,
            duration=34.0,
            keyscale="D minor",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="neon city pop",
            bpm=118,
            audio_duration=34.0,
            key_scale="D minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertIn("[Verse 2]", result[0])
        self.assertIn("formatted retry", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        retry_caption = format_sample_mock.call_args.kwargs["caption"]
        self.assertIn("same number of lines", retry_caption)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_uses_fallback_when_retry_is_still_tag_only(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Generate-lyrics action should produce scaffold lyrics after failed format retry."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Instrumental]",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="[Vocalise]",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="anthemic electro-pop",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertIn("[Verse 1]", result[0])
        self.assertIn("[Chorus]", result[0])
        self.assertNotEqual("[Instrumental]", result[0].strip())
        self.assertIn("fallback scaffold", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.create_sample")
    def test_handle_generate_lyrics_from_caption_falls_back_for_plain_instrumental_text(
        self,
        create_sample_mock,
        format_sample_mock,
        info_mock,
    ):
        """Plain instrumental outputs should trigger fallback after format retry also fails."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="instrumental",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4/4",
            status_message="created",
        )
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="<lyrics>[Vocalise]</lyrics>",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4/4",
            status_message="formatted retry",
        )

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="anthemic electro-pop",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.9,
            lm_top_k=10,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertIn("[Verse 1]", result[0])
        self.assertIn("[Chorus]", result[0])
        self.assertIn("fallback scaffold", result[-1])
        self.assertEqual(1, create_sample_mock.call_count)
        self.assertEqual(1, format_sample_mock.call_count)
        info_mock.assert_called()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.is_external_lm_active", return_value=False)
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Warning")
    def test_handle_generate_lyrics_from_caption_requires_lm(self, warning_mock, _external_active_mock):
        """Generate-lyrics action should show LM-not-initialized when no LM/external runtime."""
        llm_handler = SimpleNamespace(llm_initialized=False)

        result = generation_handlers.handle_generate_lyrics_from_caption(
            llm_handler=llm_handler,
            caption="caption",
            bpm=100,
            audio_duration=20.0,
            key_scale="D minor",
            time_signature="4/4",
            vocal_language="en",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(_t("messages.lm_not_initialized"), result[-1])
        warning_mock.assert_called_once_with(_t("messages.lm_not_initialized"))


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class LoadMetadataLmCodesTests(unittest.TestCase):
    """Tests that load_metadata sets think=False when audio_codes are present."""

    def _write_json(self, tmpdir, data):
        """Write a JSON file and return a SimpleNamespace with .name pointing to it."""
        import json, os
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return SimpleNamespace(name=path)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.get_global_gpu_config")
    def test_think_set_false_when_audio_codes_present(self, gpu_mock, info_mock):
        """When JSON has thinking=True AND non-empty audio_codes, think should be False."""
        import tempfile
        gpu_cfg = MagicMock()
        gpu_cfg.max_batch_size_with_lm = 8
        gpu_cfg.max_batch_size_without_lm = 8
        gpu_mock.return_value = gpu_cfg

        llm_handler = SimpleNamespace(llm_initialized=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "thinking": True,
                "audio_codes": "<|audio_code_1|><|audio_code_2|>",
            })
            result = generation_handlers.load_metadata(file_obj, llm_handler)

        # think is at return position 30 (0-indexed)
        think_value = result[30]
        audio_codes_value = result[31]
        self.assertFalse(think_value, "think should be False when audio_codes present")
        self.assertEqual(audio_codes_value, "<|audio_code_1|><|audio_code_2|>")

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.get_global_gpu_config")
    def test_think_unchanged_when_audio_codes_empty(self, gpu_mock, info_mock):
        """When JSON has thinking=True AND empty audio_codes, think stays True."""
        import tempfile
        gpu_cfg = MagicMock()
        gpu_cfg.max_batch_size_with_lm = 8
        gpu_cfg.max_batch_size_without_lm = 8
        gpu_mock.return_value = gpu_cfg

        llm_handler = SimpleNamespace(llm_initialized=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "thinking": True,
                "audio_codes": "",
            })
            result = generation_handlers.load_metadata(file_obj, llm_handler)

        think_value = result[30]
        self.assertTrue(think_value, "think should remain True when audio_codes is empty")

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.is_external_lm_active", return_value=True)
    @patch("acestep.ui.gradio.events.generation.metadata_loading.get_global_gpu_config")
    def test_load_metadata_keeps_think_true_when_external_lm_active(
        self,
        gpu_mock,
        _external_active_mock,
        info_mock,
        warning_mock,
    ):
        """External LM active should satisfy think gate even when local LM is not initialized."""
        import tempfile
        gpu_cfg = MagicMock()
        gpu_cfg.max_batch_size_with_lm = 8
        gpu_cfg.max_batch_size_without_lm = 8
        gpu_mock.return_value = gpu_cfg

        llm_handler = SimpleNamespace(llm_initialized=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "thinking": True,
                "audio_codes": "",
            })
            result = generation_handlers.load_metadata(file_obj, llm_handler)

        self.assertTrue(result[30], "think should remain True when external LM mode is active")
        warning_mock.assert_not_called()
        info_mock.assert_called_once()


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class LoadRandomExampleExternalLmTests(unittest.TestCase):
    """Tests that random example loading respects external LM mode for think gating."""

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.choose_random_example_file")
    @patch("acestep.ui.gradio.events.generation.metadata_loading._get_project_root")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.is_external_lm_active", return_value=True)
    def test_load_random_example_keeps_think_true_when_external_lm_active(
        self,
        _external_active_mock,
        get_project_root_mock,
        choose_random_example_file_mock,
        info_mock,
        warning_mock,
    ):
        """Example loader should not warn or disable think when external LM mode is active."""
        import json
        import os
        import tempfile

        llm_handler = SimpleNamespace(llm_initialized=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            examples_dir = os.path.join(tmpdir, "examples", "text2music")
            os.makedirs(examples_dir, exist_ok=True)
            sample_file = os.path.join(examples_dir, "sample.json")
            with open(sample_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "caption": "ambient pad swell",
                        "lyrics": "[Instrumental]",
                        "think": True,
                        "bpm": 92,
                    },
                    handle,
                )

            get_project_root_mock.return_value = tmpdir
            choose_random_example_file_mock.return_value = sample_file
            result = generation_handlers.load_random_example("text2music", llm_handler)

        self.assertTrue(result[2], "think should remain enabled under external LM mode")
        warning_mock.assert_not_called()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.metadata_loading.logger.exception")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.load_random_example")
    @patch("acestep.ui.gradio.events.generation.metadata_loading.understand_music")
    def test_sample_example_smart_logs_exception_before_fallback(
        self,
        understand_music_mock,
        load_random_example_mock,
        warning_mock,
        logger_exception_mock,
    ):
        """LM sample failures should be logged before falling back to static examples."""
        understand_music_mock.side_effect = RuntimeError("boom")
        load_random_example_mock.return_value = ("cap", "lyr", True, 120, 30, "C major", "en", "4/4")

        llm_handler = SimpleNamespace(llm_initialized=True)
        result = generation_handlers.sample_example_smart(llm_handler, "text2music")

        self.assertEqual(("cap", "lyr", True, 120, 30, "C major", "en", "4/4"), result)
        warning_mock.assert_called_once()
        logger_exception_mock.assert_called_once()


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class AutoCheckboxTests(unittest.TestCase):
    """Tests for optional-parameter Auto checkbox handler functions."""

    def test_on_auto_checkbox_change_checked_returns_default_and_non_interactive(self):
        """When Auto is checked, field should reset to default and become non-interactive."""
        result = generation_handlers.on_auto_checkbox_change(True, "bpm")
        # gr.update returns a dict-like object; check value and interactive
        self.assertIsNone(result["value"])
        self.assertFalse(result["interactive"])

    def test_on_auto_checkbox_change_unchecked_returns_interactive(self):
        """When Auto is unchecked, field should become interactive (no value reset)."""
        result = generation_handlers.on_auto_checkbox_change(False, "bpm")
        self.assertTrue(result["interactive"])

    def test_on_auto_checkbox_change_all_fields(self):
        """All supported field names should produce valid defaults when checked."""
        expected = {
            "bpm": None,
            "key_scale": "",
            "time_signature": "",
            "vocal_language": "unknown",
            "audio_duration": -1,
        }
        for field_name, expected_value in expected.items():
            result = generation_handlers.on_auto_checkbox_change(True, field_name)
            self.assertEqual(result["value"], expected_value, f"Field {field_name}")
            self.assertFalse(result["interactive"], f"Field {field_name}")

    def test_reset_all_auto_returns_correct_count(self):
        """reset_all_auto should return exactly 10 gr.update objects."""
        result = generation_handlers.reset_all_auto()
        self.assertEqual(len(result), 10)

    def test_reset_all_auto_checkboxes_are_true(self):
        """First 5 outputs (auto checkboxes) should all be set to True."""
        result = generation_handlers.reset_all_auto()
        for i in range(5):
            self.assertTrue(result[i]["value"], f"Auto checkbox at index {i}")

    def test_reset_all_auto_fields_are_defaults(self):
        """Last 5 outputs (fields) should be reset to auto defaults."""
        result = generation_handlers.reset_all_auto()
        self.assertIsNone(result[5]["value"])         # bpm
        self.assertEqual(result[6]["value"], "")       # key_scale
        self.assertEqual(result[7]["value"], "")       # time_signature
        self.assertEqual(result[8]["value"], "unknown") # vocal_language
        self.assertEqual(result[9]["value"], -1)       # audio_duration

    def test_uncheck_auto_for_populated_fields_all_default(self):
        """When all fields have default values, all auto checkboxes should stay checked."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=None, key_scale="", time_signature="",
            vocal_language="unknown", audio_duration=-1,
        )
        self.assertEqual(len(result), 10)
        # Auto checkboxes should be True (checked)
        for i in range(5):
            self.assertTrue(result[i]["value"], f"Auto checkbox at index {i}")
        # Fields should be non-interactive
        for i in range(5, 10):
            self.assertFalse(result[i]["interactive"], f"Field at index {i}")

    def test_uncheck_auto_for_populated_fields_all_populated(self):
        """When all fields have non-default values, all auto checkboxes should be unchecked."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=120, key_scale="C major", time_signature="4",
            vocal_language="en", audio_duration=30.0,
        )
        # Auto checkboxes should be False (unchecked)
        for i in range(5):
            self.assertFalse(result[i]["value"], f"Auto checkbox at index {i}")
        # Fields should be interactive
        for i in range(5, 10):
            self.assertTrue(result[i]["interactive"], f"Field at index {i}")

    def test_uncheck_auto_for_populated_fields_mixed(self):
        """Mixed populated/default fields should only uncheck populated ones."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=120, key_scale="", time_signature="4",
            vocal_language="unknown", audio_duration=-1,
        )
        self.assertFalse(result[0]["value"])   # bpm_auto unchecked
        self.assertTrue(result[1]["value"])    # key_auto stays checked
        self.assertFalse(result[2]["value"])   # timesig_auto unchecked
        self.assertTrue(result[3]["value"])    # vocal_lang_auto stays checked
        self.assertTrue(result[4]["value"])    # duration_auto stays checked



    @patch("acestep.ui.gradio.events.generation.ui_helpers.get_i18n")
    def test_sync_vocal_language_after_lyrics_generation_uses_ui_language(self, get_i18n_mock):
        """Generated vocal lyrics should force the vocal-language UI out of auto."""
        get_i18n_mock.return_value = SimpleNamespace(current_language="ja")

        result = generation_handlers.sync_vocal_language_after_lyrics_generation(
            lyrics="[Verse 1]\nTokyo lights keep calling",
            vocal_language="unknown",
        )

        self.assertFalse(result[0]["value"])
        self.assertFalse(result[1]["value"])
        self.assertEqual("ja", result[2]["value"])
        self.assertTrue(result[2]["interactive"])

    def test_sync_vocal_language_after_lyrics_generation_preserves_manual_language(self):
        """Explicit user-selected vocal languages should not be overwritten."""
        result = generation_handlers.sync_vocal_language_after_lyrics_generation(
            lyrics="[Verse 1]\nFollow the night into dawn",
            vocal_language="es",
        )

        self.assertFalse(result[0]["value"])
        self.assertFalse(result[1]["value"])
        self.assertEqual("es", result[2]["value"])
        self.assertTrue(result[2]["interactive"])

    def test_sync_vocal_language_after_lyrics_generation_ignores_instrumental_marker(self):
        """Instrumental-only lyrics should not force a vocal-language override."""
        result = generation_handlers.sync_vocal_language_after_lyrics_generation(
            lyrics="[Instrumental]",
            vocal_language="unknown",
        )

        self.assertEqual("update", result[0]["__type__"])
        self.assertEqual("update", result[1]["__type__"])
        self.assertEqual("update", result[2]["__type__"])
        self.assertNotIn("value", result[0])
        self.assertNotIn("value", result[1])
        self.assertNotIn("value", result[2])

if __name__ == "__main__":
    unittest.main()



