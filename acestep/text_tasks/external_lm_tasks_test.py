"""Tests for minimal external LM formatting helpers."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib import request

import acestep.debug_utils

from acestep.text_tasks.external_ai_types import ExternalAIClientError
from acestep.text_tasks.external_lm_tasks import (
    _apply_user_metadata_overrides,
    _build_fallback_caption,
    _caption_needs_retry,
    _coerce_keep_alive_value,
    create_sample_with_external_provider,
    format_sample_with_external_provider,
    warm_up_external_provider,
)


class _FakeResponse:
    """Simple context-manager HTTP response stub."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        """Return response bytes."""

        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        """Return self for context manager support."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """No-op context manager exit."""

        return False


class _UrlopenRecorder:
    """HTTP stub that records requests before returning a fixed response."""

    def __init__(self, response_payload: dict):
        self.response_payload = response_payload
        self.last_request: request.Request | None = None

    def __call__(self, req, timeout=None):
        """Record the request and return a fake response."""

        self.last_request = req
        return _FakeResponse(self.response_payload)


class ExternalLmTasksTests(unittest.TestCase):
    """Verify external LM format requests parse into result objects."""

    def test_caption_needs_retry_for_unchanged_or_too_short_result(self) -> None:
        """Simple echoes and very short captions should trigger one retry."""

        self.assertTrue(
            _caption_needs_retry(
                original_caption="Salsa dura with brass section, call-and-response vocals",
                generated_caption="Salsa dura with brass section, call-and-response vocals",
            )
        )
        self.assertTrue(
            _caption_needs_retry(
                original_caption="Progressive trance instrumental",
                generated_caption="Progressive trance instrumental with pads",
            )
        )
        self.assertFalse(
            _caption_needs_retry(
                original_caption="Progressive trance instrumental",
                generated_caption="A progressive trance instrumental opens with evolving pads and arpeggiators, "
                "builds through a long breakdown, and resolves in a euphoric outro.",
            )
        )

    def test_apply_user_metadata_overrides_preserves_constrained_values(self) -> None:
        """User-supplied metadata should win over provider drift."""

        plan = SimpleNamespace(
            bpm=1,
            duration=2.4,
            key_scale="C minor",
            time_signature="3/4",
            vocal_language="English",
        )

        result = _apply_user_metadata_overrides(
            plan=plan,
            user_metadata={
                "bpm": 125,
                "duration": 240,
                "keyscale": "D major",
                "timesignature": "4/4",
            },
        )

        self.assertEqual(result.bpm, 125)
        self.assertEqual(result.duration, 240.0)
        self.assertEqual(result.key_scale, "D major")
        self.assertEqual(result.time_signature, "4/4")

    def test_build_fallback_caption_uses_prompt_and_metadata(self) -> None:
        """Fallback caption should expand the original prompt into a narrative with metadata."""

        caption = _build_fallback_caption(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            user_metadata={
                "bpm": 125,
                "duration": 240,
                "keyscale": "D major",
                "timesignature": "4/4",
            },
        )

        self.assertIn("Salsa dura with brass section", caption)
        self.assertIn("125 BPM", caption)
        self.assertIn("4/4", caption)
        self.assertIn("D major", caption)

    def test_coerce_keep_alive_value_preserves_numeric_sentinel(self) -> None:
        """Numeric keep-alive env values should become JSON numbers, not strings."""

        self.assertEqual(_coerce_keep_alive_value("-1"), -1)
        self.assertEqual(_coerce_keep_alive_value("300"), 300)
        self.assertEqual(_coerce_keep_alive_value("5m"), "5m")

    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_format_sample_with_external_provider_parses_openai_style_response(
        self,
        resolve_api_key_mock,
        urlopen_mock,
    ) -> None:
        """A valid external provider response should become a format result object."""

        resolve_api_key_mock.return_value = "test-key"
        urlopen_mock.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "A bright synth-pop groove with glossy hooks.",
                                    "lyrics": "City lights / we glow tonight",
                                    "bpm": 120,
                                    "duration": 30,
                                    "key_scale": "C Major",
                                    "time_signature": "4/4",
                                    "vocal_language": "en",
                                    "instrumental": False,
                                }
                            )
                        }
                    }
                ]
            }
        )

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "openai",
                "ACESTEP_EXTERNAL_LM_MODEL": "gpt-4o-mini",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "https://api.openai.com/v1/chat/completions",
            },
            clear=False,
        ):
            result = format_sample_with_external_provider(
                caption="synth pop",
                lyrics="",
                user_metadata={"bpm": 120},
            )

        self.assertTrue(result.success)
        self.assertEqual(result.caption, "A bright synth-pop groove with glossy hooks.")
        self.assertEqual(result.bpm, 120)
        self.assertEqual(result.keyscale, "C Major")

    @patch("acestep.debug_utils.debug_log_for")
    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_format_sample_with_external_provider_logs_prompt_and_response_in_debug_mode(
        self,
        resolve_api_key_mock,
        urlopen_mock,
        debug_log_mock,
    ) -> None:
        """Debug mode should log the outbound prompt payload and raw provider response."""

        resolve_api_key_mock.return_value = ""
        urlopen_mock.return_value = _FakeResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "caption": "A warm tropical funk band eases in with syncopated guitars.",
                            "lyrics": "[Instrumental]",
                            "instrumental": False,
                        }
                    ),
                }
            }
        )

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_LM_MODEL": "qwen3:4b",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=False,
        ):
            format_sample_with_external_provider(
                caption="Tropical funk",
                lyrics="",
                user_metadata={},
                debug=True,
            )

        logged_text = "\n".join(call.args[1] for call in debug_log_mock.call_args_list)
        self.assertIn("External LM format request", logged_text)
        self.assertIn("Tropical funk", logged_text)
        self.assertIn("External LM raw response", logged_text)

    @patch("acestep.debug_utils.debug_log_for")
    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_non_ollama_debug_logging_omits_api_key_and_logs_safe_payload_view(
        self,
        resolve_api_key_mock,
        urlopen_mock,
        debug_log_mock,
    ) -> None:
        """Non-Ollama debug logs should exclude secrets and log only safe payload fields."""

        resolve_api_key_mock.return_value = "test-key"
        urlopen_mock.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "A bright synth-pop groove with glossy hooks.",
                                    "lyrics": "City lights / we glow tonight",
                                    "bpm": 120,
                                    "duration": 30,
                                    "key_scale": "C Major",
                                    "time_signature": "4/4",
                                    "vocal_language": "en",
                                    "instrumental": False,
                                }
                            )
                        }
                    }
                ]
            }
        )

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "openai",
                "ACESTEP_EXTERNAL_LM_MODEL": "gpt-4o-mini",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "https://api.openai.com/v1/chat/completions",
            },
            clear=False,
        ):
            format_sample_with_external_provider(
                caption="synth pop",
                lyrics="",
                user_metadata={"bpm": 120},
                debug=True,
            )

        logged_text = "\n".join(call.args[1] for call in debug_log_mock.call_args_list)
        self.assertIn("External LM format request", logged_text)
        self.assertIn("\"model\": \"gpt-4o-mini\"", logged_text)
        self.assertIn("\"max_tokens\": 768", logged_text)
        self.assertNotIn("Authorization", logged_text)
        self.assertNotIn("test-key", logged_text)

    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_format_sample_with_external_provider_uses_native_ollama_chat_without_thinking(
        self,
        resolve_api_key_mock,
    ) -> None:
        """Ollama formatting should use native chat with ``think`` disabled."""

        resolve_api_key_mock.return_value = ""
        recorder = _UrlopenRecorder(
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "caption": "A bright tropical funk arrangement grows from airy synths into a lively chorus.",
                            "lyrics": "",
                            "bpm": 125,
                            "duration": 240,
                            "key_scale": "G Major",
                            "time_signature": "4/4",
                            "vocal_language": "English",
                            "instrumental": False,
                        }
                    ),
                }
            }
        )

        with patch(
            "acestep.text_tasks.external_lm_http_common.request.urlopen",
            side_effect=recorder,
        ), patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_LM_MODEL": "qwen3:4b",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=False,
        ):
            result = format_sample_with_external_provider(
                caption="Tropical funk",
                lyrics="",
                user_metadata={},
            )

        self.assertTrue(result.success)
        self.assertIsNotNone(recorder.last_request)
        self.assertEqual(recorder.last_request.full_url, "http://127.0.0.1:11434/api/chat")
        payload = json.loads(recorder.last_request.data.decode("utf-8"))
        self.assertEqual(payload["think"], False)
        self.assertEqual(payload["format"]["type"], "object")
        self.assertIn("caption", payload["format"]["properties"])
        self.assertEqual(payload["format"]["properties"]["instrumental"]["type"], "boolean")
        self.assertEqual(payload["options"]["num_predict"], 768)

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_format_sample_with_external_provider_retries_when_caption_is_unchanged(
        self,
        request_plan_mock,
    ) -> None:
        """An unchanged caption should trigger one retry before returning."""

        request_plan_mock.side_effect = [
            (
                SimpleNamespace(
                    caption="Salsa dura with brass section, call-and-response vocals, live club energy",
                    lyrics="",
                    bpm=125,
                    duration=240,
                    key_scale="D major",
                    time_signature="4/4",
                    vocal_language="en",
                    instrumental=False,
                ),
                SimpleNamespace(label="Ollama"),
                "qwen3:4b",
            ),
            (
                SimpleNamespace(
                    caption="A salsa dura arrangement opens with brass fanfares and call-and-response vocals, "
                    "then builds through piano montunos into a high-energy live chorus.",
                    lyrics="",
                    bpm=125,
                    duration=240,
                    key_scale="D major",
                    time_signature="4/4",
                    vocal_language="en",
                    instrumental=False,
                ),
                SimpleNamespace(label="Ollama"),
                "qwen3:4b",
            ),
        ]

        result = format_sample_with_external_provider(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            lyrics="",
            user_metadata={"bpm": 125, "duration": 240, "keyscale": "D major", "timesignature": "4/4"},
        )

        self.assertEqual(request_plan_mock.call_count, 2)
        self.assertIn("Retry instruction", request_plan_mock.call_args_list[1].kwargs["intent"])
        self.assertIn("A salsa dura arrangement", result.caption)

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_format_sample_with_external_provider_uses_local_fallback_after_failed_retry(
        self,
        request_plan_mock,
    ) -> None:
        """A second unchanged caption should fall back to a local narrative caption."""

        unchanged = SimpleNamespace(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            lyrics="",
            bpm=125,
            duration=240,
            key_scale="D major",
            time_signature="4/4",
            vocal_language="es",
            instrumental=False,
        )
        request_plan_mock.side_effect = [
            (unchanged, SimpleNamespace(label="Ollama"), "qwen3:4b"),
            (unchanged, SimpleNamespace(label="Ollama"), "qwen3:4b"),
        ]

        result = format_sample_with_external_provider(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            lyrics="",
            user_metadata={"bpm": 125, "duration": 240, "keyscale": "D major", "timesignature": "4/4"},
        )

        self.assertEqual(request_plan_mock.call_count, 2)
        self.assertNotEqual(
            result.caption,
            "Salsa dura with brass section, call-and-response vocals, live club energy",
        )
        self.assertIn("125 BPM", result.caption)

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_format_sample_with_external_provider_uses_fallback_when_retry_request_fails(
        self,
        request_plan_mock,
    ) -> None:
        """A failed retry request should still return a local fallback caption."""

        request_plan_mock.side_effect = [
            (
                SimpleNamespace(
                    caption="Salsa dura with brass section, call-and-response vocals, live club energy",
                    lyrics="",
                    bpm=125,
                    duration=240,
                    key_scale="D major",
                    time_signature="4/4",
                    vocal_language="es",
                    instrumental=False,
                ),
                SimpleNamespace(label="Ollama"),
                "qwen3:4b",
            ),
            ExternalAIClientError("retry failed"),
        ]

        result = format_sample_with_external_provider(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            lyrics="",
            user_metadata={"bpm": 125, "duration": 240, "keyscale": "D major", "timesignature": "4/4"},
        )

        self.assertEqual(request_plan_mock.call_count, 2)
        self.assertNotEqual(
            result.caption,
            "Salsa dura with brass section, call-and-response vocals, live club energy",
        )
        self.assertIn("125 BPM", result.caption)

    @patch("acestep.text_tasks.external_lm_tasks._request_external_plan")
    def test_create_sample_with_external_provider_preserves_explicit_vocal_request(
        self,
        request_plan_mock,
    ) -> None:
        """Provider instrumental drift should not override an explicit vocal request."""

        request_plan_mock.return_value = (
            SimpleNamespace(
                caption="A bright pop duet grows into a glossy final chorus.",
                lyrics="",
                bpm=125,
                duration=240,
                key_scale="D major",
                time_signature="4/4",
                vocal_language="en",
                instrumental=True,
            ),
            SimpleNamespace(label="OpenAI"),
            "gpt-4o-mini",
        )

        result = create_sample_with_external_provider(
            query="Bright pop duet with male and female vocals",
            instrumental=False,
            vocal_language="en",
        )

        self.assertTrue(result.success)
        self.assertFalse(result.instrumental)
        self.assertEqual(result.lyrics, "")

    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_format_sample_with_external_provider_surfaces_timeout_cleanly(
        self,
        resolve_api_key_mock,
        urlopen_mock,
    ) -> None:
        """Provider timeouts should raise a user-facing external-client error."""

        resolve_api_key_mock.return_value = ""
        urlopen_mock.side_effect = TimeoutError("timed out")

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_LM_MODEL": "gpt4all:deepseek-coder-6-7b-instruct-q4-0",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=False,
        ):
            with self.assertRaises(ExternalAIClientError) as exc:
                format_sample_with_external_provider(
                    caption="synth pop",
                    lyrics="",
                    user_metadata={},
                )

        self.assertIn("Timed out waiting for the external provider response", str(exc.exception))

    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_warmup.resolve_external_api_key_for_runtime")
    def test_warm_up_external_provider_for_ollama(self, resolve_api_key_mock, urlopen_mock) -> None:
        """Ollama warm-up should issue a tiny request and report success."""

        resolve_api_key_mock.return_value = ""
        urlopen_mock.side_effect = [
            _FakeResponse({"models": []}),
            _FakeResponse({"response": "", "done": True}),
        ]

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_LM_MODEL": "qwen3:8b",
                "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=False,
        ):
            status = warm_up_external_provider(timeout_sec=5)

        self.assertEqual(status, "Ollama warm-up complete (qwen3:8b)")


if __name__ == "__main__":
    unittest.main()
