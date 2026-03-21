"""Tests for external LM task adapters."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from urllib import request
from unittest.mock import patch

from acestep.text_tasks.external_ai_types import ExternalAIClientError
from acestep.text_tasks.external_lm_captioning import (
    apply_user_metadata_overrides,
    build_fallback_caption,
    caption_needs_retry,
)
from acestep.text_tasks.external_lm_tasks import (
    create_sample_with_external_provider,
    format_sample_with_external_provider,
)
from acestep.text_tasks.external_lm_warmup import warm_up_external_provider


class _FakeResponse:
    """Simple context-manager HTTP response stub."""

    def __init__(self, payload: dict[str, object]):
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

    def __init__(self, response_payload: dict[str, object]):
        self.response_payload = response_payload
        self.last_request: request.Request | None = None

    def __call__(self, req, timeout=None):
        """Record the request and return a fake response."""

        _ = timeout
        self.last_request = req
        return _FakeResponse(self.response_payload)


class ExternalLmTasksTests(unittest.TestCase):
    """Verify external LM task adapters stay narrowly scoped and deterministic."""

    def test_caption_helpers_stay_available_for_adapter_flow(self) -> None:
        """Retry heuristics and fallback captioning should still behave as expected."""

        self.assertTrue(
            caption_needs_retry(
                original_caption="Progressive trance instrumental",
                generated_caption="Progressive trance instrumental with pads",
            )
        )
        fallback = build_fallback_caption(
            caption="Salsa dura with brass section, call-and-response vocals",
            user_metadata={"bpm": 125, "timesignature": "4/4", "keyscale": "D major"},
        )
        self.assertIn("125 BPM", fallback)
        plan = SimpleNamespace(
            bpm=1,
            duration=2.4,
            key_scale="C minor",
            time_signature="3/4",
            vocal_language="English",
        )
        result = apply_user_metadata_overrides(
            plan=plan,
            user_metadata={"bpm": 125, "duration": 240, "keyscale": "D major"},
        )
        self.assertEqual(125, result.bpm)
        self.assertEqual(240.0, result.duration)

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
        self.assertEqual("A bright synth-pop groove with glossy hooks.", result.caption)
        self.assertEqual(120, result.bpm)
        self.assertEqual("C Major", result.keyscale)

    @patch("acestep.debug_utils.debug_log_for")
    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_debug_mode_logs_safe_summaries_without_raw_response_or_api_key(
        self,
        resolve_api_key_mock,
        urlopen_mock,
        debug_log_mock,
    ) -> None:
        """Debug mode should log safe payload summaries without secrets or raw response text."""

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
        self.assertIn("\"content_length\":", logged_text)
        self.assertIn("\"max_tokens\": 768", logged_text)
        self.assertIn("External LM response summary", logged_text)
        self.assertIn("\"response_length\":", logged_text)
        self.assertNotIn("Authorization", logged_text)
        self.assertNotIn("test-key", logged_text)
        self.assertNotIn("City lights / we glow tonight", logged_text)

    @patch("acestep.text_tasks.external_lm_plan_requests.resolve_external_api_key_for_runtime")
    def test_format_sample_with_external_provider_uses_native_ollama_chat_without_thinking(
        self,
        resolve_api_key_mock,
    ) -> None:
        """Ollama formatting should use native chat with `think` disabled."""

        resolve_api_key_mock.return_value = ""
        recorder = _UrlopenRecorder(
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "caption": "A bright tropical funk arrangement grows into a lively chorus.",
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
        self.assertEqual("http://127.0.0.1:11434/api/chat", recorder.last_request.full_url)
        payload = json.loads(recorder.last_request.data.decode("utf-8"))
        self.assertFalse(payload["think"])
        self.assertEqual("object", payload["format"]["type"])
        self.assertEqual("boolean", payload["format"]["properties"]["instrumental"]["type"])
        self.assertEqual(768, payload["options"]["num_predict"])

    @patch("acestep.text_tasks.external_lm_tasks.request_external_plan")
    def test_format_sample_with_external_provider_retries_and_falls_back(
        self,
        request_plan_mock,
    ) -> None:
        """An unchanged caption should retry once and then fall back if needed."""

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
            ExternalAIClientError("retry failed"),
        ]

        result = format_sample_with_external_provider(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            lyrics="",
            user_metadata={"bpm": 125, "duration": 240, "keyscale": "D major"},
        )

        self.assertEqual(2, request_plan_mock.call_count)
        self.assertIn("125 BPM", result.caption)

    @patch("acestep.text_tasks.external_lm_tasks.request_external_plan")
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
        self.assertEqual("", result.lyrics)

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
            with self.assertRaisesRegex(
                ExternalAIClientError,
                "Timed out waiting for the external provider response",
            ):
                format_sample_with_external_provider(
                    caption="synth pop",
                    lyrics="",
                    user_metadata={},
                )

    @patch("acestep.text_tasks.external_lm_warmup.resolve_external_api_key_for_runtime")
    @patch("acestep.text_tasks.external_lm_http_common.request.urlopen")
    def test_warm_up_external_provider_for_ollama(self, urlopen_mock, resolve_api_key_mock) -> None:
        """The adapter slice should still expose the warm-up helper for service init."""

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

        self.assertEqual("Ollama warm-up complete (qwen3:8b)", status)


if __name__ == "__main__":
    unittest.main()
