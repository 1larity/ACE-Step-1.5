"""Tests for External AI text-task planning helpers."""

from __future__ import annotations

import json
import os
from io import BytesIO
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from acestep.text_tasks.external_ai_text_tasks import (
    ExternalAIClientError,
    ExternalAIPlan,
    build_planning_messages,
    build_acestep_generation_payload,
    parse_external_ai_chat_response,
    request_external_ai_plan,
)


class _FakeHttpResponse:
    """Minimal response object for mocking urllib chat-completions calls."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        """Return prepared response bytes."""
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit context manager without suppressing errors."""
        return False


class ExternalAITextTasksTests(unittest.TestCase):
    """Validate External AI response parsing and ACE-Step payload mapping."""

    def test_build_planning_messages_requests_narrative_structured_caption(self) -> None:
        """System prompt should request arrangement, instrumentation, and vocalist detail."""
        messages = build_planning_messages("dreamy synth ballad", task_focus="all")
        system_content = messages[0]["content"]

        self.assertIn("linear arrangement arc", system_content)
        self.assertIn("core instrumentation", system_content)
        self.assertIn("singer gender and delivery mood/timbre", system_content)

    def test_build_planning_messages_format_focus_preserves_intent(self) -> None:
        """Format-focus prompt should preserve intent while improving specificity."""
        messages = build_planning_messages("tighten this caption", task_focus="format")
        user_content = messages[1]["content"]
        self.assertIn("For format focus: preserve user intent", user_content)

    def test_parse_external_ai_response_supports_json_fences(self) -> None:
        """Parser should accept markdown-fenced JSON content from providers."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"caption\":\"bright synthwave\",\"lyrics\":\"[Instrumental]\",\"bpm\":120,\"duration\":30,\"key_scale\":\"C Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":true}\n```"
                    }
                }
            ]
        }
        plan = parse_external_ai_chat_response(json.dumps(response))

        self.assertEqual("bright synthwave", plan.caption)
        self.assertTrue(plan.instrumental)
        self.assertEqual(120, plan.bpm)

    def test_parse_response_supports_anthropic_content_shape(self) -> None:
        """Parser should support Anthropic messages API text-block shape."""
        response = {
            "content": [
                {
                    "type": "text",
                    "text": "{\"caption\":\"warm piano\",\"lyrics\":\"[Instrumental]\",\"bpm\":95,"
                    "\"duration\":25,\"key_scale\":\"D Major\",\"time_signature\":\"4/4\","
                    "\"vocal_language\":\"unknown\",\"instrumental\":true}",
                }
            ]
        }
        plan = parse_external_ai_chat_response(json.dumps(response), protocol="anthropic_messages")
        self.assertEqual("warm piano", plan.caption)
        self.assertTrue(plan.instrumental)

    def test_parse_external_ai_response_supports_think_wrapped_json(self) -> None:
        """Parser should recover the JSON object when provider prepends think blocks."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": '<think>drafting fields</think>\n{"caption":"bright synthwave","lyrics":"[Instrumental]","bpm":120,"duration":30,"key_scale":"C Minor","time_signature":"4/4","vocal_language":"en","instrumental":true}'
                    }
                }
            ]
        }
        plan = parse_external_ai_chat_response(json.dumps(response))

        self.assertEqual("bright synthwave", plan.caption)
        self.assertTrue(plan.instrumental)

    def test_parse_external_ai_response_repairs_trailing_commas(self) -> None:
        """Parser should tolerate trailing commas in otherwise valid JSON objects."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"caption":"night drive","lyrics":"[Instrumental]","bpm":118,"duration":28,"key_scale":"A Minor","time_signature":"4/4","vocal_language":"en","instrumental":true,}'
                    }
                }
            ]
        }
        plan = parse_external_ai_chat_response(json.dumps(response))

        self.assertEqual("night drive", plan.caption)
        self.assertEqual(118, plan.bpm)

    def test_parse_external_ai_response_lyrics_focus_accepts_plain_sectioned_lyrics(self) -> None:
        """Lyrics-focused parsing should accept plain sectioned lyric text from providers."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "[Verse 1]\nBeneath the stars, I trace my name\n\n[Chorus]\nWe rise through the shadows, united in flame"
                    }
                }
            ]
        }

        plan = parse_external_ai_chat_response(json.dumps(response), task_focus="lyrics")

        self.assertEqual("", plan.caption)
        self.assertEqual(
            "[Verse 1]\nBeneath the stars, I trace my name\n\n[Chorus]\nWe rise through the shadows, united in flame",
            plan.lyrics,
        )
        self.assertFalse(plan.instrumental)

    def test_parse_external_ai_response_invalid_json_includes_content_preview(self) -> None:
        """Invalid JSON errors should surface a preview of the raw provider content."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "caption: bright synthwave | lyrics: [Verse 1] neon skies"
                    }
                }
            ]
        }

        with self.assertRaises(ExternalAIClientError) as exc_context:
            parse_external_ai_chat_response(json.dumps(response))

        message = str(exc_context.exception)
        self.assertIn("External AI content is not valid JSON.", message)
        self.assertIn("Task focus 'all' requires valid JSON.", message)
        self.assertIn("Content preview:", message)
        self.assertIn("caption: bright synthwave", message)

    def test_parse_external_ai_response_format_focus_rejects_plain_lyrics(self) -> None:
        """Format/CoT parsing should reject plain lyric text and require JSON."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "[Verse 1]\nBeneath the stars, I trace my name\n\n[Chorus]\nWe rise through the shadows"
                    }
                }
            ]
        }

        with self.assertRaises(ExternalAIClientError) as exc_context:
            parse_external_ai_chat_response(json.dumps(response), task_focus="format")

        message = str(exc_context.exception)
        self.assertIn("External AI content is not valid JSON.", message)
        self.assertIn("Task focus 'format' requires valid JSON.", message)

    def test_parse_external_ai_response_surfaces_reasoning_token_exhaustion(self) -> None:
        """Empty content plus reasoning-only output should produce an actionable error."""
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "Drafting JSON fields but ran out of tokens before emitting content.",
                    },
                }
            ]
        }

        with self.assertRaises(ExternalAIClientError) as exc_context:
            parse_external_ai_chat_response(json.dumps(response))

        message = str(exc_context.exception)
        self.assertIn("empty content", message)
        self.assertIn("ACESTEP_OPENAI_MAX_TOKENS", message)
        self.assertIn("Reasoning preview:", message)

    def test_build_payload_disables_local_llm_flags(self) -> None:
        """ACE-Step payload output should bypass local LM rewrite/sample/cot steps."""
        plan = ExternalAIPlan(
            caption="cinematic piano",
            lyrics="",
            bpm=90,
            duration=45.0,
            key_scale="A Minor",
            time_signature="4/4",
            vocal_language="en",
            instrumental=True,
        )
        payload = build_acestep_generation_payload(plan)

        self.assertEqual("[Instrumental]", payload["lyrics"])
        self.assertFalse(payload["use_format"])
        self.assertFalse(payload["sample_mode"])
        self.assertFalse(payload["use_cot_caption"])
        self.assertFalse(payload["use_cot_language"])

    @patch("acestep.text_tasks.external_ai_text_tasks.logger.debug")
    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_logs_request_and_response_when_debug_enabled(
        self,
        urlopen_mock,
        logger_debug_mock,
    ) -> None:
        """Debug mode should log request payloads and raw/extracted responses."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "{\"caption\":\"edm drop\",\"lyrics\":\"[Verse]...\",\"bpm\":128,\"duration\":32,\"key_scale\":\"F Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":false}"
                    }
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        with patch.dict(os.environ, {"ACESTEP_DEBUG_LM_TASKS": "1"}, clear=False):
            plan = request_external_ai_plan(
                api_key="secret",
                intent="energetic edm with vocal hooks",
                model="glm-4-flash",
                base_url="https://api.z.ai/api/paas/v4/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

        self.assertEqual("edm drop", plan.caption)
        log_messages = [call.args[0] for call in logger_debug_mock.call_args_list]
        self.assertTrue(any("External AI request protocol=" in message for message in log_messages))
        self.assertTrue(any("External AI raw response protocol=" in message for message in log_messages))
        self.assertTrue(any("External AI extracted content protocol=" in message for message in log_messages))

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_uses_longer_default_timeout(self, urlopen_mock) -> None:
        """Requests should allow slower providers more than the old 60 second default."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "{\"caption\":\"edm drop\",\"lyrics\":\"[Verse]...\",\"bpm\":128,\"duration\":32,\"key_scale\":\"F Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":false}"
                    }
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        request_external_ai_plan(
            api_key="secret",
            intent="energetic edm with vocal hooks",
            model="glm-4-flash",
            base_url="https://example.invalid/chat/completions",
            task_focus="all",
        )

        self.assertEqual(120, urlopen_mock.call_args.kwargs["timeout"])

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_calls_chat_completions_endpoint(self, urlopen_mock) -> None:
        """Client should call the configured external endpoint and parse plan content."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "{\"caption\":\"edm drop\",\"lyrics\":\"[Verse]...\",\"bpm\":128,\"duration\":32,\"key_scale\":\"F Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":false}"
                    }
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        plan = request_external_ai_plan(
            api_key="secret",
            intent="energetic edm with vocal hooks",
            model="glm-4-flash",
            base_url="https://example.invalid/chat/completions",
            timeout_sec=5,
            task_focus="all",
        )

        self.assertEqual("edm drop", plan.caption)
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual("POST", sent_request.method)
        self.assertEqual(
            "Bearer secret",
            sent_request.headers["Authorization"],
        )
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(3072, payload["max_tokens"])
        self.assertNotIn("thinking", payload)

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_allows_thinking_override_for_supported_endpoints(self, urlopen_mock) -> None:
        """Provider-specific thinking controls should only be sent to endpoints that support them."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "{\"caption\":\"edm drop\",\"lyrics\":\"[Verse]...\",\"bpm\":128,\"duration\":32,\"key_scale\":\"F Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":false}"
                    }
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        with patch.dict(os.environ, {"ACESTEP_EXTERNAL_AI_THINKING": "enabled"}, clear=False):
            request_external_ai_plan(
                api_key="secret",
                intent="energetic edm with vocal hooks",
                model="glm-4-flash",
                base_url="https://api.z.ai/api/paas/v4/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

        payload = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        self.assertEqual("enabled", payload["thinking"]["type"])

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_does_not_send_thinking_to_openai(self, urlopen_mock) -> None:
        """Standard OpenAI chat-completions payloads should omit unsupported thinking controls."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "{\"caption\":\"edm drop\",\"lyrics\":\"[Verse]...\",\"bpm\":128,\"duration\":32,\"key_scale\":\"F Minor\",\"time_signature\":\"4/4\",\"vocal_language\":\"en\",\"instrumental\":false}"
                    }
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        with patch.dict(os.environ, {"ACESTEP_EXTERNAL_AI_THINKING": "enabled"}, clear=False):
            request_external_ai_plan(
                api_key="secret",
                intent="energetic edm with vocal hooks",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

        payload = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("thinking", payload)

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_surfaces_model_not_found_guidance(self, urlopen_mock) -> None:
        """HTTP 1211 responses should include targeted model-selection guidance."""
        body = b'{"error":{"code":"1211","message":"model not found"}}'
        urlopen_mock.side_effect = HTTPError(
            url="https://example.invalid/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(body),
        )

        with self.assertRaisesRegex(Exception, "Model not found"):
            request_external_ai_plan(
                api_key="secret",
                intent="test intent",
                model="glm-4-flash",
                base_url="https://example.invalid/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_surfaces_openai_quota_guidance(self, urlopen_mock) -> None:
        """OpenAI insufficient_quota errors should explain API billing vs ChatGPT/Codex usage."""
        body = b'{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"You exceeded your current quota"}}'
        urlopen_mock.side_effect = HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(body),
        )

        with self.assertRaises(ExternalAIClientError) as exc_context:
            request_external_ai_plan(
                api_key="secret",
                intent="test intent",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

        message = str(exc_context.exception)
        self.assertIn("OpenAI API quota is unavailable", message)
        self.assertIn("ChatGPT/Codex subscription usage is separate", message)
        self.assertIn("Responses API", message)

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_surfaces_1113_balance_guidance(self, urlopen_mock) -> None:
        """HTTP 1113 responses should include billing-path guidance and request target."""
        body = b'{"error":{"code":"1113","message":"Insufficient balance"}}'
        urlopen_mock.side_effect = HTTPError(
            url="https://example.invalid/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(body),
        )

        with self.assertRaises(ExternalAIClientError) as exc_context:
            request_external_ai_plan(
                api_key="secret",
                intent="test intent",
                model="glm-5",
                base_url="https://api.z.ai/api/paas/v4/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

        message = str(exc_context.exception)
        self.assertIn("1113 means billing quota is unavailable", message)
        self.assertIn("model=glm-5", message)
        self.assertIn("api/coding/paas/v4/chat/completions", message)

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_anthropic_uses_x_api_key_header(self, urlopen_mock) -> None:
        """Anthropic protocol should use x-api-key headers instead of bearer auth."""
        response = {
            "content": [
                {
                    "type": "text",
                    "text": "{\"caption\":\"anthemic\",\"lyrics\":\"[Verse]...\",\"bpm\":110,"
                    "\"duration\":40,\"key_scale\":\"A Minor\",\"time_signature\":\"4/4\","
                    "\"vocal_language\":\"en\",\"instrumental\":false}",
                }
            ]
        }
        urlopen_mock.return_value = _FakeHttpResponse(json.dumps(response))

        plan = request_external_ai_plan(
            api_key="anthropic-secret",
            intent="anthemic vocal lead",
            model="claude-3-7-sonnet-latest",
            base_url="https://example.invalid/v1/messages",
            timeout_sec=5,
            task_focus="all",
            protocol="anthropic_messages",
        )

        self.assertEqual("anthemic", plan.caption)
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual("anthropic-secret", sent_request.headers["X-api-key"])
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(1024, payload["max_tokens"])

    @patch("acestep.text_tasks.external_ai_text_tasks.request.urlopen")
    def test_request_external_ai_plan_wraps_timeout_error(self, urlopen_mock) -> None:
        """Timeout errors should be wrapped as ExternalAIClientError for UI-safe handling."""
        urlopen_mock.side_effect = TimeoutError("read timed out")

        with self.assertRaisesRegex(ExternalAIClientError, "timed out"):
            request_external_ai_plan(
                api_key="secret",
                intent="test intent",
                model="glm-4-flash",
                base_url="https://example.invalid/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )


if __name__ == "__main__":
    unittest.main()



