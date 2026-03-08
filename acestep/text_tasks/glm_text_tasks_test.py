"""Tests for GLM text-task planning helpers."""

from __future__ import annotations

import json
from io import BytesIO
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from acestep.text_tasks.glm_text_tasks import (
    GlmClientError,
    GlmPlan,
    build_planning_messages,
    build_acestep_generation_payload,
    parse_glm_chat_response,
    request_glm_plan,
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


class GlmTextTasksTests(unittest.TestCase):
    """Validate GLM response parsing and ACE-Step payload mapping."""

    def test_build_planning_messages_requests_narrative_structured_caption(self) -> None:
        """System prompt should request arrangement, instrumentation, and vocalist detail."""
        messages = build_planning_messages("dreamy synth ballad", task_focus="all")
        system_content = messages[0]["content"]

        self.assertIn("linear arrangement arc", system_content)
        self.assertIn("core instrumentation", system_content)
        self.assertIn("vocalist timbre/delivery", system_content)

    def test_build_planning_messages_format_focus_preserves_intent(self) -> None:
        """Format-focus prompt should preserve intent while improving specificity."""
        messages = build_planning_messages("tighten this caption", task_focus="format")
        user_content = messages[1]["content"]
        self.assertIn("For format focus: preserve user intent", user_content)

    def test_parse_glm_response_supports_json_fences(self) -> None:
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
        plan = parse_glm_chat_response(json.dumps(response))

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
        plan = parse_glm_chat_response(json.dumps(response), protocol="anthropic_messages")
        self.assertEqual("warm piano", plan.caption)
        self.assertTrue(plan.instrumental)

    def test_build_payload_disables_local_llm_flags(self) -> None:
        """ACE-Step payload output should bypass local LM rewrite/sample/cot steps."""
        plan = GlmPlan(
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

    @patch("acestep.text_tasks.glm_text_tasks.request.urlopen")
    def test_request_glm_plan_calls_chat_completions_endpoint(self, urlopen_mock) -> None:
        """Client should call configured GLM endpoint and parse plan content."""
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

        plan = request_glm_plan(
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

    @patch("acestep.text_tasks.glm_text_tasks.request.urlopen")
    def test_request_glm_plan_surfaces_model_not_found_guidance(self, urlopen_mock) -> None:
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
            request_glm_plan(
                api_key="secret",
                intent="test intent",
                model="glm-4-flash",
                base_url="https://example.invalid/chat/completions",
                timeout_sec=5,
                task_focus="all",
            )

    @patch("acestep.text_tasks.glm_text_tasks.request.urlopen")
    def test_request_glm_plan_surfaces_1113_balance_guidance(self, urlopen_mock) -> None:
        """HTTP 1113 responses should include billing-path guidance and request target."""
        body = b'{"error":{"code":"1113","message":"Insufficient balance"}}'
        urlopen_mock.side_effect = HTTPError(
            url="https://example.invalid/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(body),
        )

        with self.assertRaises(GlmClientError) as exc_context:
            request_glm_plan(
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

    @patch("acestep.text_tasks.glm_text_tasks.request.urlopen")
    def test_request_glm_plan_anthropic_uses_x_api_key_header(self, urlopen_mock) -> None:
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

        plan = request_glm_plan(
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


if __name__ == "__main__":
    unittest.main()
