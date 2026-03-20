"""Tests for provider-specific external AI response parsing helpers."""

from __future__ import annotations

import unittest

from acestep.text_tasks.external_ai_response_parsing import extract_protocol_message_content
from acestep.text_tasks.external_ai_types import ExternalAIClientError


class ExternalAIResponseParsingTests(unittest.TestCase):
    """Verify protocol-specific content extraction handles malformed payloads safely."""

    def test_extract_protocol_message_content_rejects_empty_openai_choices(self) -> None:
        """OpenAI-style parsing should fail cleanly when no choices are present."""

        with self.assertRaises(ExternalAIClientError) as exc:
            extract_protocol_message_content(
                raw_response='{"choices":[]}',
                protocol="openai_chat",
            )

        self.assertIn("missing choices", str(exc.exception))

    def test_extract_protocol_message_content_rejects_unknown_protocol(self) -> None:
        """Unknown response protocols should fail fast instead of guessing a shape."""

        with self.assertRaises(ExternalAIClientError) as exc:
            extract_protocol_message_content(
                raw_response='{"choices":[{"message":{"content":"ok"}}]}',
                protocol="mystery_protocol",
            )

        self.assertIn("Unsupported external response protocol", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
