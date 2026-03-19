"""Focused tests for debug-log redaction helpers."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from acestep.debug_utils import _redact_sensitive_mapping, _redact_sensitive_text, debug_log


class DebugUtilsTests(unittest.TestCase):
    """Verify debug logging redacts common secret-like values."""

    def test_redact_sensitive_text_masks_common_key_value_pairs(self) -> None:
        """Inline secret-like assignments should be redacted."""

        message = (
            "api_key=test-key passphrase=hunter2 password: secret123 "
            "Authorization=Bearer abc.def token='xyz'"
        )

        redacted = _redact_sensitive_text(message)

        self.assertNotIn("test-key", redacted)
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("secret123", redacted)
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("xyz", redacted)
        self.assertIn("api_key=[REDACTED]", redacted)
        self.assertIn("passphrase=[REDACTED]", redacted)
        self.assertIn("password: [REDACTED]", redacted)
        self.assertIn("Authorization=[REDACTED]", redacted)
        self.assertIn("token=[REDACTED]", redacted)

    def test_debug_log_prints_safe_text(self) -> None:
        """The public logger should print safe text with the requested prefix."""

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            debug_log(
                "safe summary content_length=42",
                mode="ON",
                prefix="llm",
            )

        output = buffer.getvalue()
        self.assertIn("[llm]", output)
        self.assertIn("safe summary content_length=42", output)

    def test_redact_sensitive_mapping_masks_nested_secret_keys(self) -> None:
        """Nested mapping structures should redact secret-bearing keys by name."""

        payload = {
            "headers": {
                "Authorization": "Bearer abc123",
                "x-api-key": "super-secret",
            },
            "nested": [{"token": "xyz"}],
        }

        redacted = _redact_sensitive_mapping(payload)

        self.assertEqual(redacted["headers"]["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["headers"]["x-api-key"], "[REDACTED]")
        self.assertEqual(redacted["nested"][0]["token"], "[REDACTED]")

    def test_redact_sensitive_text_masks_json_secret_values(self) -> None:
        """JSON strings should have secret-bearing keys redacted before logging."""

        payload = json.dumps(
            {
                "Authorization": "Bearer abc123",
                "x-api-key": "super-secret",
                "nested": {"password": "hunter2"},
            }
        )

        redacted = _redact_sensitive_text(payload)

        self.assertIn("\"Authorization\": \"[REDACTED]\"", redacted)
        self.assertIn("\"x-api-key\": \"[REDACTED]\"", redacted)
        self.assertIn("\"password\": \"[REDACTED]\"", redacted)
        self.assertNotIn("abc123", redacted)
        self.assertNotIn("super-secret", redacted)
        self.assertNotIn("hunter2", redacted)

    def test_debug_log_prints_safe_mapping_messages(self) -> None:
        """Structured safe messages should still be stringified for debug output."""

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            debug_log(
                {
                    "role": "assistant",
                    "content_length": 128,
                },
                mode="ON",
                prefix="llm",
            )

        output = buffer.getvalue()
        self.assertIn("\"role\": \"assistant\"", output)
        self.assertIn("\"content_length\": 128", output)


if __name__ == "__main__":
    unittest.main()
