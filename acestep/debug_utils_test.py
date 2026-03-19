"""Focused tests for debug-log redaction helpers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from acestep.debug_utils import _redact_sensitive_text, debug_log


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

    def test_debug_log_prints_redacted_text(self) -> None:
        """The public logger should print redacted values instead of secrets."""

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            debug_log(
                "x-api-key: super-secret\nAuthorization: Bearer abc123",
                mode="ON",
                prefix="llm",
            )

        output = buffer.getvalue()
        self.assertIn("x-api-key: [REDACTED]", output)
        self.assertIn("Authorization: [REDACTED]", output)
        self.assertNotIn("super-secret", output)
        self.assertNotIn("abc123", output)


if __name__ == "__main__":
    unittest.main()
