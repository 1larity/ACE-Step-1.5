"""Tests for external LM task adapter error handling behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_tasks import (
    GlmClientError,
    create_sample_with_external_provider,
    format_sample_with_external_provider,
)
from acestep.text_tasks.secure_secret_store import SecretStoreError


class ExternalLmTasksErrorTests(unittest.TestCase):
    """Verify secret-store failures are surfaced as GLM client errors."""

    @patch(
        "acestep.text_tasks.external_lm_tasks.resolve_external_api_key_for_runtime",
        side_effect=SecretStoreError("Missing credentials"),
    )
    def test_create_sample_wraps_secret_store_error(self, _resolve_key_mock) -> None:
        """Create-sample adapter should wrap secret-store errors as GlmClientError."""
        with self.assertRaisesRegex(GlmClientError, "Missing credentials"):
            create_sample_with_external_provider(
                query="test",
                instrumental=False,
                vocal_language="en",
            )

    @patch(
        "acestep.text_tasks.external_lm_tasks.resolve_external_api_key_for_runtime",
        side_effect=SecretStoreError("Missing credentials"),
    )
    def test_format_sample_wraps_secret_store_error(self, _resolve_key_mock) -> None:
        """Format adapter should wrap secret-store errors as GlmClientError."""
        with self.assertRaisesRegex(GlmClientError, "Missing credentials"):
            format_sample_with_external_provider(
                caption="caption",
                lyrics="lyrics",
                user_metadata={},
            )


if __name__ == "__main__":
    unittest.main()
