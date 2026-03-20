"""Tests for external LM picker integration helpers."""

import os
import unittest
from unittest.mock import patch

from acestep.text_tasks.external_lm_mode import (
    activate_external_lm_mode,
    deactivate_external_lm_mode,
    get_external_lm_choices,
    is_external_lm_active,
    is_lm_ready,
    parse_external_lm_selection,
)


class ExternalLmModeTests(unittest.TestCase):
    """Verify external LM dropdown helpers and active-mode state."""

    def test_get_choices_uses_configured_provider_and_model(self) -> None:
        """Configured provider/model should surface as a main dropdown choice."""

        with patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "openai",
                "ACESTEP_EXTERNAL_LM_MODEL": "gpt-4o-mini",
            },
            clear=True,
        ):
            self.assertEqual(get_external_lm_choices(), ["external:openai:gpt-4o-mini"])

    def test_activate_and_deactivate_external_mode(self) -> None:
        """Dropdown activation should enable and disable external mode state."""

        with patch.dict(os.environ, {}, clear=True):
            selection = activate_external_lm_mode("external:claude:claude-3-7-sonnet-latest")
            self.assertEqual(selection.provider, "claude")
            self.assertTrue(is_external_lm_active())
            self.assertEqual(
                parse_external_lm_selection("external:claude:claude-3-7-sonnet-latest").model,
                "claude-3-7-sonnet-latest",
            )

            deactivate_external_lm_mode()
            self.assertFalse(is_external_lm_active())

    def test_is_lm_ready_accepts_external_selection_without_local_llm(self) -> None:
        """An external LM picker value should count as LM-ready for UI gating."""

        with patch(
            "acestep.text_tasks.external_lm_runtime_access.resolve_external_api_key_for_runtime",
            return_value="test-key",
        ), patch.dict(os.environ, {}, clear=True):
            self.assertTrue(is_lm_ready(lm_model_path="external:openai:gpt-4o-mini"))

    def test_is_lm_ready_checks_active_external_credentials(self) -> None:
        """Active external mode should require runtime credentials for ready state."""

        with patch(
            "acestep.text_tasks.external_lm_runtime_access.resolve_external_api_key_for_runtime",
            return_value="",
        ), patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_LM_ENABLED": "true",
                "ACESTEP_EXTERNAL_LM_PROVIDER": "openai",
            },
            clear=True,
        ):
            self.assertFalse(is_lm_ready())

        with patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_LM_ENABLED": "true",
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=True,
        ), patch(
            "acestep.text_tasks.external_lm_runtime_access.socket.create_connection"
        ) as create_connection_mock:
            create_connection_mock.return_value.__enter__.return_value = object()
            self.assertTrue(is_lm_ready())

    def test_is_lm_ready_marks_ollama_unavailable_when_endpoint_is_down(self) -> None:
        """Ollama should not count as ready when its local endpoint is unreachable."""

        with patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_LM_ENABLED": "true",
                "ACESTEP_EXTERNAL_LM_PROVIDER": "ollama",
                "ACESTEP_EXTERNAL_BASE_URL": "http://127.0.0.1:11434/v1/chat/completions",
            },
            clear=True,
        ), patch(
            "acestep.text_tasks.external_lm_runtime_access.socket.create_connection",
            side_effect=OSError("connection refused"),
        ):
            self.assertFalse(is_lm_ready())


if __name__ == "__main__":
    unittest.main()
