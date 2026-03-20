"""Tests for external LM runtime credential resolution helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from acestep.text_tasks.external_lm_runtime_access import (
    has_external_lm_runtime_credentials,
    resolve_external_api_key_for_runtime,
)
from acestep.text_tasks.secure_secret_store import SecretStoreError


class ExternalLmRuntimeAccessTests(unittest.TestCase):
    """Verify runtime credential resolution for external LM providers."""

    def test_resolve_prefers_generic_external_api_key_env(self) -> None:
        """The generic external API key env var should take precedence."""

        with patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_API_KEY": "shared-key",
                "ACESTEP_OPENAI_API_KEY": "provider-key",
            },
            clear=True,
        ):
            resolved = resolve_external_api_key_for_runtime("openai", lambda provider: provider or "zai")

        self.assertEqual(resolved, "shared-key")

    def test_resolve_uses_native_keyring_store_without_runtime_passphrase(self) -> None:
        """Native keyring stores should bypass passphrase lookup entirely."""

        store = MagicMock()
        store.uses_native_keyring.return_value = True
        store.load.return_value = "native-key"

        with patch.dict(os.environ, {}, clear=True), patch(
            "acestep.text_tasks.external_lm_runtime_access._resolve_secret_store",
            return_value=store,
        ), patch(
            "acestep.text_tasks.external_lm_runtime_access.resolve_runtime_passphrase",
        ) as resolve_passphrase_mock:
            resolved = resolve_external_api_key_for_runtime("openai", lambda provider: provider or "zai")

        resolve_passphrase_mock.assert_not_called()
        store.load.assert_called_once_with(passphrase="")
        self.assertEqual(resolved, "native-key")

    def test_has_credentials_returns_false_when_secret_store_errors(self) -> None:
        """Secret store failures should degrade to unavailable credentials."""

        with patch(
            "acestep.text_tasks.external_lm_runtime_access.resolve_external_api_key_for_runtime",
            side_effect=SecretStoreError("missing"),
        ):
            available = has_external_lm_runtime_credentials("openai", lambda provider: provider or "zai")

        self.assertFalse(available)


if __name__ == "__main__":
    unittest.main()
