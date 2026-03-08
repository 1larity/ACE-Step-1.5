"""Tests for external LM runtime credential resolution helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks.external_lm_mode import (
    deactivate_external_lm_mode,
    get_external_lm_choices,
    is_external_lm_active,
    parse_external_lm_selection,
    resolve_glm_api_key_for_runtime,
)
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


class ExternalLmModeRuntimeCredentialTests(unittest.TestCase):
    """Verify runtime API key resolution for external LM mode."""

    @patch.dict("os.environ", {"ACESTEP_GLM_API_KEY": "direct-key"}, clear=True)
    def test_resolve_runtime_key_uses_direct_env_key(self) -> None:
        """Direct API key env var should bypass encrypted store decryption."""
        self.assertEqual("direct-key", resolve_glm_api_key_for_runtime())

    @patch.dict("os.environ", {}, clear=True)
    @patch("acestep.text_tasks.external_lm_mode.resolve_runtime_passphrase", return_value=None)
    def test_resolve_runtime_key_raises_when_no_credentials(
        self,
        _resolve_passphrase_mock,
    ) -> None:
        """Missing key and passphrase should raise explicit configuration guidance."""
        with self.assertRaises(SecretStoreError):
            resolve_glm_api_key_for_runtime()

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "acestep.text_tasks.external_lm_mode.resolve_runtime_passphrase",
        return_value="store-pass",
    )
    def test_resolve_runtime_key_uses_encrypted_store_when_passphrase_available(
        self,
        _resolve_passphrase_mock,
    ) -> None:
        """Runtime passphrase should allow decryption of stored API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "glm_api_key.enc"
            store = EncryptedSecretStore(secret_path=path)
            store.save(secret="stored-key", passphrase="store-pass")

            with patch.dict("os.environ", {"ACESTEP_GLM_SECRET_PATH": str(path)}, clear=True):
                self.assertEqual("stored-key", resolve_glm_api_key_for_runtime())

    def test_parse_external_selection_supports_provider_prefixed_values(self) -> None:
        """Parser should support explicit provider:model external dropdown tokens."""
        selection = parse_external_lm_selection("external:openai:gpt-4o-mini")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual("openai", selection.provider)
        self.assertEqual("gpt-4o-mini", selection.model)

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "acestep.text_tasks.external_lm_mode.hydrate_external_lm_env_from_store",
        return_value=False,
    )
    def test_default_external_choices_empty_when_not_configured(
        self,
        _hydrate_mock,
    ) -> None:
        """No external choices should be shown when external runtime is not configured."""
        choices = get_external_lm_choices()
        self.assertEqual([], choices)

    @patch.dict(
        "os.environ",
        {
            "ACESTEP_EXTERNAL_LM_PROVIDER": "zai",
            "ACESTEP_EXTERNAL_LM_MODEL": "glm-5",
        },
        clear=True,
    )
    def test_choices_include_only_configured_external_model(self) -> None:
        """Configured provider/model should expose a single external LM dropdown token."""
        choices = get_external_lm_choices()
        self.assertEqual(["external:zai:glm-5"], choices)

    @patch.dict(
        "os.environ",
        {"ACESTEP_EXTERNAL_LM_CHOICES": "external:zai:glm-5,external:openai:gpt-4o-mini"},
        clear=True,
    )
    def test_choices_env_override_still_supported(self) -> None:
        """Explicit ACESTEP_EXTERNAL_LM_CHOICES should override auto-derived choices."""
        choices = get_external_lm_choices()
        self.assertEqual(["external:zai:glm-5", "external:openai:gpt-4o-mini"], choices)

    @patch.dict("os.environ", {}, clear=True)
    @patch("acestep.text_tasks.external_lm_mode.hydrate_external_lm_env_from_store")
    def test_choices_can_be_hydrated_from_persisted_runtime_store(
        self,
        hydrate_mock,
    ) -> None:
        """When env is empty, persisted runtime store hydration should restore external choice."""

        def _hydrate() -> bool:
            os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"] = "zai"
            os.environ["ACESTEP_EXTERNAL_LM_MODEL"] = "glm-5"
            return True

        hydrate_mock.side_effect = _hydrate
        choices = get_external_lm_choices()
        self.assertEqual(["external:zai:glm-5"], choices)

    @patch.dict(
        "os.environ",
        {
            "ACESTEP_EXTERNAL_LM_ENABLED": "true",
            "ACESTEP_EXTERNAL_LM_PROVIDER": "zai",
            "ACESTEP_EXTERNAL_LM_MODEL": "glm-5",
            "ACESTEP_EXTERNAL_LM_PROTOCOL": "openai_chat",
            "ACESTEP_TEXT_PROVIDER": "zai",
        },
        clear=True,
    )
    def test_deactivate_disables_runtime_but_preserves_configured_choice(self) -> None:
        """Deactivation should keep configured provider/model for dropdown persistence."""
        deactivate_external_lm_mode()

        self.assertFalse(is_external_lm_active())
        self.assertIsNone(os.environ.get("ACESTEP_TEXT_PROVIDER"))
        self.assertEqual("zai", os.environ.get("ACESTEP_EXTERNAL_LM_PROVIDER"))
        self.assertEqual("glm-5", os.environ.get("ACESTEP_EXTERNAL_LM_MODEL"))
        self.assertEqual("openai_chat", os.environ.get("ACESTEP_EXTERNAL_LM_PROTOCOL"))
        self.assertEqual(["external:zai:glm-5"], get_external_lm_choices())


if __name__ == "__main__":
    unittest.main()
