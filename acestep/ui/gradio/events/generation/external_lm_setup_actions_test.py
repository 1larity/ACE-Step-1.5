"""Tests for external LM setup actions."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.generation.external_lm_setup_actions import (
    apply_external_lm_base_url_preset,
    build_external_lm_dropdown_sync_updates,
    build_external_lm_inactive_updates,
    fetch_external_lm_models_from_ui,
    load_external_lm_provider_defaults,
    save_external_lm_settings_from_ui,
)


class ExternalLmSetupActionsTests(unittest.TestCase):
    """Verify save action persists settings and updates the LM dropdown."""

    def test_save_updates_env_and_main_lm_picker(self) -> None:
        """Saving external settings should select the external LM in the main picker."""

        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = ["acestep-5Hz-lm-1.7B"]

        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "openai.enc"
            with patch.dict(
                os.environ,
                {
                    "XDG_DATA_HOME": tmpdir,
                    "ACESTEP_OPENAI_SECRET_PATH": str(secret_path),
                },
                clear=True,
            ):
                status, api_key_update, passphrase_update, lm_update = (
                    save_external_lm_settings_from_ui(
                        provider="openai",
                        protocol="openai_chat",
                        model="gpt-4o-mini",
                        base_url="https://api.openai.com/v1/chat/completions",
                        api_key="test-key",
                        store_passphrase="test-passphrase",
                        save_passphrase_to_keyring=False,
                        llm_handler=llm_handler,
                    )
                )
                self.assertEqual(os.environ["ACESTEP_EXTERNAL_LM_PROVIDER"], "openai")

        self.assertIn("Main LM picker synced", status)
        self.assertNotIn("Session API key set via env", status)
        self.assertEqual(lm_update["value"], "external:openai:gpt-4o-mini")
        self.assertIn("external:openai:gpt-4o-mini", lm_update["choices"])
        self.assertEqual(api_key_update["value"], "")
        self.assertEqual(passphrase_update["value"], "")

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_persistence.discover_external_models"
    )
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_persistence.save_cached_external_models"
    )
    def test_fetch_models_updates_dropdown(self, _mock_save_cached, mock_discover) -> None:
        """Fetching models should update the model dropdown choices and selection."""

        mock_discover.return_value = ["gpt-4o-mini", "gpt-4.1-mini"]

        model_update, status = fetch_external_lm_models_from_ui(
            provider="openai",
            protocol="openai_chat",
            base_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            current_model="",
        )

        self.assertEqual(model_update["choices"], ["gpt-4o-mini", "gpt-4.1-mini"])
        self.assertEqual(model_update["value"], "gpt-4o-mini")
        self.assertIn("Discovered models: 2", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_defaults.load_cached_external_models"
    )
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_defaults.load_external_lm_runtime_settings"
    )
    def test_load_defaults_uses_cached_models_when_available(
        self,
        mock_load_settings,
        mock_load_cached,
    ) -> None:
        """Provider defaults should surface cached model choices when present."""

        mock_load_settings.return_value = None
        mock_load_cached.return_value = ["glm-5", "glm-4.7"]

        _protocol_update, model_update, _preset_update, _base_url_update, status = (
            load_external_lm_provider_defaults("zai")
        )

        self.assertEqual(model_update["choices"], ["glm-4.5-flash", "glm-5", "glm-4.7"])
        self.assertEqual(model_update["value"], "glm-4.5-flash")
        self.assertIn("Cached models available: 2", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_defaults.load_external_lm_runtime_settings"
    )
    def test_load_defaults_uses_saved_provider_settings(self, mock_load_settings) -> None:
        """Provider defaults should reuse the last saved settings for that provider."""

        mock_load_settings.return_value = {
            "provider": "ollama",
            "protocol": "openai_chat",
            "model": "llama3.1:8b-instruct",
            "base_url": "http://192.168.1.124:11434/v1/chat/completions",
        }

        protocol_update, model_update, preset_update, base_url_update, status = (
            load_external_lm_provider_defaults("ollama")
        )

        self.assertEqual(protocol_update["value"], "openai_chat")
        self.assertEqual(model_update["value"], "llama3.1:8b-instruct")
        self.assertEqual(base_url_update["value"], "http://192.168.1.124:11434/v1/chat/completions")
        self.assertEqual(preset_update["value"], "__custom__")
        self.assertIn("Default model: llama3.1:8b-instruct", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_defaults.load_external_lm_runtime_settings"
    )
    def test_load_defaults_includes_coding_endpoint_tip_and_presets(self, mock_load_settings) -> None:
        """Loading provider defaults should expose endpoint presets and guidance."""

        mock_load_settings.return_value = None
        protocol_update, model_update, preset_update, base_url_update, status = (
            load_external_lm_provider_defaults("zai")
        )

        self.assertEqual(protocol_update["value"], "openai_chat")
        self.assertEqual(model_update["value"], "glm-4.5-flash")
        self.assertEqual(
            preset_update["choices"][:2],
            [
                ("Standard chat endpoint", "https://api.z.ai/api/paas/v4/chat/completions"),
                ("Coding endpoint", "https://api.z.ai/api/coding/paas/v4/chat/completions"),
            ],
        )
        self.assertEqual(
            base_url_update["value"],
            "https://api.z.ai/api/paas/v4/chat/completions",
        )
        self.assertIn("coding model and coding endpoint", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_defaults.load_external_lm_runtime_settings"
    )
    def test_load_defaults_uses_small_ollama_general_model(self, mock_load_settings) -> None:
        """Ollama defaults should prefer a smaller general-purpose model."""

        mock_load_settings.return_value = None
        protocol_update, model_update, preset_update, base_url_update, status = (
            load_external_lm_provider_defaults("ollama")
        )

        self.assertEqual(protocol_update["value"], "openai_chat")
        self.assertEqual(model_update["value"], "qwen3:4b")
        self.assertEqual(
            base_url_update["value"],
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        self.assertEqual(
            preset_update["value"],
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        self.assertIn("Provider: Ollama", status)

    def test_save_syncs_picker_choice_without_llm_handler(self) -> None:
        """Saving should still surface the external choice when no LM handler is present."""

        with patch.dict(os.environ, {}, clear=True):
            status, _, _, lm_update = save_external_lm_settings_from_ui(
                provider="ollama",
                protocol="openai_chat",
                model="llama3.1:8b-instruct",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                api_key="",
                store_passphrase="",
                save_passphrase_to_keyring=False,
                llm_handler=None,
            )

        self.assertIn("Main LM picker synced", status)
        self.assertEqual(lm_update["value"], "external:ollama:llama3.1:8b-instruct")
        self.assertEqual(
            lm_update["choices"],
            ["external:ollama:llama3.1:8b-instruct"],
        )

    def test_base_url_preset_switches_to_coding_endpoint(self) -> None:
        """Selecting a preset should update the editable base URL value."""

        update = apply_external_lm_base_url_preset(
            provider="zai",
            preset_value="https://api.z.ai/api/coding/paas/v4/chat/completions",
            current_base_url="https://api.z.ai/api/paas/v4/chat/completions",
        )

        self.assertEqual(
            update["value"],
            "https://api.z.ai/api/coding/paas/v4/chat/completions",
        )

    def test_base_url_preset_custom_preserves_existing_value(self) -> None:
        """Selecting the custom preset should keep the current base URL."""

        update = apply_external_lm_base_url_preset(
            provider="zai",
            preset_value="__custom__",
            current_base_url="https://example.invalid/custom",
        )

        self.assertEqual(update["value"], "https://example.invalid/custom")

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_persistence.invalidate_cached_external_models"
    )
    def test_save_invalidates_provider_model_cache(self, mock_invalidate) -> None:
        """Saving settings should invalidate stale cached model lists for that provider."""

        with patch.dict(os.environ, {}, clear=True):
            save_external_lm_settings_from_ui(
                provider="ollama",
                protocol="openai_chat",
                model="qwen3:4b",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                api_key="",
                store_passphrase="",
                save_passphrase_to_keyring=False,
                llm_handler=None,
            )

        mock_invalidate.assert_called_once_with(provider="ollama")

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_persistence.resolve_external_api_key_for_runtime"
    )
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_persistence.discover_external_models"
    )
    def test_fetch_models_skips_credential_lookup_for_ollama(
        self,
        mock_discover,
        mock_resolve_api_key,
    ) -> None:
        """Ollama model discovery should not request credentials."""

        mock_discover.return_value = ["llama3.1:8b-instruct"]

        model_update, status = fetch_external_lm_models_from_ui(
            provider="ollama",
            protocol="openai_chat",
            base_url="http://127.0.0.1:11434/v1/chat/completions",
            api_key="",
            current_model="",
        )

        mock_resolve_api_key.assert_not_called()
        self.assertEqual(model_update["value"], "llama3.1:8b-instruct")
        self.assertIn("Provider: Ollama", status)

    def test_dropdown_sync_updates_provider_specific_base_url(self) -> None:
        """Main LM dropdown sync should refresh provider-specific URL controls."""

        with patch.dict(os.environ, {}, clear=True):
            status_updates = build_external_lm_dropdown_sync_updates(
                "external:openai:gpt-4o-mini"
            )

        provider_update, protocol_update, model_update, preset_update, base_url_update, status = (
            status_updates
        )
        self.assertEqual(provider_update["value"], "openai")
        self.assertEqual(protocol_update["value"], "openai_chat")
        self.assertEqual(model_update["value"], "gpt-4o-mini")
        self.assertEqual(base_url_update["value"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(
            preset_update["value"],
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertIn("Provider: OpenAI", status)

    def test_inactive_updates_clear_external_controls(self) -> None:
        """Switching back to a local LM should clear the external setup panel."""

        provider_update, protocol_update, model_update, preset_update, base_url_update, status = (
            build_external_lm_inactive_updates()
        )

        self.assertIsNone(provider_update["value"])
        self.assertIsNone(protocol_update["value"])
        self.assertEqual(model_update["choices"], [])
        self.assertEqual(preset_update["choices"], [])
        self.assertEqual(base_url_update["value"], "")
        self.assertIn("External LM inactive", status)


if __name__ == "__main__":
    unittest.main()
