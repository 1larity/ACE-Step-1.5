"""Tests for external LLM setup Gradio action handlers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.text_tasks.secure_secret_store import SecretStoreError
from acestep.ui.gradio.events.generation.external_lm_setup_actions import (
    check_external_lm_runtime_from_ui,
    fetch_external_lm_models_from_ui,
    fetch_external_lm_models_from_ui_with_lm_dropdown,
    load_external_lm_provider_defaults,
    load_external_lm_provider_defaults_with_lm_dropdown,
    save_external_lm_settings_from_ui,
)


class ExternalLmSetupActionsTests(unittest.TestCase):
    """Validate external LLM setup tab action behavior."""

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.load_external_lm_runtime_settings_for_provider",
        return_value=None,
    )
    def test_load_provider_defaults_populates_protocol_model_and_base_url(self, _load_saved_mock) -> None:
        """Selecting provider defaults should update protocol/model/base URL controls."""
        protocol_update, model_update, base_url_update, status = load_external_lm_provider_defaults(
            "openai"
        )

        self.assertEqual("openai_chat", protocol_update.get("value"))
        self.assertEqual("gpt-4o-mini", model_update.get("value"))
        self.assertEqual(["gpt-4o-mini"], model_update.get("choices"))
        self.assertEqual("https://api.openai.com/v1/chat/completions", base_url_update.get("value"))
        self.assertIn(
            ("OpenAI chat completions", "https://api.openai.com/v1/chat/completions"),
            base_url_update.get("choices"),
        )
        self.assertIn("Provider: OpenAI", status)
        self.assertNotIn("Coding Plan tip", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.load_external_lm_runtime_settings_for_provider",
        return_value={
            "provider": "ollama",
            "protocol": "openai_chat",
            "model": "qwen2.5:14b",
            "base_url": "http://127.0.0.1:11434/v1/chat/completions",
        },
    )
    def test_load_provider_defaults_prefers_saved_provider_preferences(
        self,
        _load_saved_mock,
    ) -> None:
        """Selecting a provider should reuse previously saved non-secret provider settings."""
        protocol_update, model_update, base_url_update, status = load_external_lm_provider_defaults(
            "ollama"
        )

        self.assertEqual("openai_chat", protocol_update.get("value"))
        self.assertEqual("qwen2.5:14b", model_update.get("value"))
        self.assertEqual(
            ["qwen2.5:14b", "qwen3:4b"],
            model_update.get("choices"),
        )
        self.assertEqual(
            "http://127.0.0.1:11434/v1/chat/completions",
            base_url_update.get("value"),
        )
        self.assertIn(
            ("Local Ollama", "http://127.0.0.1:11434/v1/chat/completions"),
            base_url_update.get("choices"),
        )
        self.assertIn("Loaded saved provider preferences.", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.load_external_lm_runtime_settings_for_provider",
        return_value=None,
    )
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.get_external_lm_choices", return_value=["external:zai:glm-5"])
    def test_load_provider_defaults_with_dropdown_keeps_saved_entries_visible(
        self,
        _choices_mock,
        _load_saved_mock,
    ) -> None:
        """Loading provider defaults should preview the provider in service config without dropping existing entries."""
        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = ["acestep-5Hz-lm-1.7B"]

        _, _, _, _, lm_model_update = load_external_lm_provider_defaults_with_lm_dropdown(
            "ollama",
            llm_handler=llm_handler,
        )

        self.assertEqual(
            [
                "acestep-5Hz-lm-1.7B",
                "external:zai:glm-5",
                "external:ollama:qwen3:4b",
            ],
            lm_model_update.get("choices"),
        )
        self.assertNotIn("value", lm_model_update)

    def test_load_zai_defaults_includes_coding_plan_tip(self) -> None:
        """Z.ai defaults status should include guidance for coding-plan endpoint usage."""
        _, _, _, status = load_external_lm_provider_defaults("zai")
        self.assertIn("Coding Plan tip", status)
        self.assertIn("api/coding/paas/v4/chat/completions", status)

    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.discover_external_models",
        return_value=["gpt-4o-mini", "gpt-4.1-mini"],
    )
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    def test_fetch_models_populates_dropdown_choices(
        self,
        _info_mock,
        discover_mock,
    ) -> None:
        """Fetch models action should update model dropdown choices/value from API results."""
        model_update, status = fetch_external_lm_models_from_ui(
            provider="openai",
            protocol="openai_chat",
            base_url="https://api.openai.com/v1/chat/completions",
            api_key="sk-test",
            current_model="",
        )
        self.assertEqual(["gpt-4o-mini", "gpt-4.1-mini"], model_update.get("choices"))
        self.assertEqual("gpt-4o-mini", model_update.get("value"))
        self.assertIn("Discovered models: 2", status)
        discover_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.get_external_lm_choices", return_value=["external:zai:glm-5"])
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.discover_external_models",
        return_value=["qwen2.5:14b", "llama3.2:latest"],
    )
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    def test_fetch_models_with_dropdown_previews_new_external_model_without_save(
        self,
        _info_mock,
        discover_mock,
        _choices_mock,
    ) -> None:
        """Fetched provider models should appear in the service dropdown before save."""
        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = ["acestep-5Hz-lm-1.7B"]

        model_update, status, lm_model_update = fetch_external_lm_models_from_ui_with_lm_dropdown(
            provider="ollama",
            protocol="openai_chat",
            base_url="http://127.0.0.1:11434/v1/chat/completions",
            api_key="",
            current_model="",
            llm_handler=llm_handler,
        )

        self.assertEqual(["qwen2.5:14b", "llama3.2:latest"], model_update.get("choices"))
        self.assertEqual("qwen2.5:14b", model_update.get("value"))
        self.assertIn("Discovered models: 2", status)
        self.assertEqual(
            [
                "acestep-5Hz-lm-1.7B",
                "external:zai:glm-5",
                "external:ollama:qwen2.5:14b",
            ],
            lm_model_update.get("choices"),
        )
        discover_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Warning")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.resolve_external_api_key_for_runtime",
        side_effect=SecretStoreError("missing"),
    )
    def test_save_settings_requires_key_for_required_provider(
        self,
        _resolve_key_mock,
        warning_mock,
    ) -> None:
        """Save should fail with clear status when required provider has no credentials."""
        with patch.dict("os.environ", {}, clear=True):
            status, _, _, _ = save_external_lm_settings_from_ui(
                provider="openai",
                protocol="openai_chat",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                api_key="",
                store_passphrase="",
                save_passphrase_to_keyring=True,
            )
            self.assertIsNone(os.getenv("ACESTEP_EXTERNAL_LM_ENABLED"))
        self.assertIn("OpenAI API key is required", status)
        warning_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.store_runtime_passphrase")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.EncryptedSecretStore")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.save_external_lm_runtime_settings"
    )
    def test_save_settings_success_clears_sensitive_inputs_and_updates_lm_dropdown(
        self,
        save_runtime_settings_mock,
        store_cls_mock,
        store_runtime_passphrase_mock,
        info_mock,
    ) -> None:
        """Successful save should clear sensitive inputs and update LM dropdown token."""
        store_instance = MagicMock()
        store_instance.secret_path = Path("/tmp/openai_api_key.enc")
        store_cls_mock.resolve_existing_default_path.return_value = Path("/tmp/openai_api_key.enc")
        store_cls_mock.return_value = store_instance
        store_runtime_passphrase_mock.return_value = (True, "stored in keyring")
        save_runtime_settings_mock.return_value = Path("/tmp/external_lm_runtime.json")

        with patch.dict("os.environ", {}, clear=True):
            status, api_update, pass_update, lm_model_update = save_external_lm_settings_from_ui(
                provider="openai",
                protocol="openai_chat",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                api_key="sk-test",
                store_passphrase="pass",
                save_passphrase_to_keyring=True,
            )
            self.assertEqual("sk-test", os.getenv("ACESTEP_OPENAI_API_KEY"))
            self.assertEqual("true", os.getenv("ACESTEP_EXTERNAL_LM_ENABLED"))

        self.assertIn("Encrypted API key stored at:", status)
        self.assertEqual("", api_update.get("value"))
        self.assertEqual("", pass_update.get("value"))
        self.assertEqual("external:openai:gpt-4o-mini", lm_model_update.get("value"))
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.save_external_lm_runtime_settings"
    )
    def test_save_settings_rebuilds_lm_dropdown_choices_with_external_entry(
        self,
        save_runtime_settings_mock,
        _info_mock,
    ) -> None:
        """Save should return LM choices containing local 5Hz models plus configured external token."""
        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = [
            "acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B",
        ]
        save_runtime_settings_mock.return_value = Path("/tmp/external_lm_runtime.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                _, _, _, lm_model_update = save_external_lm_settings_from_ui(
                    provider="zai",
                    protocol="openai_chat",
                    model="glm-5",
                    base_url="https://api.z.ai/api/paas/v4/chat/completions",
                    api_key="sk-test",
                    store_passphrase="",
                    save_passphrase_to_keyring=False,
                    llm_handler=llm_handler,
                )

        self.assertEqual("external:zai:glm-5", lm_model_update.get("value"))
        self.assertEqual(
            [
                "acestep-5Hz-lm-0.6B",
                "acestep-5Hz-lm-1.7B",
                "external:zai:glm-5",
            ],
            lm_model_update.get("choices"),
        )

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.save_external_lm_runtime_settings"
    )
    def test_save_zai_general_endpoint_includes_coding_plan_tip(
        self,
        save_runtime_settings_mock,
        _info_mock,
    ) -> None:
        """Saving Z.ai with general endpoint should include coding-plan guidance in status."""
        save_runtime_settings_mock.return_value = Path("/tmp/external_lm_runtime.json")
        with patch.dict("os.environ", {}, clear=True):
            status, _, _, _ = save_external_lm_settings_from_ui(
                provider="zai",
                protocol="openai_chat",
                model="glm-5",
                base_url="https://api.z.ai/api/paas/v4/chat/completions",
                api_key="sk-test",
                store_passphrase="",
                save_passphrase_to_keyring=False,
            )

        self.assertIn("Coding Plan tip", status)
        self.assertIn("api/coding/paas/v4/chat/completions", status)

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions._python_keyring_available")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions._secret_tool_available")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.EncryptedSecretStore")
    def test_runtime_doctor_reports_not_ready_without_required_credentials(
        self,
        store_cls_mock,
        secret_tool_available_mock,
        keyring_available_mock,
        warning_mock,
    ) -> None:
        """Runtime doctor should report not-ready for required provider without credentials."""
        store_instance = MagicMock()
        store_instance.secret_path = Path("/tmp/openai_api_key.enc")
        store_instance.exists.return_value = False
        store_cls_mock.resolve_existing_default_path.return_value = Path("/tmp/openai_api_key.enc")
        store_cls_mock.return_value = store_instance
        secret_tool_available_mock.return_value = False
        keyring_available_mock.return_value = False

        with patch.dict("os.environ", {}, clear=True):
            status = check_external_lm_runtime_from_ui("openai")

        self.assertIn("External runtime status: not ready", status)
        self.assertIn("External LM mode enabled: no", status)
        self.assertIn("secret-tool available: no", status)
        self.assertIn("python keyring available: no", status)
        warning_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.EncryptedSecretStore")
    def test_runtime_doctor_flags_ui_model_mismatch(
        self,
        store_cls_mock,
        _info_mock,
    ) -> None:
        """Doctor should flag when UI model differs from saved runtime model env."""
        store_instance = MagicMock()
        store_instance.secret_path = Path("/tmp/zai_api_key.enc")
        store_instance.exists.return_value = True
        store_instance.load.return_value = "key"
        store_cls_mock.resolve_existing_default_path.return_value = Path("/tmp/zai_api_key.enc")
        store_cls_mock.return_value = store_instance

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_LM_MODEL": "glm-4.5-flash",
                "ACESTEP_EXTERNAL_LM_ENABLED": "true",
                "ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE": "pass",
            },
            clear=True,
        ):
            status = check_external_lm_runtime_from_ui(
                "zai",
                "openai_chat",
                "glm-5",
                "https://api.z.ai/api/paas/v4/chat/completions",
            )

        self.assertIn("Configured model: glm-5", status)
        self.assertIn("External LM mode enabled: yes", status)
        self.assertIn("UI model differs from saved runtime model", status)

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.save_external_lm_runtime_settings"
    )
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.resolve_external_api_key_for_runtime",
        return_value="stored-secret",
    )
    def test_save_settings_clears_stale_session_key_and_passphrase_when_inputs_blank(
        self,
        _resolve_key_mock,
        save_runtime_settings_mock,
        _info_mock,
    ) -> None:
        """Saving with blank inputs should clear stale session key and passphrase env vars."""
        save_runtime_settings_mock.return_value = Path("/tmp/external_lm_runtime.json")

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_OPENAI_API_KEY": "stale-key",
                "ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE": "stale-pass",
            },
            clear=True,
        ):
            status, _, _, _ = save_external_lm_settings_from_ui(
                provider="openai",
                protocol="openai_chat",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1/chat/completions",
                api_key="",
                store_passphrase="",
                save_passphrase_to_keyring=False,
            )

            self.assertIsNone(os.getenv("ACESTEP_OPENAI_API_KEY"))
            self.assertIsNone(os.getenv("ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE"))

        self.assertIn("Cleared session API key env: ACESTEP_OPENAI_API_KEY", status)
        self.assertIn("Cleared session encrypted-store passphrase.", status)

    @patch("acestep.ui.gradio.events.generation.external_lm_setup_actions.gr.Info")
    @patch(
        "acestep.ui.gradio.events.generation.external_lm_setup_actions.save_external_lm_runtime_settings"
    )
    def test_save_settings_clears_stale_zai_runtime_env_when_switching_provider(
        self,
        save_runtime_settings_mock,
        _info_mock,
    ) -> None:
        """Saving a non-Z.ai provider should clear stale Z.ai runtime model/base URL env vars."""
        save_runtime_settings_mock.return_value = Path("/tmp/external_lm_runtime.json")

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_ZAI_MODEL": "glm-5",
                "ACESTEP_ZAI_BASE_URL": "https://api.z.ai/api/paas/v4/chat/completions",
            },
            clear=True,
        ):
            status, _, _, _ = save_external_lm_settings_from_ui(
                provider="ollama",
                protocol="openai_chat",
                model="qwen3:8b",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                api_key="",
                store_passphrase="",
                save_passphrase_to_keyring=False,
            )

            self.assertIsNone(os.getenv("ACESTEP_ZAI_MODEL"))
            self.assertIsNone(os.getenv("ACESTEP_ZAI_BASE_URL"))

        self.assertIn("Cleared stale Z.ai runtime env", status)


if __name__ == "__main__":
    unittest.main()
