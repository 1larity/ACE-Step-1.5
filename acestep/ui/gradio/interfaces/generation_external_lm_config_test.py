"""Tests for external LLM config initialization behavior."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from acestep.ui.gradio.interfaces import generation_external_lm_config


class ExternalLmConfigTests(unittest.TestCase):
    """Validate provider resolution and scoped env hydration for external LM config."""

    def test_resolve_initial_provider_prefers_init_params_external_model_path(self) -> None:
        """An explicit external lm_model_path should win over hydrated env defaults."""
        with patch.dict(
            os.environ,
            {
                "ACESTEP_EXTERNAL_LM_PROVIDER": "zai",
            },
            clear=False,
        ):
            provider = generation_external_lm_config._resolve_initial_provider(
                {"lm_model_path": "external:ollama:qwen3:8b"}
            )

        self.assertEqual("ollama", provider)

    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.get_external_provider_profile")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Accordion")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Markdown")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Row")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Column")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Dropdown")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Textbox")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Checkbox")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.gr.Button")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.get_external_provider_choices", return_value=[("Z.ai", "zai"), ("Ollama", "ollama")])
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.get_active_external_lm_model", return_value="fallback-model")
    @patch("acestep.ui.gradio.interfaces.generation_external_lm_config.hydrate_external_lm_env_from_store")
    def test_create_external_lm_config_hydrates_selected_provider(
        self,
        hydrate_mock,
        _active_model_mock,
        _choices_mock,
        button_mock,
        checkbox_mock,
        textbox_mock,
        dropdown_mock,
        column_mock,
        row_mock,
        markdown_mock,
        accordion_mock,
        profile_mock,
    ) -> None:
        """Hydration should be scoped to the provider implied by init_params."""
        class _Ctx:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        accordion_mock.return_value = _Ctx()
        row_mock.return_value = _Ctx()
        column_mock.return_value = _Ctx()
        dropdown_mock.return_value = object()
        textbox_mock.return_value = object()
        checkbox_mock.return_value = object()
        button_mock.return_value = object()
        markdown_mock.return_value = object()
        profile_mock.return_value = type(
            'Profile',
            (),
            {
                'protocol': 'openai_chat',
                'default_model': 'default-ollama',
                'default_base_url': 'http://127.0.0.1:11434/v1/chat/completions',
                'base_url_presets': (
                    ("Local Ollama", "http://127.0.0.1:11434/v1/chat/completions"),
                ),
            },
        )()

        with patch.dict(os.environ, {}, clear=True):
            generation_external_lm_config.create_external_lm_config_content(
                {"lm_model_path": "external:ollama:qwen3:8b"}
            )

        hydrate_mock.assert_called_once_with("ollama")
        profile_mock.assert_called_once_with("ollama")
        accordion_mock.assert_called_once_with(
            "🧠 External LM",
            open=False,
            elem_classes=["has-info-container"],
        )
        self.assertIn(
            unittest.mock.call(
                label="Base URL",
                choices=[
                    ("Local Ollama", "http://127.0.0.1:11434/v1/chat/completions"),
                ],
                value="http://127.0.0.1:11434/v1/chat/completions",
                allow_custom_value=True,
                info="Choose a provider preset or type a custom chat endpoint URL.",
                elem_classes=["has-info-container"],
            ),
            dropdown_mock.call_args_list,
        )


if __name__ == "__main__":
    unittest.main()
