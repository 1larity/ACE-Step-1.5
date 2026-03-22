"""Tests for advanced settings section assembly."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.interfaces import generation_advanced_settings


class AdvancedSettingsSectionTests(unittest.TestCase):
    """Verify settings-section assembly for generation UI."""

    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.build_automation_controls")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.build_output_controls")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.build_lm_controls")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.build_dit_controls")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.build_lora_controls")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.create_service_config_content")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.get_ui_control_config")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.compute_init_defaults")
    @patch("acestep.ui.gradio.interfaces.generation_advanced_settings.gr.Accordion")
    def test_create_advanced_settings_section_includes_external_lm_components(
        self,
        accordion_mock,
        defaults_mock,
        ui_config_mock,
        service_mock,
        lora_mock,
        dit_mock,
        lm_mock,
        output_mock,
        automation_mock,
    ) -> None:
        """Service-owned External-LM controls should still reach the merged settings map."""

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        accordion_mock.return_value = _Ctx()
        defaults_mock.return_value = {
            "service_pre_initialized": False,
            "service_mode": False,
        }
        ui_config_mock.return_value = object()
        external_component = object()
        service_mock.return_value = {
            "service_key": object(),
            "external_lm_provider_dropdown": external_component,
        }
        lora_mock.return_value = {}
        dit_mock.return_value = {}
        lm_mock.return_value = {}
        output_mock.return_value = {}
        automation_mock.return_value = {}

        result = generation_advanced_settings.create_advanced_settings_section(
            dit_handler=MagicMock(),
            llm_handler=MagicMock(),
            init_params={"lm_model_path": "external:ollama:qwen3:4b"},
        )

        self.assertIs(
            external_component,
            result["external_lm_provider_dropdown"],
        )
        service_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
