"""Contract tests for generation service wiring."""

import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.wiring import generation_service_wiring as wiring


_WIRING_PATH = Path(__file__).resolve().parent / "generation_service_wiring.py"


class GenerationServiceWiringTests(unittest.TestCase):
    """Verify key event hooks are present in generation service wiring."""

    def test_registers_language_dropdown_change_handler(self):
        """Service wiring should attach a change handler for language dropdown."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_service_handlers"
        )

        found_language_change = False
        for node in ast.walk(register_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "change":
                continue
            if not isinstance(node.func.value, ast.Subscript):
                continue
            target = node.func.value
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "generation_section"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "language_dropdown"
            ):
                found_language_change = True
                break

        self.assertTrue(found_language_change, "language_dropdown.change handler was not found")

    def test_language_runtime_helper_exists(self):
        """Runtime language helper should exist for dropdown change wiring."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_apply_runtime_language", function_names)

    def test_registers_lm_model_path_change_handler(self):
        """Service wiring should attach a change handler for lm_model_path dropdown."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_service_handlers"
        )

        found_lm_model_change = False
        for node in ast.walk(register_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "change":
                continue
            if not isinstance(node.func.value, ast.Subscript):
                continue
            target = node.func.value
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "generation_section"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "lm_model_path"
            ):
                found_lm_model_change = True
                break

        self.assertTrue(found_lm_model_change, "lm_model_path.change handler was not found")

    def test_registers_external_lm_save_click_handler(self):
        """Service wiring should attach click handler for external settings save action."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        register_fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_service_handlers"
        )

        found_save_click = False
        for node in ast.walk(register_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "click":
                continue
            if not isinstance(node.func.value, ast.Subscript):
                continue
            target = node.func.value
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "generation_section"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "external_lm_save_btn"
            ):
                found_save_click = True
                break

        self.assertTrue(found_save_click, "external_lm_save_btn.click handler was not found")

    @patch("acestep.ui.gradio.events.wiring.generation_service_wiring.activate_external_lm_mode")
    def test_external_lm_selection_unchecks_local_init(
        self,
        activate_mock,
    ):
        """Selecting an external LM should turn off local 5Hz initialization."""
        update = wiring._sync_external_lm_mode_from_dropdown("external:ollama:qwen2.5:14b")

        activate_mock.assert_called_once_with("external:ollama:qwen2.5:14b")
        self.assertEqual(False, update.get("value"))

    @patch("acestep.ui.gradio.events.wiring.generation_service_wiring.deactivate_external_lm_mode")
    def test_local_lm_selection_checks_local_init(
        self,
        deactivate_mock,
    ):
        """Selecting a local LM should turn local 5Hz initialization back on."""
        update = wiring._sync_external_lm_mode_from_dropdown("acestep-5Hz-lm-1.7B")

        deactivate_mock.assert_called_once()
        self.assertEqual(True, update.get("value"))

    @patch("acestep.ui.gradio.events.wiring.generation_service_wiring.deactivate_external_lm_mode")
    def test_init_checkbox_switches_external_selection_back_to_local_model(
        self,
        deactivate_mock,
    ):
        """Checking 5Hz init should replace external selection with a local LM when available."""
        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = ["acestep-5Hz-lm-1.7B"]

        checkbox_update, model_update = wiring._sync_lm_selection_from_init_checkbox(
            True,
            "external:zai:glm-5",
            llm_handler=llm_handler,
        )

        deactivate_mock.assert_called_once()
        self.assertEqual(True, checkbox_update.get("value"))
        self.assertEqual("acestep-5Hz-lm-1.7B", model_update.get("value"))

    def test_init_checkbox_reverts_when_no_local_model_is_available(self):
        """Checking 5Hz init should revert when no local LM can back that choice."""
        llm_handler = MagicMock()
        llm_handler.get_available_5hz_lm_models.return_value = []

        checkbox_update, model_update = wiring._sync_lm_selection_from_init_checkbox(
            True,
            "external:openai:gpt-4o-mini",
            llm_handler=llm_handler,
        )

        self.assertEqual(False, checkbox_update.get("value"))
        self.assertNotIn("value", model_update)


if __name__ == "__main__":
    unittest.main()
