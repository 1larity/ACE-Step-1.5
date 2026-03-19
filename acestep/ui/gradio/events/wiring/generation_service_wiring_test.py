"""Contract tests for generation service wiring."""

import ast
from pathlib import Path
import unittest


_WIRING_PATH = Path(__file__).resolve().parent / "generation_service_wiring.py"


class GenerationServiceWiringTests(unittest.TestCase):
    """Verify key event hooks are present in generation service wiring."""

    def _load_register_generation_service_handlers(self) -> ast.FunctionDef:
        """Return the parsed register function from the wiring module."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        return next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_service_handlers"
        )

    def test_registers_language_dropdown_change_handler(self):
        """Service wiring should attach a change handler for language dropdown."""

        register_fn = self._load_register_generation_service_handlers()

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

    def test_lm_picker_change_updates_init_llm_checkbox(self):
        """LM picker changes should drive the Init 5Hz LM checkbox."""

        register_fn = self._load_register_generation_service_handlers()

        found_init_checkbox_output = False
        for node in ast.walk(register_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "change":
                continue
            if not isinstance(node.func.value, ast.Subscript):
                continue
            target = node.func.value
            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == "generation_section"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "lm_model_path"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "outputs" or not isinstance(keyword.value, ast.List):
                    continue
                for element in keyword.value.elts:
                    if not isinstance(element, ast.Subscript):
                        continue
                    if not (
                        isinstance(element.value, ast.Name)
                        and element.value.id == "generation_section"
                        and isinstance(element.slice, ast.Constant)
                        and element.slice.value == "init_llm_checkbox"
                    ):
                        continue
                    found_init_checkbox_output = True
                    break

        self.assertTrue(
            found_init_checkbox_output,
            "lm_model_path.change should output to init_llm_checkbox",
        )

    def test_language_runtime_helper_exists(self):
        """Runtime language helper should exist for dropdown change wiring."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_apply_runtime_language", function_names)


if __name__ == "__main__":
    unittest.main()
