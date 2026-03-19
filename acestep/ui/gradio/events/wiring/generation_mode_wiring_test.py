"""Contract tests for generation mode wiring."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


_WIRING_PATH = Path(__file__).resolve().parent / "generation_mode_wiring.py"


class GenerationModeWiringTests(unittest.TestCase):
    """Verify key repaint synchronization hooks remain registered."""

    def _load_register_generation_mode_handlers(self) -> ast.FunctionDef:
        """Return the parsed register function from the mode wiring module."""

        module = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"))
        return next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_generation_mode_handlers"
        )

    def test_repaint_mode_change_is_wired(self) -> None:
        """Repaint mode changes should continue to drive repaint strength updates."""

        register_fn = self._load_register_generation_mode_handlers()

        found_repaint_mode_change = False
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
                and target.slice.value == "repaint_mode"
            ):
                found_repaint_mode_change = True
                break

        self.assertTrue(found_repaint_mode_change)

    def test_repaint_strength_change_is_wired(self) -> None:
        """Repaint strength changes should continue to auto-switch repaint mode."""

        register_fn = self._load_register_generation_mode_handlers()

        found_repaint_strength_change = False
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
                and target.slice.value == "repaint_strength"
            ):
                found_repaint_strength_change = True
                break

        self.assertTrue(found_repaint_strength_change)


if __name__ == "__main__":
    unittest.main()
