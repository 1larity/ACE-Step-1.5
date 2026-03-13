"""Unit tests for service_init quantization selection helpers."""

import importlib
import unittest
from unittest.mock import patch


class QuantizationSelectionTests(unittest.TestCase):
    """Verify pre-Ampere quantization mode selection."""

    def _import_module(self):
        """Import service_init lazily to avoid heavy transitive imports."""
        from acestep.ui.gradio.events.generation import service_init

        return service_init

    def test_select_quantization_value_uses_dynamic_mode_for_pre_ampere_cuda(self):
        """It selects w8a8_dynamic for pre-Ampere CUDA devices."""
        module = self._import_module()

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(6, 1)
        ):
            self.assertEqual(
                module._select_quantization_value(
                    quantization_enabled=True,
                    device="cuda",
                ),
                "w8a8_dynamic",
            )

    def test_select_quantization_value_keeps_default_when_torch_import_fails(self):
        """It keeps the default quantization when torch cannot be imported."""
        module = importlib.import_module(
            "acestep.ui.gradio.events.generation.service_init_quantization"
        )
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Raise ImportError only for torch imports from the helper."""
            if name == "torch":
                raise ImportError("torch missing")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(
                module._select_quantization_value(
                    quantization_enabled=True,
                    device="cuda",
                ),
                "int8_weight_only",
            )


if __name__ == "__main__":
    unittest.main()
