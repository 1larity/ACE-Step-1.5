"""Unit tests for service_init.init_service_wrapper device handling."""

import unittest
from unittest.mock import MagicMock, patch


class InitServiceWrapperDeviceResolutionTests(unittest.TestCase):
    """Verify device handling for LLM initialization."""

    def _import_module(self):
        """Import service_init lazily to avoid heavy transitive imports."""
        from acestep.ui.gradio.events.generation import service_init

        return service_init

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    def test_reinit_without_llm_preserves_resolved_device(self, mock_gpu_config):
        """It does not overwrite a resolved LLM device when init_llm is false."""
        module = self._import_module()

        mock_gpu_config.return_value = MagicMock(
            available_lm_models=["acestep-5Hz-lm-1.7B"],
            lm_backend_restriction=None,
            tier="tier6",
            gpu_memory_gb=24.0,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=8,
        )

        dit_handler = MagicMock()
        dit_handler.initialize_service.return_value = ("ok", True)
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.device = "cuda"

        module.init_service_wrapper(
            dit_handler,
            llm_handler,
            "/some/project/checkpoints",
            "acestep-v15-turbo",
            "auto",
            False,
            None,
            "vllm",
            use_flash_attention=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
        )

        llm_handler.initialize.assert_not_called()
        self.assertEqual(llm_handler.device, "cuda")

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    def test_init_llm_with_auto_device_calls_initialize(self, mock_gpu_config):
        """It passes the raw auto device through to llm_handler.initialize."""
        module = self._import_module()

        mock_gpu_config.return_value = MagicMock(
            available_lm_models=["acestep-5Hz-lm-1.7B"],
            lm_backend_restriction=None,
            tier="tier6",
            gpu_memory_gb=24.0,
            max_duration_with_lm=600,
            max_duration_without_lm=600,
            max_batch_size_with_lm=4,
            max_batch_size_without_lm=8,
        )

        dit_handler = MagicMock()
        dit_handler.initialize_service.return_value = ("ok", True)
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True

        llm_handler = MagicMock()
        llm_handler.llm_initialized = False
        llm_handler.initialize.return_value = ("LLM initialized", True)

        module.init_service_wrapper(
            dit_handler,
            llm_handler,
            "/some/project/checkpoints",
            "acestep-v15-turbo",
            "auto",
            True,
            "acestep-5Hz-lm-1.7B",
            "pt",
            use_flash_attention=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
        )

        llm_handler.initialize.assert_called_once()
        _, call_kwargs = llm_handler.initialize.call_args
        self.assertEqual(call_kwargs.get("device"), "auto")


if __name__ == "__main__":
    unittest.main()
