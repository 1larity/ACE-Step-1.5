"""Tests for external LM setup default/model discovery helpers."""

from __future__ import annotations

import unittest

from acestep.ui.gradio.events.generation.external_lm_setup_defaults import fetch_models_data


class ExternalLmSetupDefaultsTests(unittest.TestCase):
    """Validate model selection fallback behavior for external LM discovery."""

    def test_fetch_models_data_handles_empty_discovery_results(self) -> None:
        """Empty discovery responses should not raise or index into an empty list."""
        profile = type(
            "Profile",
            (),
            {
                "provider_id": "ollama",
                "label": "Ollama",
                "protocol": "openai_chat",
                "default_base_url": "http://127.0.0.1:11434/v1/chat/completions",
                "api_key_required": False,
            },
        )()

        models, selected, status = fetch_models_data(
            provider="ollama",
            protocol="openai_chat",
            base_url="http://127.0.0.1:11434/v1/chat/completions",
            api_key="",
            current_model="",
            get_external_provider_profile=lambda _provider: profile,
            resolve_external_api_key_for_runtime=lambda _provider: "",
            discover_external_models=lambda **_kwargs: [],
        )

        self.assertEqual([], models)
        self.assertEqual("", selected)
        self.assertIn("Discovered models: 0", status)
        self.assertIn("Selected: (none)", status)


if __name__ == "__main__":
    unittest.main()
