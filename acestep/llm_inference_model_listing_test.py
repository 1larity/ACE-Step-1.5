"""Focused tests for local 5Hz LM model listing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from acestep.llm_inference import LLMHandler


class LlmInferenceModelListingTests(unittest.TestCase):
    """Verify local LM model discovery excludes external provider tokens."""

    def test_available_5hz_models_only_returns_local_checkpoint_names(self) -> None:
        """External dropdown tokens should not be exposed as local checkpoint models."""

        handler = LLMHandler()

        with patch.dict(
            os.environ,
            {"ACESTEP_EXTERNAL_LM_CHOICES": "external:openai:gpt-4o-mini"},
            clear=True,
        ), patch.object(
            handler,
            "_get_checkpoint_dir",
            return_value="/tmp/fake-checkpoints",
        ), patch(
            "acestep.llm_inference.os.path.exists",
            return_value=True,
        ), patch(
            "acestep.llm_inference.os.listdir",
            return_value=["acestep-5Hz-lm-1.7B"],
        ), patch(
            "acestep.llm_inference.os.path.isdir",
            return_value=True,
        ):
            self.assertEqual(
                handler.get_available_5hz_lm_models(),
                ["acestep-5Hz-lm-1.7B"],
            )


if __name__ == "__main__":
    unittest.main()
