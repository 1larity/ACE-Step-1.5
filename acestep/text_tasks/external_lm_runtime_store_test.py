"""Tests for persisted external LM runtime settings storage."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks.external_lm_runtime_store import (
    external_lm_settings_path,
    hydrate_external_lm_env_from_store,
    load_external_lm_runtime_settings,
    load_external_lm_runtime_settings_for_provider,
    save_external_lm_runtime_settings,
)


class ExternalLmRuntimeStoreTests(unittest.TestCase):
    """Verify save/load/hydrate behavior for external LM runtime settings."""

    def test_save_and_load_round_trip(self) -> None:
        """Saved runtime settings should round-trip through JSON persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                saved_path = save_external_lm_runtime_settings(
                    provider="zai",
                    protocol="openai_chat",
                    model="glm-5",
                    base_url="https://api.z.ai/api/coding/paas/v4/chat/completions",
                )
                self.assertEqual(
                    Path(tmpdir) / "acestep" / "config" / "external_lm_runtime.json",
                    saved_path,
                )
                loaded = load_external_lm_runtime_settings()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("zai", loaded["provider"])
        self.assertEqual("glm-5", loaded["model"])

    def test_save_tracks_provider_specific_preferences(self) -> None:
        """Saving multiple providers should preserve one non-secret config per provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                save_external_lm_runtime_settings(
                    provider="zai",
                    protocol="openai_chat",
                    model="glm-5",
                    base_url="https://api.z.ai/api/coding/paas/v4/chat/completions",
                )
                save_external_lm_runtime_settings(
                    provider="ollama",
                    protocol="openai_chat",
                    model="qwen2.5:14b",
                    base_url="http://127.0.0.1:11434/v1/chat/completions",
                )

                active = load_external_lm_runtime_settings()
                zai = load_external_lm_runtime_settings_for_provider("zai")
                ollama = load_external_lm_runtime_settings_for_provider("ollama")

        self.assertEqual("ollama", active["provider"])
        self.assertEqual("qwen2.5:14b", active["model"])
        self.assertEqual("glm-5", zai["model"])
        self.assertEqual("qwen2.5:14b", ollama["model"])

    def test_save_migrates_legacy_active_config_into_provider_map(self) -> None:
        """Saving a new provider should retain older single-config runtime settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                path = external_lm_settings_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "provider": "zai",
                            "protocol": "openai_chat",
                            "model": "glm-5",
                            "base_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
                        },
                        ensure_ascii=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                save_external_lm_runtime_settings(
                    provider="ollama",
                    protocol="openai_chat",
                    model="qwen2.5:14b",
                    base_url="http://127.0.0.1:11434/v1/chat/completions",
                )

                zai = load_external_lm_runtime_settings_for_provider("zai")
                ollama = load_external_lm_runtime_settings_for_provider("ollama")

        self.assertEqual("glm-5", zai["model"])
        self.assertEqual("qwen2.5:14b", ollama["model"])

    def test_load_provider_settings_supports_legacy_single_config_payload(self) -> None:
        """Provider-specific load should work with the older single-config JSON shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                path = external_lm_settings_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "provider": "ollama",
                            "protocol": "openai_chat",
                            "model": "llama3.1:8b",
                            "base_url": "http://127.0.0.1:11434/v1/chat/completions",
                        },
                        ensure_ascii=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                loaded = load_external_lm_runtime_settings_for_provider("ollama")
                missing = load_external_lm_runtime_settings_for_provider("zai")

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("llama3.1:8b", loaded["model"])
        self.assertIsNone(missing)

    def test_hydrate_populates_missing_env_values(self) -> None:
        """Hydration should populate external LM env vars from persisted settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                save_external_lm_runtime_settings(
                    provider="zai",
                    protocol="openai_chat",
                    model="glm-5",
                    base_url="https://api.z.ai/api/coding/paas/v4/chat/completions",
                )
                changed = hydrate_external_lm_env_from_store()

                self.assertTrue(changed)
                self.assertEqual("zai", os.getenv("ACESTEP_EXTERNAL_LM_PROVIDER"))
                self.assertEqual("openai_chat", os.getenv("ACESTEP_EXTERNAL_LM_PROTOCOL"))
                self.assertEqual("glm-5", os.getenv("ACESTEP_EXTERNAL_LM_MODEL"))
                self.assertEqual(
                    "https://api.z.ai/api/coding/paas/v4/chat/completions",
                    os.getenv("ACESTEP_EXTERNAL_BASE_URL"),
                )
                self.assertEqual("glm-5", os.getenv("ACESTEP_ZAI_MODEL"))

    def test_hydrate_does_not_override_existing_env_values(self) -> None:
        """Hydration should preserve explicitly set env values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                "os.environ",
                {
                    "XDG_DATA_HOME": tmpdir,
                    "ACESTEP_EXTERNAL_LM_MODEL": "pre-set-model",
                },
                clear=True,
            ):
                save_external_lm_runtime_settings(
                    provider="zai",
                    protocol="openai_chat",
                    model="glm-5",
                    base_url="https://api.z.ai/api/coding/paas/v4/chat/completions",
                )
                hydrate_external_lm_env_from_store()
                self.assertEqual("pre-set-model", os.getenv("ACESTEP_EXTERNAL_LM_MODEL"))

    def test_load_returns_none_for_missing_file(self) -> None:
        """Load should return ``None`` when no persisted settings file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmpdir}, clear=True):
                self.assertIsNone(load_external_lm_runtime_settings())
                self.assertEqual(
                    Path(tmpdir) / "acestep" / "config" / "external_lm_runtime.json",
                    external_lm_settings_path(),
                )


if __name__ == "__main__":
    unittest.main()
