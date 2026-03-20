"""Tests for cached external model-list helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks.external_lm_model_cache import (
    external_lm_model_cache_path,
    invalidate_cached_external_models,
    load_cached_external_models,
    save_cached_external_models,
)


class ExternalLmModelCacheTests(unittest.TestCase):
    """Verify external model-list caching stays bounded and invalidatable."""

    def test_save_and_load_cached_external_models(self) -> None:
        """Fresh cached model entries should round-trip successfully."""

        with tempfile.TemporaryDirectory() as tmpdir, _PatchedEnv(tmpdir):
            save_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
                models=["gpt-4o-mini", "gpt-4.1-mini"],
            )

            cached = load_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
            )

        self.assertEqual(cached, ["gpt-4o-mini", "gpt-4.1-mini"])

    def test_load_cached_external_models_ignores_stale_entries(self) -> None:
        """Expired cache entries should be ignored automatically."""

        with tempfile.TemporaryDirectory() as tmpdir, _PatchedEnv(tmpdir):
            path = external_lm_model_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                '["openai","openai_chat","https://api.openai.com/v1/chat/completions"]': {
                    "models": ["gpt-4o-mini"],
                    "updated_at": time.time() - 100,
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            os.environ["ACESTEP_EXTERNAL_MODEL_CACHE_TTL_SEC"] = "10"

            cached = load_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
            )

        self.assertIsNone(cached)

    def test_invalidate_cached_external_models_removes_matching_entries(self) -> None:
        """Invalidation should clear matching provider entries while preserving others."""

        with tempfile.TemporaryDirectory() as tmpdir, _PatchedEnv(tmpdir):
            save_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
                models=["gpt-4o-mini"],
            )
            save_cached_external_models(
                provider="ollama",
                protocol="openai_chat",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
                models=["qwen3:4b"],
            )

            invalidate_cached_external_models(provider="openai")

            openai_cached = load_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
            )
            ollama_cached = load_cached_external_models(
                provider="ollama",
                protocol="openai_chat",
                base_url="http://127.0.0.1:11434/v1/chat/completions",
            )

        self.assertIsNone(openai_cached)
        self.assertEqual(ollama_cached, ["qwen3:4b"])

    def test_invalidate_cached_external_models_swallows_unlink_errors(self) -> None:
        """Cache invalidation should stay best-effort if the cache file cannot be deleted."""

        with tempfile.TemporaryDirectory() as tmpdir, _PatchedEnv(tmpdir):
            save_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url="https://api.openai.com/v1/chat/completions",
                models=["gpt-4o-mini"],
            )

            with patch.object(Path, "unlink", side_effect=OSError("read only")):
                invalidate_cached_external_models(provider="openai")

    def test_invalidate_cached_external_models_handles_base_urls_with_delimiters(self) -> None:
        """Invalidation should keep working even if base URLs contain delimiter-like text."""

        special_base_url = "https://example.invalid/query?next=a||b"
        with tempfile.TemporaryDirectory() as tmpdir, _PatchedEnv(tmpdir):
            save_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url=special_base_url,
                models=["gpt-4o-mini"],
            )

            invalidate_cached_external_models(base_url=special_base_url)

            cached = load_cached_external_models(
                provider="openai",
                protocol="openai_chat",
                base_url=special_base_url,
            )

        self.assertIsNone(cached)


class _PatchedEnv:
    """Temporarily redirect user-local storage into a writable temp directory."""

    def __init__(self, tmpdir: str) -> None:
        self.tmpdir = tmpdir
        self.original = None

    def __enter__(self) -> "_PatchedEnv":
        """Apply the temporary XDG data home override."""

        self.original = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = self.tmpdir
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Restore the original XDG data home value."""

        if self.original is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = self.original
        os.environ.pop("ACESTEP_EXTERNAL_MODEL_CACHE_TTL_SEC", None)
        return False


if __name__ == "__main__":
    unittest.main()
