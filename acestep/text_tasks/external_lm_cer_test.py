"""Tests for bulk caption enhancement review helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.text_tasks.external_lm_cer import (
    DEFAULT_CER_PROMPTS,
    DEFAULT_OLLAMA_MAX_MODEL_GB,
    filter_ollama_models_by_size,
    is_model_access_error,
    is_quota_like_api_error,
    load_cer_prompts,
    resolve_coding_base_url,
    run_cer_campaign,
)
from acestep.text_tasks.external_lm_ollama_catalog import OllamaModelInfo


class ExternalLmCerHelpersTest(unittest.TestCase):
    """Exercise CER helper behavior without real provider calls."""

    def test_load_cer_prompts_uses_default_limit(self) -> None:
        prompts = load_cer_prompts(prompt_file=None, prompt_limit=3)
        self.assertEqual(DEFAULT_CER_PROMPTS[:3], prompts)

    def test_load_cer_prompts_reads_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompts.txt"
            prompt_file.write_text("one\ntwo\nthree\n", encoding="utf-8")
            prompts = load_cer_prompts(prompt_file=prompt_file, prompt_limit=2)
        self.assertEqual(["one", "two"], prompts)

    def test_is_quota_like_api_error_detects_credit_messages(self) -> None:
        self.assertTrue(is_quota_like_api_error("HTTP 402: no account balance for this request"))
        self.assertTrue(is_quota_like_api_error("insufficient_quota"))
        self.assertFalse(is_quota_like_api_error("model not found"))

    def test_is_model_access_error_detects_subscription_gap(self) -> None:
        self.assertTrue(
            is_model_access_error(
                'HTTP 429: {"error":{"code":"1311","message":"Your current subscription plan '
                'does not yet include access to GLM-5-Turbo"}}'
            )
        )
        self.assertFalse(is_model_access_error("HTTP 402: no account balance"))

    def test_resolve_coding_base_url_promotes_zai_endpoint(self) -> None:
        self.assertEqual(
            "https://api.z.ai/api/coding/paas/v4/chat/completions",
            resolve_coding_base_url(
                "zai",
                "https://api.z.ai/api/paas/v4/chat/completions",
            ),
        )

    def test_filter_ollama_models_by_size_skips_large_or_unknown_models(self) -> None:
        kept, skipped = filter_ollama_models_by_size(
            models=["qwen3:4b", "qwen3:30b", "missing:model"],
            catalog=[
                OllamaModelInfo(name="qwen3:4b", size_bytes=2497293931),
                OllamaModelInfo(name="qwen3:30b", size_bytes=18556699314),
            ],
            max_model_gb=DEFAULT_OLLAMA_MAX_MODEL_GB,
        )
        self.assertEqual(["qwen3:4b"], kept)
        self.assertEqual(
            [("qwen3:30b", 18556699314), ("missing:model", None)],
            skipped,
        )


class ExternalLmCerRunnerTest(unittest.TestCase):
    """Verify campaign control flow around coding-endpoint fallback."""

    def test_run_cer_campaign_retries_with_coding_endpoint(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_format_fn(*, caption, lyrics, user_metadata, debug):
            calls.append((caption, "format"))
            from os import getenv

            if getenv("ACESTEP_EXTERNAL_BASE_URL", "").find("/api/coding/paas/") == -1:
                raise RuntimeError("HTTP 402: no account balance")
            return type(
                "Result",
                (),
                {
                    "caption": f"{caption} expanded",
                    "bpm": 125,
                    "duration": 240.0,
                    "keyscale": "D major",
                    "language": "English",
                    "timesignature": "4/4",
                    "status_message": "ok",
                },
            )()

        def fake_discover_fn(*, provider, protocol, base_url):
            return ["glm-4.7"], base_url

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cer.jsonl"
            exit_code = run_cer_campaign(
                provider="zai",
                protocol="openai_chat",
                prompt_file=None,
                prompt_limit=1,
                repeat=1,
                user_metadata={"bpm": 125},
                lyrics="",
                sleep_sec=0.0,
                output_path=output,
                models=None,
                format_fn=fake_format_fn,
                discover_fn=fake_discover_fn,
                ollama_max_model_gb=None,
            )
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(rows))
        self.assertEqual(2, len(calls))
        self.assertIn('"endpoint_switched_to_coding": true', rows[0].lower())

    def test_run_cer_campaign_skips_oversized_ollama_models(self) -> None:
        seen_models: list[str] = []

        def fake_format_fn(*, caption, lyrics, user_metadata, debug):
            from os import getenv

            model = getenv("ACESTEP_EXTERNAL_LM_MODEL", "")
            seen_models.append(model)
            return type(
                "Result",
                (),
                {
                    "caption": f"{caption} expanded",
                    "bpm": 125,
                    "duration": 240.0,
                    "keyscale": "D major",
                    "language": "English",
                    "timesignature": "4/4",
                    "status_message": "ok",
                },
            )()

        def fake_catalog_fn(*, base_url):
            return [
                OllamaModelInfo(name="small:model", size_bytes=2 * (1024**3)),
                OllamaModelInfo(name="huge:model", size_bytes=18 * (1024**3)),
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cer.jsonl"
            exit_code = run_cer_campaign(
                provider="ollama",
                protocol="openai_chat",
                prompt_file=None,
                prompt_limit=1,
                repeat=1,
                user_metadata={},
                lyrics="",
                sleep_sec=0.0,
                output_path=output,
                ollama_max_model_gb=6.0,
                models=["small:model", "huge:model"],
                format_fn=fake_format_fn,
                ollama_catalog_fn=fake_catalog_fn,
            )
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(0, exit_code)
        self.assertEqual(["small:model"], seen_models)
        self.assertEqual(1, len(rows))
        self.assertIn('"model_size_bytes": 2147483648', rows[0])

    def test_run_cer_campaign_skips_remaining_attempts_for_unavailable_model(self) -> None:
        seen_models: list[str] = []

        def fake_format_fn(*, caption, lyrics, user_metadata, debug):
            from os import getenv

            model = getenv("ACESTEP_EXTERNAL_LM_MODEL", "")
            seen_models.append(model)
            if model == "blocked:model":
                raise RuntimeError(
                    'HTTP 429: {"error":{"code":"1311","message":"Your current subscription '
                    'plan does not yet include access to blocked:model"}}'
                )
            return type(
                "Result",
                (),
                {
                    "caption": f"{caption} expanded",
                    "bpm": 125,
                    "duration": 240.0,
                    "keyscale": "D major",
                    "language": "English",
                    "timesignature": "4/4",
                    "status_message": "ok",
                },
            )()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cer.jsonl"
            exit_code = run_cer_campaign(
                provider="zai",
                protocol="openai_chat",
                prompt_file=None,
                prompt_limit=2,
                repeat=3,
                user_metadata={},
                lyrics="",
                sleep_sec=0.0,
                output_path=output,
                ollama_max_model_gb=None,
                models=["blocked:model", "ok:model"],
                format_fn=fake_format_fn,
            )
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(1, exit_code)
        self.assertEqual(["blocked:model", "ok:model", "ok:model", "ok:model", "ok:model", "ok:model", "ok:model"], seen_models)
        self.assertEqual(7, len(rows))
