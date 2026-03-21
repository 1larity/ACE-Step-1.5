"""Tests for Ollama model catalog helpers."""

from __future__ import annotations

import unittest

from acestep.text_tasks.external_lm_ollama_catalog import (
    build_ollama_tags_url,
    parse_ollama_tags_payload,
)


class OllamaCatalogTest(unittest.TestCase):
    """Verify Ollama catalog URL and payload parsing helpers."""

    def test_build_ollama_tags_url_uses_root_host(self) -> None:
        self.assertEqual(
            "http://192.168.1.124:11434/api/tags",
            build_ollama_tags_url("http://192.168.1.124:11434/v1/chat/completions"),
        )

    def test_parse_ollama_tags_payload_extracts_sizes(self) -> None:
        payload = {
            "models": [
                {"name": "qwen3:4b", "size": 2497293931},
                {"model": "qwen3:8b", "size": "5225388164"},
            ]
        }
        parsed = parse_ollama_tags_payload(payload)
        self.assertEqual(
            [("qwen3:4b", 2497293931), ("qwen3:8b", 5225388164)],
            [(item.name, item.size_bytes) for item in parsed],
        )
