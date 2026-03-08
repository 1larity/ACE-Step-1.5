"""Unit tests for inference prompt logging behavior in ``ConditioningTextMixin``."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import acestep.core.generation.handler.conditioning_text as conditioning_text_module
from acestep.core.generation.handler.conditioning_text import ConditioningTextMixin


class _Tokenizer:
    """Tiny tokenizer stub returning deterministic token ids and masks."""

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> SimpleNamespace:
        del padding, truncation, return_tensors
        token_count = min(max_length, max(1, len(text.split())))
        token_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        return SimpleNamespace(input_ids=token_ids, attention_mask=torch.ones_like(token_ids))


class _Host(ConditioningTextMixin):
    """Minimal host implementing the dependencies used by the mixin under test."""

    def __init__(self):
        self.text_tokenizer = _Tokenizer()
        self.device = "cpu"
        self.dtype = torch.float32
        self.silence_latent = torch.zeros(1, 1, 1)

    @staticmethod
    def _extract_caption_and_language(parsed_metas, captions, vocal_languages):
        """Return captions/languages directly for deterministic tests."""
        del parsed_metas
        return captions, vocal_languages

    @staticmethod
    def _format_instruction(instruction: str) -> str:
        """Return instruction unchanged."""
        return instruction

    @staticmethod
    def _format_lyrics(lyrics: str, language: str) -> str:
        """Return deterministic lyric block matching inference formatting style."""
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}\n<|endoftext|>"

    @staticmethod
    def _pad_sequences(sequences, max_length: int, pad_value: int) -> torch.Tensor:
        """Pad 1D token/mask sequences to ``max_length`` and stack them."""
        padded = []
        for sequence in sequences:
            if len(sequence) < max_length:
                pad = torch.full((max_length - len(sequence),), pad_value, dtype=sequence.dtype)
                sequence = torch.cat([sequence, pad], dim=0)
            padded.append(sequence)
        return torch.stack(padded, dim=0)


class ConditioningTextLoggingTests(unittest.TestCase):
    """Verify prompt logging defaults stay concise while retaining opt-in debug dumps."""

    def test_prompt_debug_dump_is_disabled_by_default(self):
        """It emits a compact DEBUG preview instead of full prompt/lyrics info logs."""
        host = _Host()
        with patch.dict(os.environ, {"ACESTEP_LOG_PROMPT_DEBUG": "0"}, clear=False):
            with patch.object(conditioning_text_module.logger, "info") as info_log, patch.object(
                conditioning_text_module.logger, "debug"
            ) as debug_log:
                host._prepare_text_conditioning_inputs(
                    batch_size=1,
                    instructions=["Fill the audio semantic mask based on conditions."],
                    captions=["caption"],
                    lyrics=["lyric line"],
                    parsed_metas=["- duration: 10 seconds"],
                    vocal_languages=["en"],
                    audio_cover_strength=1.0,
                )
        info_log.assert_not_called()
        self.assertGreaterEqual(debug_log.call_count, 1)

    def test_prompt_debug_dump_can_be_enabled_explicitly(self):
        """It restores full prompt/lyrics info logging when debug env flag is set."""
        host = _Host()
        with patch.dict(os.environ, {"ACESTEP_LOG_PROMPT_DEBUG": "1"}, clear=False):
            with patch.object(conditioning_text_module.logger, "info") as info_log:
                host._prepare_text_conditioning_inputs(
                    batch_size=1,
                    instructions=["Fill the audio semantic mask based on conditions."],
                    captions=["caption"],
                    lyrics=["lyric line"],
                    parsed_metas=["- duration: 10 seconds"],
                    vocal_languages=["en"],
                    audio_cover_strength=1.0,
                )
        self.assertTrue(
            any("DiT TEXT ENCODER INPUT (Inference)" in str(call.args[0]) for call in info_log.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
