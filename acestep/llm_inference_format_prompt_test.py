"""Unit tests for format-prompt scaffold preservation in ``LLMHandler``."""

from __future__ import annotations

import unittest

try:
    from acestep.constants import DEFAULT_LM_INSPIRED_INSTRUCTION, DEFAULT_LM_REWRITE_INSTRUCTION
    from acestep.llm_inference import LLMHandler
except Exception as exc:  # pragma: no cover - import guard for constrained envs
    LLMHandler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _EchoTokenizer:
    """Tokenizer stub that returns the user-content payload for prompt assertions."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize
        _ = add_generation_prompt
        return messages[1]["content"]


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmFormatPromptScaffoldTests(unittest.TestCase):
    """Validate scaffold-preservation directives added to local format prompts."""

    def test_format_prompt_includes_preserve_block_when_tags_exist(self) -> None:
        """Prompt should include preserve directives for arrangement/instrument tags."""
        handler = LLMHandler()
        handler.llm_tokenizer = _EchoTokenizer()

        prompt = handler.build_formatted_prompt_for_format(
            caption="Synth intro with driving bass and vocals",
            lyrics="[Verse 1]\nplain text",
        )

        self.assertIn("# Preserve", prompt)
        self.assertIn("Preserve arrangement tags exactly", prompt)
        self.assertIn("Preserve instrument tags exactly", prompt)

    def test_format_prompt_omits_preserve_block_without_tags(self) -> None:
        """Prompt should stay compact when no arrangement/instrument tags are present."""
        handler = LLMHandler()
        handler.llm_tokenizer = _EchoTokenizer()

        prompt = handler.build_formatted_prompt_for_format(
            caption="dreamy and emotional",
            lyrics="falling into moonlight",
        )

        self.assertNotIn("# Preserve", prompt)

    def test_local_instructions_require_singer_gender_and_delivery(self) -> None:
        """Local caption instructions should explicitly request singer gender and delivery mood."""
        self.assertIn("singer gender and delivery mood", DEFAULT_LM_INSPIRED_INSTRUCTION)
        self.assertIn("singer gender and delivery mood", DEFAULT_LM_REWRITE_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()

