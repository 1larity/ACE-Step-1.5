"""Unit tests for local format prompt wording in ``LLMHandler``."""

from __future__ import annotations

import unittest

try:
    from acestep.constants import DEFAULT_LM_REWRITE_INSTRUCTION
    from acestep.llm_inference import LLMHandler
except Exception as exc:  # pragma: no cover - import guard for constrained envs
    LLMHandler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _EchoTokenizer:
    """Tokenizer stub that returns the system instruction for prompt assertions."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Return the system message content for direct instruction assertions."""
        _ = tokenize
        _ = add_generation_prompt
        return messages[0]["content"]


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmFormatPromptInstructionTests(unittest.TestCase):
    """Validate sparse-caption expansion guidance in local rewrite prompts."""

    def test_rewrite_instruction_requires_sparse_caption_expansion(self) -> None:
        """The rewrite instruction should require short caption fragments to be expanded."""
        self.assertIn("short fragment or keyword list", DEFAULT_LM_REWRITE_INSTRUCTION)
        self.assertIn("complete ACE-Step narrative caption", DEFAULT_LM_REWRITE_INSTRUCTION)
        self.assertIn("global musical traits first", DEFAULT_LM_REWRITE_INSTRUCTION)

    def test_format_prompt_uses_updated_rewrite_instruction(self) -> None:
        """The format prompt should include the updated rewrite instruction text."""
        handler = LLMHandler()
        handler.llm_tokenizer = _EchoTokenizer()

        prompt = handler.build_formatted_prompt_for_format(
            caption="dark techno",
            lyrics="[Instrumental]",
        )

        self.assertIn("short fragment or keyword list", prompt)
        self.assertIn("complete ACE-Step narrative caption", prompt)


if __name__ == "__main__":
    unittest.main()
