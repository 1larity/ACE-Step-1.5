"""Tests for the shared LM task debug switch."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from acestep.lm_task_debug import is_lm_task_debug_enabled


class LmTaskDebugTests(unittest.TestCase):
    """Validate the single global LM task debug switch behavior."""

    def test_env_override_can_disable_logging(self) -> None:
        """An explicit global env override should disable LM task logging."""
        with patch("acestep.lm_task_debug.DEBUG_LLM", "ON"), patch(
            "acestep.lm_task_debug.DEBUG_EXTERNAL_AI", "ON"
        ), patch.dict(os.environ, {"ACESTEP_DEBUG_LM_TASKS": "0"}, clear=False):
            self.assertFalse(is_lm_task_debug_enabled())

    def test_env_override_can_enable_logging(self) -> None:
        """An explicit global env override should enable LM task logging."""
        with patch("acestep.lm_task_debug.DEBUG_LLM", "OFF"), patch(
            "acestep.lm_task_debug.DEBUG_EXTERNAL_AI", "OFF"
        ), patch.dict(os.environ, {"ACESTEP_DEBUG_LM_TASKS": "1"}, clear=False):
            self.assertTrue(is_lm_task_debug_enabled())

    def test_external_ai_env_enables_logging(self) -> None:
        """External AI debug env should enable the shared LM task logs."""
        with patch("acestep.lm_task_debug.DEBUG_LLM", "OFF"), patch(
            "acestep.lm_task_debug.DEBUG_EXTERNAL_AI", "OFF"
        ), patch.dict(os.environ, {"ACESTEP_EXTERNAL_AI_DEBUG": "1"}, clear=True):
            self.assertTrue(is_lm_task_debug_enabled())
