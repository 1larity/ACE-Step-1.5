"""Tests for the external AI text-task CLI script."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_SCRIPT_PATH = Path(__file__).with_name("external_ai_text_tasks.py")
_SPEC = importlib.util.spec_from_file_location("external_ai_text_tasks_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load external_ai_text_tasks.py for testing.")
external_ai_text_tasks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(external_ai_text_tasks)


class ExternalAiTextTasksScriptTests(unittest.TestCase):
    """Validate external AI text-task CLI behaviors."""

    @patch.object(external_ai_text_tasks, "_resolve_passphrase")
    @patch.object(external_ai_text_tasks, "_build_store")
    @patch.object(external_ai_text_tasks, "request_external_ai_plan")
    def test_cmd_plan_uses_env_api_key_without_loading_store(
        self,
        request_plan_mock,
        build_store_mock,
        resolve_passphrase_mock,
    ) -> None:
        """Plan command should short-circuit on direct env API key for headless runs."""
        request_plan_mock.return_value = MagicMock(to_dict=lambda: {"caption": "ok"})
        store = build_store_mock.return_value
        store.load.return_value = "stored-key"
        args = argparse.Namespace(
            store_path=None,
            api_key=None,
            passphrase=None,
            intent="write a plan",
            model=None,
            base_url=None,
            timeout=60,
            task_focus="all",
            acestep_payload=False,
            out=None,
        )

        with patch.dict("os.environ", {"ACESTEP_ZAI_API_KEY": "env-key"}, clear=True):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = external_ai_text_tasks._cmd_plan(args)

        self.assertEqual(0, result)
        resolve_passphrase_mock.assert_not_called()
        store.load.assert_not_called()
        self.assertEqual("env-key", request_plan_mock.call_args.kwargs["api_key"])


if __name__ == "__main__":
    unittest.main()
