"""Tests for runtime passphrase resolution and storage helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks.passphrase_store import resolve_runtime_passphrase, store_runtime_passphrase


class PassphraseStoreTests(unittest.TestCase):
    """Validate secure passphrase helper resolution order and storage paths."""

    @patch.dict("os.environ", {"ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE": "env-pass"}, clear=True)
    def test_resolve_runtime_passphrase_uses_env_first(self) -> None:
        """Env passphrase should take precedence over other backends."""
        self.assertEqual("env-pass", resolve_runtime_passphrase())

    @patch.dict("os.environ", {"ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE": "env-pass"}, clear=True)
    def test_resolve_runtime_passphrase_uses_external_ai_env_alias_first(self) -> None:
        """Generic External AI env alias should be accepted for passphrase lookup."""
        self.assertEqual("env-pass", resolve_runtime_passphrase())

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_runtime_passphrase_reads_from_file(self) -> None:
        """Passphrase file should be used when env variable is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "passphrase.txt"
            path.write_text("file-pass\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE_FILE": str(path)},
                clear=True,
            ):
                self.assertEqual("file-pass", resolve_runtime_passphrase())

    @patch.dict("os.environ", {"ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE_FILE": "/missing/passphrase.txt"}, clear=True)
    @patch("acestep.text_tasks.passphrase_store.logger.debug")
    @patch("acestep.text_tasks.passphrase_store.Path.read_text", side_effect=FileNotFoundError())
    @patch("acestep.text_tasks.passphrase_store._load_passphrase_from_keyring", return_value=None)
    @patch(
        "acestep.text_tasks.passphrase_store._load_passphrase_from_secret_tool",
        return_value="secret-tool-pass",
    )
    def test_resolve_runtime_passphrase_ignores_missing_file_and_falls_back(
        self,
        secret_tool_mock,
        _keyring_mock,
        _read_text_mock,
        logger_debug_mock,
    ) -> None:
        """Missing passphrase files should not block fallback secret resolution."""
        self.assertEqual("secret-tool-pass", resolve_runtime_passphrase())
        secret_tool_mock.assert_called_once()
        logger_debug_mock.assert_called_once()

    @patch.dict("os.environ", {"ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE_FILE": "/unreadable/passphrase.txt"}, clear=True)
    @patch("acestep.text_tasks.passphrase_store.logger.debug")
    @patch("acestep.text_tasks.passphrase_store.Path.read_text", side_effect=PermissionError())
    @patch("acestep.text_tasks.passphrase_store._load_passphrase_from_keyring", return_value=None)
    @patch(
        "acestep.text_tasks.passphrase_store._load_passphrase_from_secret_tool",
        return_value="secret-tool-pass",
    )
    def test_resolve_runtime_passphrase_ignores_unreadable_file_and_falls_back(
        self,
        secret_tool_mock,
        _keyring_mock,
        _read_text_mock,
        logger_debug_mock,
    ) -> None:
        """Unreadable passphrase files should not block fallback secret resolution."""
        self.assertEqual("secret-tool-pass", resolve_runtime_passphrase())
        secret_tool_mock.assert_called_once()
        logger_debug_mock.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    @patch("acestep.text_tasks.passphrase_store._load_passphrase_from_keyring", return_value=None)
    @patch(
        "acestep.text_tasks.passphrase_store._load_passphrase_from_secret_tool",
        return_value="secret-tool-pass",
    )
    def test_resolve_runtime_passphrase_uses_secret_tool(
        self,
        secret_tool_mock,
        _keyring_mock,
    ) -> None:
        """secret-tool result should be used when available."""
        resolved = resolve_runtime_passphrase()
        self.assertEqual("secret-tool-pass", resolved)
        secret_tool_mock.assert_called_once()

    @patch("acestep.text_tasks.passphrase_store._store_passphrase_in_secret_tool")
    @patch("acestep.text_tasks.passphrase_store._store_passphrase_in_keyring")
    def test_store_runtime_passphrase_prefers_secret_tool(
        self,
        keyring_store_mock,
        secret_tool_store_mock,
    ) -> None:
        """Store helper should accept successful secret-tool writes."""
        secret_tool_store_mock.return_value = (True, "stored")
        ok, message = store_runtime_passphrase("abc")
        self.assertTrue(ok)
        self.assertIn("stored", message)
        keyring_store_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

