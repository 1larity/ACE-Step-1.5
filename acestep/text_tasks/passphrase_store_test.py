"""Tests for cross-platform runtime passphrase storage helpers."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from acestep.text_tasks import passphrase_store


class PassphraseStoreTests(unittest.TestCase):
    """Verify env, file, and keyring passphrase fallback behavior."""

    def test_resolve_uses_env_first(self) -> None:
        """Environment passphrase should take precedence."""

        with patch.dict(os.environ, {"ACESTEP_GLM_STORE_PASSPHRASE": "secret"}, clear=True):
            self.assertEqual(passphrase_store.resolve_runtime_passphrase(), "secret")

    def test_resolve_preserves_env_whitespace(self) -> None:
        """Environment passphrases should remain byte-for-byte intact."""

        with patch.dict(
            os.environ,
            {"ACESTEP_GLM_STORE_PASSPHRASE": "  secret  "},
            clear=True,
        ):
            self.assertEqual(passphrase_store.resolve_runtime_passphrase(), "  secret  ")

    def test_resolve_reads_passphrase_file(self) -> None:
        """A passphrase file should be used when env var is absent."""

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("  file-secret  ")
            file_path = handle.name
        try:
            with patch.dict(
                os.environ,
                {"ACESTEP_GLM_STORE_PASSPHRASE_FILE": file_path},
                clear=True,
            ):
                self.assertEqual(passphrase_store.resolve_runtime_passphrase(), "  file-secret  ")
        finally:
            os.unlink(file_path)

    def test_store_falls_back_to_python_keyring(self) -> None:
        """Python keyring should back passphrase storage when secret-tool is absent."""

        fake_keyring = types.SimpleNamespace()
        captured: dict[str, str] = {}

        def set_password(service: str, username: str, passphrase: str) -> None:
            captured["service"] = service
            captured["username"] = username
            captured["passphrase"] = passphrase

        fake_keyring.set_password = set_password
        fake_keyring.get_password = lambda *_args: "saved-secret"

        with patch.object(passphrase_store.shutil, "which", return_value=None):
            with patch.dict(sys.modules, {"keyring": fake_keyring}):
                ok, message = passphrase_store.store_runtime_passphrase("saved-secret")
                resolved = passphrase_store.resolve_runtime_passphrase()

        self.assertTrue(ok)
        self.assertIn("python keyring", message)
        self.assertEqual(captured["passphrase"], "saved-secret")
        self.assertEqual(resolved, "saved-secret")

    def test_resolve_falls_back_to_keyring_when_passphrase_file_is_missing(self) -> None:
        """A missing passphrase file should not block later fallback providers."""

        fake_keyring = types.SimpleNamespace()
        fake_keyring.get_password = lambda *_args: "keyring-secret"

        with patch.object(passphrase_store.shutil, "which", return_value=None):
            with patch.dict(sys.modules, {"keyring": fake_keyring}):
                with patch.dict(
                    os.environ,
                    {"ACESTEP_GLM_STORE_PASSPHRASE_FILE": "/tmp/does-not-exist-passphrase.txt"},
                    clear=True,
                ):
                    resolved = passphrase_store.resolve_runtime_passphrase()

        self.assertEqual(resolved, "keyring-secret")

    def test_resolve_preserves_keyring_whitespace(self) -> None:
        """Keyring-stored passphrases should remain byte-for-byte intact."""

        fake_keyring = types.SimpleNamespace()
        fake_keyring.get_password = lambda *_args: "  keyring-secret  "

        with patch.object(passphrase_store.shutil, "which", return_value=None):
            with patch.dict(sys.modules, {"keyring": fake_keyring}):
                resolved = passphrase_store.resolve_runtime_passphrase()

        self.assertEqual(resolved, "  keyring-secret  ")


if __name__ == "__main__":
    unittest.main()
