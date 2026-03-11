"""Tests for encrypted local secret storage helpers."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


class EncryptedSecretStoreTests(unittest.TestCase):
    """Validate save/load/clear behavior for encrypted user-local secrets."""

    def test_round_trip_encrypts_and_decrypts_secret(self) -> None:
        """Stored secret should decrypt back to original plaintext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "external_ai_key.enc"
            store = EncryptedSecretStore(secret_path=path)
            store.save(secret="abc123", passphrase="p@ssphrase")

            recovered = store.load(passphrase="p@ssphrase")
            self.assertEqual("abc123", recovered)
            self.assertTrue(path.exists())

    def test_load_fails_with_wrong_passphrase(self) -> None:
        """Decrypting with incorrect passphrase should raise an explicit error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "external_ai_key.enc"
            store = EncryptedSecretStore(secret_path=path)
            store.save(secret="super-secret", passphrase="correct-pass")

            with self.assertRaises(SecretStoreError):
                store.load(passphrase="wrong-pass")

    def test_clear_removes_encrypted_secret_file(self) -> None:
        """Clear operation should delete the encrypted secret file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "external_ai_key.enc"
            store = EncryptedSecretStore(secret_path=path)
            store.save(secret="secret", passphrase="pass")
            self.assertTrue(store.exists())

            store.clear()
            self.assertFalse(store.exists())

    def test_resolve_existing_default_path_uses_legacy_when_primary_missing(self) -> None:
        """Resolver should pick legacy path when primary path does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "primary" / "external_ai_api_key.enc"
            legacy = Path(tmpdir) / "legacy" / "external_ai_api_key.enc"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text("x", encoding="utf-8")

            with patch.object(
                EncryptedSecretStore,
                "default_path",
                return_value=primary,
            ), patch.object(
                EncryptedSecretStore,
                "legacy_default_path",
                return_value=legacy,
            ):
                resolved = EncryptedSecretStore.resolve_existing_default_path()
                self.assertEqual(legacy, resolved)

    @patch("acestep.text_tasks.secure_secret_store.subprocess.run")
    @patch("acestep.text_tasks.secure_secret_store.os.name", "nt")
    def test_run_openssl_uses_env_passphrase_on_windows(self, run_mock) -> None:
        """Windows OpenSSL calls should avoid ``pass_fds`` and use child-only env."""
        completed = subprocess.CompletedProcess(args=["openssl"], returncode=0, stdout=b"", stderr=b"")
        run_mock.return_value = completed
        store = EncryptedSecretStore(
            secret_path=Path("C:/temp/external_ai_key.enc"),
            openssl_path="openssl",
        )

        result = store._run_openssl(
            args=["enc", "-aes-256-cbc", "-pbkdf2"],
            passphrase="secret-pass",
            stdin_bytes=b"payload",
        )

        self.assertIs(result, completed)
        cmd = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertIn("env:ACESTEP_OPENSSL_PASSPHRASE", cmd)
        self.assertEqual("secret-pass", kwargs["env"]["ACESTEP_OPENSSL_PASSPHRASE"])
        self.assertNotIn("pass_fds", kwargs)


if __name__ == "__main__":
    unittest.main()