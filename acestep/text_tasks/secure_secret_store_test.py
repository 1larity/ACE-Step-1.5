"""Tests for encrypted secret storage helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.text_tasks import secure_secret_store
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore


class SecureSecretStoreTests(unittest.TestCase):
    """Verify encrypted secret storage save/load flows."""

    def test_save_and_load_round_trip_with_mocked_openssl(self) -> None:
        """Save followed by load should round-trip the secret content."""

        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "secret.enc"
            with patch("shutil.which", return_value="/usr/bin/openssl"):
                store = EncryptedSecretStore(secret_path=secret_path)

            def fake_run_openssl(*, args, passphrase, stdin_bytes):
                if "-d" in args:
                    return subprocess.CompletedProcess(
                        args=["openssl"],
                        returncode=0,
                        stdout=b"test-key",
                        stderr=b"",
                    )
                secret_path.write_bytes(b"encrypted")
                return subprocess.CompletedProcess(
                    args=["openssl"],
                    returncode=0,
                    stdout=b"",
                    stderr=b"",
                )

            with patch.object(
                EncryptedSecretStore,
                "_run_openssl",
                autospec=True,
                side_effect=lambda _self, **kwargs: fake_run_openssl(**kwargs),
            ):
                store.save(secret="test-key", passphrase="passphrase")
                secret = store.load(passphrase="passphrase")
                self.assertTrue(secret_path.exists())

        self.assertEqual(secret, "test-key")

    def test_save_and_load_round_trip_with_native_keyring(self) -> None:
        """Windows/macOS should prefer the native keyring backend when available."""

        fake_keyring = types.SimpleNamespace()
        captured: dict[tuple[str, str], str] = {}

        def set_password(service: str, username: str, secret: str) -> None:
            captured[(service, username)] = secret

        def get_password(service: str, username: str) -> str | None:
            return captured.get((service, username))

        fake_keyring.set_password = set_password
        fake_keyring.get_password = get_password

        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "openai_api_key.enc"
            with patch.object(secure_secret_store.sys, "platform", "win32"):
                with patch.dict(sys.modules, {"keyring": fake_keyring}):
                    with patch("shutil.which", return_value=None):
                        store = EncryptedSecretStore(secret_path=secret_path)
                        store.save(secret="test-key", passphrase="")
                        secret = store.load(passphrase="")

        self.assertEqual(secret, "test-key")
        self.assertEqual(
            captured[("acestep.external_lm.secret_store", str(secret_path))],
            "test-key",
        )


if __name__ == "__main__":
    unittest.main()
