"""Tests for encrypted secret storage helpers."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
