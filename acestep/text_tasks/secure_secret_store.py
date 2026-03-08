"""Encrypted local secret storage for external text-task API credentials.

This module stores secrets under user-local persistent data directories using
OpenSSL symmetric encryption, avoiding plaintext key files on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SecretStoreError(RuntimeError):
    """Raised when encrypted secret read/write operations fail."""


@dataclass(frozen=True)
class EncryptedSecretStore:
    """OpenSSL-backed encrypted secret store in user-local persistent storage.

    Args:
        secret_path: Absolute file path for encrypted secret bytes.
        openssl_path: Optional OpenSSL binary path override.
    """

    secret_path: Path
    openssl_path: str | None = None

    def __post_init__(self) -> None:
        """Validate OpenSSL availability for encryption/decryption operations."""
        openssl_binary = self.openssl_path or shutil.which("openssl")
        if not openssl_binary:
            raise SecretStoreError("OpenSSL is required for encrypted secret storage.")
        object.__setattr__(self, "openssl_path", openssl_binary)

    @staticmethod
    def default_path(filename: str = "glm_api_key.enc") -> Path:
        """Return default encrypted secret path in persistent user data storage."""
        xdg_data_home = os.getenv("XDG_DATA_HOME")
        base = (
            Path(xdg_data_home).expanduser()
            if xdg_data_home
            else Path.home() / ".local" / "share"
        )
        return base / "acestep" / "secrets" / filename

    @staticmethod
    def legacy_default_path(filename: str = "glm_api_key.enc") -> Path:
        """Return historical encrypted secret path under ``~/.local/share``."""
        return Path.home() / ".local" / "share" / "acestep" / "secrets" / filename

    @staticmethod
    def resolve_existing_default_path(filename: str = "glm_api_key.enc") -> Path:
        """Return existing default path with legacy fallback when available."""
        primary = EncryptedSecretStore.default_path(filename=filename)
        if primary.exists():
            return primary
        legacy = EncryptedSecretStore.legacy_default_path(filename=filename)
        if legacy.exists():
            return legacy
        return primary

    def exists(self) -> bool:
        """Return whether an encrypted secret file exists."""
        return self.secret_path.exists()

    def save(self, *, secret: str, passphrase: str) -> None:
        """Encrypt and store a secret value at ``secret_path``.

        Args:
            secret: Plaintext secret value.
            passphrase: User-supplied encryption passphrase.

        Raises:
            SecretStoreError: If encryption or file write fails.
        """
        if not secret:
            raise SecretStoreError("Secret cannot be empty.")
        if not passphrase:
            raise SecretStoreError("Passphrase cannot be empty.")

        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path.parent.chmod(0o700)

        result = self._run_openssl(
            args=[
                "enc",
                "-aes-256-cbc",
                "-pbkdf2",
                "-salt",
                "-out",
                str(self.secret_path),
            ],
            passphrase=passphrase,
            stdin_bytes=secret.encode("utf-8"),
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise SecretStoreError(self._sanitize_error(stderr))

        self.secret_path.chmod(0o600)

    def load(self, *, passphrase: str) -> str:
        """Decrypt and return secret value from ``secret_path``.

        Args:
            passphrase: User-supplied decryption passphrase.

        Returns:
            str: Decrypted secret value.

        Raises:
            SecretStoreError: If secret file is missing or decryption fails.
        """
        if not self.secret_path.exists():
            raise SecretStoreError(f"Secret not found at: {self.secret_path}")
        if not passphrase:
            raise SecretStoreError("Passphrase cannot be empty.")

        result = self._run_openssl(
            args=[
                "enc",
                "-d",
                "-aes-256-cbc",
                "-pbkdf2",
                "-in",
                str(self.secret_path),
            ],
            passphrase=passphrase,
            stdin_bytes=None,
        )
        if result.returncode != 0:
            raise SecretStoreError("Failed to decrypt secret. Check passphrase.")

        try:
            return result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SecretStoreError("Decrypted secret is not valid UTF-8.") from exc

    def clear(self) -> None:
        """Delete encrypted secret file if it exists."""
        if self.secret_path.exists():
            self.secret_path.unlink()

    def _run_openssl(
        self,
        *,
        args: list[str],
        passphrase: str,
        stdin_bytes: bytes | None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Execute OpenSSL with passphrase provided via private file descriptor."""
        pass_r, pass_w = os.pipe()
        try:
            os.write(pass_w, passphrase.encode("utf-8"))
        finally:
            os.close(pass_w)

        cmd = [self.openssl_path, *args, "-pass", f"fd:{pass_r}"]
        try:
            result = subprocess.run(
                cmd,
                input=stdin_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(pass_r,),
                check=False,
            )
        finally:
            os.close(pass_r)
        return result

    @staticmethod
    def _sanitize_error(stderr: str) -> str:
        """Return a concise non-sensitive error message."""
        if not stderr:
            return "OpenSSL operation failed."
        first_line = stderr.strip().splitlines()[0]
        return first_line[:200]
