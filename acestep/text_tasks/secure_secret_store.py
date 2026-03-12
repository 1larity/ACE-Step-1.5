"""Encrypted local secret storage for external text-task API credentials.

This module stores secrets under user-local persistent data directories using
OpenSSL symmetric encryption, while using a Windows-safe passphrase handoff.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .secure_secret_store_exec import (
    run_openssl_posix,
    run_openssl_windows,
    sanitize_error,
)
from .secure_secret_store_paths import default_secret_path, legacy_secret_path


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
    def default_path(filename: str = "external_ai_api_key.enc") -> Path:
        """Return default encrypted secret path in persistent user data storage."""
        return default_secret_path(filename=filename)

    @staticmethod
    def legacy_default_path(filename: str = "external_ai_api_key.enc") -> Path:
        """Return historical encrypted secret path under ``~/.local/share``."""
        return legacy_secret_path(filename=filename)

    @staticmethod
    def resolve_existing_default_path(filename: str = "external_ai_api_key.enc") -> Path:
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
        """Encrypt and store a secret value at ``secret_path``."""
        if not secret:
            raise SecretStoreError("Secret cannot be empty.")
        if not passphrase:
            raise SecretStoreError("Passphrase cannot be empty.")

        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path.parent.chmod(0o700)
        result = self._run_openssl(
            args=["enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-out", str(self.secret_path)],
            passphrase=passphrase,
            stdin_bytes=secret.encode("utf-8"),
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise SecretStoreError(sanitize_error(stderr))
        self.secret_path.chmod(0o600)

    def load(self, *, passphrase: str) -> str:
        """Decrypt and return secret value from ``secret_path``."""
        if not self.secret_path.exists():
            raise SecretStoreError(f"Secret not found at: {self.secret_path}")
        if not passphrase:
            raise SecretStoreError("Passphrase cannot be empty.")

        result = self._run_openssl(
            args=["enc", "-d", "-aes-256-cbc", "-pbkdf2", "-in", str(self.secret_path)],
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
    ):
        """Execute OpenSSL via the shared platform-safe helpers."""
        if os.name == "nt":
            return run_openssl_windows(
                openssl_path=self.openssl_path,
                args=args,
                passphrase=passphrase,
                stdin_bytes=stdin_bytes,
            )
        return run_openssl_posix(
            openssl_path=self.openssl_path,
            args=args,
            passphrase=passphrase,
            stdin_bytes=stdin_bytes,
        )
