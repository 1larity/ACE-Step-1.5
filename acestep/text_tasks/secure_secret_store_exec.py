"""OpenSSL execution helpers for encrypted external AI secret storage."""

from __future__ import annotations

import os
import subprocess

OPENSSL_PASSPHRASE_ENV = "ACESTEP_OPENSSL_PASSPHRASE"


def run_openssl(
    *,
    openssl_path: str,
    args: list[str],
    passphrase: str,
    stdin_bytes: bytes | None,
) -> subprocess.CompletedProcess[bytes]:
    """Execute OpenSSL with a platform-safe passphrase handoff."""
    if os.name == "nt":
        return run_openssl_windows(
            openssl_path=openssl_path,
            args=args,
            passphrase=passphrase,
            stdin_bytes=stdin_bytes,
        )
    return run_openssl_posix(
        openssl_path=openssl_path,
        args=args,
        passphrase=passphrase,
        stdin_bytes=stdin_bytes,
    )


def run_openssl_windows(
    *,
    openssl_path: str,
    args: list[str],
    passphrase: str,
    stdin_bytes: bytes | None,
) -> subprocess.CompletedProcess[bytes]:
    """Execute OpenSSL on Windows without unsupported ``pass_fds`` usage."""
    env = os.environ.copy()
    env[OPENSSL_PASSPHRASE_ENV] = passphrase
    cmd = [openssl_path, *args, "-pass", f"env:{OPENSSL_PASSPHRASE_ENV}"]
    return subprocess.run(
        cmd,
        input=stdin_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )


def run_openssl_posix(
    *,
    openssl_path: str,
    args: list[str],
    passphrase: str,
    stdin_bytes: bytes | None,
) -> subprocess.CompletedProcess[bytes]:
    """Execute OpenSSL on POSIX via a private passphrase file descriptor."""
    pass_r, pass_w = os.pipe()
    try:
        os.write(pass_w, passphrase.encode("utf-8"))
    finally:
        os.close(pass_w)

    cmd = [openssl_path, *args, "-pass", f"fd:{pass_r}"]
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


def sanitize_error(stderr: str) -> str:
    """Return a concise non-sensitive error message."""
    if not stderr:
        return "OpenSSL operation failed."
    return stderr.strip().splitlines()[0][:200]
