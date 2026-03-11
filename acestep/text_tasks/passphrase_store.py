"""Runtime passphrase storage/resolution helpers for external AI workflows."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger


EXTERNAL_AI_SECRET_SERVICE = "acestep.external_ai"
EXTERNAL_AI_SECRET_USERNAME = "external_ai_store_passphrase"
_SECRET_TOOL_PATH = "secret-tool"


def resolve_runtime_passphrase() -> str | None:
    """Resolve non-interactive passphrase for encrypted external AI key access."""
    env_passphrase = os.getenv("ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE", "").strip()
    if env_passphrase:
        return env_passphrase

    file_path_raw = os.getenv("ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE_FILE", "").strip()
    if file_path_raw:
        try:
            text = Path(file_path_raw).expanduser().read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as exc:
            logger.debug("Ignoring external AI passphrase file {}: {}", file_path_raw, exc)
        else:
            if text:
                return text

    service = os.getenv(
        "ACESTEP_EXTERNAL_AI_SECRET_SERVICE",
        EXTERNAL_AI_SECRET_SERVICE,
    ).strip()
    username = os.getenv(
        "ACESTEP_EXTERNAL_AI_SECRET_USERNAME",
        EXTERNAL_AI_SECRET_USERNAME,
    ).strip()

    secret_tool_passphrase = _load_passphrase_from_secret_tool(
        service=service,
        username=username,
    )
    if secret_tool_passphrase:
        return secret_tool_passphrase

    keyring_passphrase = _load_passphrase_from_keyring(
        service=service,
        username=username,
    )
    if keyring_passphrase:
        return keyring_passphrase
    return None


def store_runtime_passphrase(passphrase: str) -> tuple[bool, str]:
    """Store passphrase in system secret storage for non-interactive runtime."""
    if not passphrase:
        return False, "Passphrase cannot be empty."

    service = os.getenv(
        "ACESTEP_EXTERNAL_AI_SECRET_SERVICE",
        EXTERNAL_AI_SECRET_SERVICE,
    ).strip()
    username = os.getenv(
        "ACESTEP_EXTERNAL_AI_SECRET_USERNAME",
        EXTERNAL_AI_SECRET_USERNAME,
    ).strip()

    ok_secret_tool, msg_secret_tool = _store_passphrase_in_secret_tool(
        service=service,
        username=username,
        passphrase=passphrase,
    )
    if ok_secret_tool:
        return True, msg_secret_tool

    ok_keyring, msg_keyring = _store_passphrase_in_keyring(
        service=service,
        username=username,
        passphrase=passphrase,
    )
    if ok_keyring:
        return True, msg_keyring
    return False, f"{msg_secret_tool} | {msg_keyring}"


def _load_passphrase_from_secret_tool(*, service: str, username: str) -> str | None:
    """Read passphrase from libsecret keyring via ``secret-tool lookup``."""
    tool_path = shutil.which(_SECRET_TOOL_PATH)
    if not tool_path:
        return None
    result = subprocess.run(
        [tool_path, "lookup", "service", service, "username", username],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _store_passphrase_in_secret_tool(
    *,
    service: str,
    username: str,
    passphrase: str,
) -> tuple[bool, str]:
    """Write passphrase into libsecret keyring via ``secret-tool store``."""
    tool_path = shutil.which(_SECRET_TOOL_PATH)
    if not tool_path:
        return False, "secret-tool not available"

    result = subprocess.run(
        [
            tool_path,
            "store",
            "--label",
            "ACE-Step External AI store passphrase",
            "service",
            service,
            "username",
            username,
        ],
        input=passphrase,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, "Failed writing passphrase with secret-tool"
    return True, f"Stored passphrase in secret-tool ({service}/{username})"


def _load_passphrase_from_keyring(*, service: str, username: str) -> str | None:
    """Read passphrase from Python keyring backend when available."""
    try:
        import keyring
    except Exception:
        return None

    try:
        value = keyring.get_password(service, username)
    except Exception:
        return None
    return value.strip() if value else None


def _store_passphrase_in_keyring(
    *,
    service: str,
    username: str,
    passphrase: str,
) -> tuple[bool, str]:
    """Write passphrase to Python keyring backend when available."""
    try:
        import keyring
    except Exception:
        return False, "python keyring backend unavailable"

    try:
        keyring.set_password(service, username, passphrase)
    except Exception:
        return False, "Failed writing passphrase with python keyring"
    return True, f"Stored passphrase in python keyring ({service}/{username})"
