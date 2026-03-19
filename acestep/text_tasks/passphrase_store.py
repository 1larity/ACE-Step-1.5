"""Cross-platform runtime passphrase storage helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


EXTERNAL_LM_SECRET_SERVICE = "acestep.external_lm"
EXTERNAL_LM_SECRET_USERNAME = "external_lm_store_passphrase"
GLM_SECRET_SERVICE = EXTERNAL_LM_SECRET_SERVICE
GLM_SECRET_USERNAME = EXTERNAL_LM_SECRET_USERNAME
_SECRET_TOOL_PATH = "secret-tool"


def resolve_runtime_passphrase() -> str | None:
    """Resolve a runtime passphrase from env, file, secret-tool, or keyring."""

    env_passphrase = os.getenv("ACESTEP_GLM_STORE_PASSPHRASE", "").strip()
    if env_passphrase:
        return env_passphrase

    file_path_raw = os.getenv("ACESTEP_GLM_STORE_PASSPHRASE_FILE", "").strip()
    if file_path_raw:
        try:
            text = Path(file_path_raw).expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text

    service = os.getenv("ACESTEP_GLM_SECRET_SERVICE", EXTERNAL_LM_SECRET_SERVICE).strip()
    username = os.getenv("ACESTEP_GLM_SECRET_USERNAME", EXTERNAL_LM_SECRET_USERNAME).strip()

    secret_tool_passphrase = _load_passphrase_from_secret_tool(
        service=service,
        username=username,
    )
    if secret_tool_passphrase:
        return secret_tool_passphrase

    return _load_passphrase_from_keyring(service=service, username=username)


def store_runtime_passphrase(passphrase: str) -> tuple[bool, str]:
    """Persist a runtime passphrase using the best available secret store."""

    if not passphrase:
        return False, "Passphrase cannot be empty."

    service = os.getenv("ACESTEP_GLM_SECRET_SERVICE", EXTERNAL_LM_SECRET_SERVICE).strip()
    username = os.getenv("ACESTEP_GLM_SECRET_USERNAME", EXTERNAL_LM_SECRET_USERNAME).strip()

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
    """Read passphrase from Linux libsecret when ``secret-tool`` exists."""

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
    """Write passphrase to Linux libsecret when ``secret-tool`` exists."""

    tool_path = shutil.which(_SECRET_TOOL_PATH)
    if not tool_path:
        return False, "secret-tool not available"

    result = subprocess.run(
        [
            tool_path,
            "store",
            "--label",
            "ACE-Step external LM passphrase",
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
    """Read passphrase from Python keyring when available."""

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
    """Write passphrase to Python keyring when available."""

    try:
        import keyring
    except Exception:
        return False, "python keyring backend unavailable"

    try:
        keyring.set_password(service, username, passphrase)
    except Exception:
        return False, "Failed writing passphrase with python keyring"
    return True, f"Stored passphrase in python keyring ({service}/{username})"
