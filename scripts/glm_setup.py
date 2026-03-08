#!/usr/bin/env python3
"""Setup and diagnostics CLI for external GLM runtime credentials."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from acestep.text_tasks.passphrase_store import (
    GLM_SECRET_SERVICE,
    GLM_SECRET_USERNAME,
    resolve_runtime_passphrase,
    store_runtime_passphrase,
)
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/paas/v4/chat/completions"
DEFAULT_GLM_MODEL = "glm-4.5-flash"


def _build_store(path_override: str | None) -> EncryptedSecretStore:
    env_path = os.getenv("ACESTEP_GLM_SECRET_PATH", "").strip()
    if path_override or env_path:
        path = Path(path_override or env_path).expanduser()
    else:
        path = EncryptedSecretStore.resolve_existing_default_path()
    return EncryptedSecretStore(secret_path=path)


def _resolve_api_key(raw: str | None) -> str:
    if raw:
        return raw.strip()
    env_value = os.getenv("ACESTEP_GLM_API_KEY", "").strip()
    if env_value:
        return env_value
    return getpass.getpass("GLM API key (input hidden): ").strip()


def _resolve_passphrase(raw: str | None) -> str:
    if raw:
        return raw
    env_value = os.getenv("ACESTEP_GLM_STORE_PASSPHRASE", "")
    if env_value:
        return env_value
    first = getpass.getpass("Secret-store passphrase: ")
    second = getpass.getpass("Confirm passphrase: ")
    if first != second:
        raise SecretStoreError("Passphrases did not match.")
    return first


def _cmd_setup(args: argparse.Namespace) -> int:
    store = _build_store(args.store_path)
    api_key = _resolve_api_key(args.api_key)
    passphrase = _resolve_passphrase(args.passphrase)

    store.save(secret=api_key, passphrase=passphrase)
    print(f"Stored encrypted GLM API key at: {store.secret_path}")

    if args.save_passphrase:
        ok, message = store_runtime_passphrase(passphrase)
        if ok:
            print(message)
        else:
            print(f"Warning: passphrase not saved in keyring ({message}).")
            print(
                "Set ACESTEP_GLM_STORE_PASSPHRASE or "
                "ACESTEP_GLM_STORE_PASSPHRASE_FILE before launching Gradio."
            )

    model = args.model or os.getenv("ACESTEP_GLM_MODEL", DEFAULT_GLM_MODEL)
    base_url = args.base_url or os.getenv("ACESTEP_GLM_BASE_URL", DEFAULT_GLM_BASE_URL)
    print(f"Recommended export: ACESTEP_GLM_MODEL={model}")
    print(f"Recommended export: ACESTEP_GLM_BASE_URL={base_url}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    store = _build_store(args.store_path)
    direct_key_set = bool(os.getenv("ACESTEP_GLM_API_KEY", "").strip())
    passphrase = resolve_runtime_passphrase()

    print(f"Encrypted key file: {store.secret_path}")
    print(f"Encrypted key exists: {'yes' if store.exists() else 'no'}")
    print(f"Direct API key env set: {'yes' if direct_key_set else 'no'}")
    print(f"Runtime passphrase source found: {'yes' if bool(passphrase) else 'no'}")
    print(
        "Secret lookup identity: "
        f"service={os.getenv('ACESTEP_GLM_SECRET_SERVICE', GLM_SECRET_SERVICE)} "
        f"username={os.getenv('ACESTEP_GLM_SECRET_USERNAME', GLM_SECRET_USERNAME)}"
    )

    key_usable = False
    if direct_key_set:
        key_usable = True
    elif passphrase and store.exists():
        try:
            key_usable = bool(store.load(passphrase=passphrase).strip())
        except SecretStoreError as exc:
            print(f"Decrypt check failed: {exc}")

    if key_usable:
        print("GLM runtime status: ready")
        return 0
    print("GLM runtime status: not ready")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Store encrypted GLM key and keyring passphrase")
    setup.add_argument("--api-key", help="API key value (omit to input hidden)")
    setup.add_argument("--passphrase", help="Secret-store passphrase (omit to prompt)")
    setup.add_argument("--store-path", help="Override encrypted key file path")
    setup.add_argument(
        "--save-passphrase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save passphrase in system keyring for runtime use (default: true)",
    )
    setup.add_argument("--model", help=f"Default model hint (default: {DEFAULT_GLM_MODEL})")
    setup.add_argument("--base-url", help=f"Default base URL hint (default: {DEFAULT_GLM_BASE_URL})")
    setup.set_defaults(func=_cmd_setup)

    doctor = sub.add_parser("doctor", help="Check GLM runtime readiness")
    doctor.add_argument("--store-path", help="Override encrypted key file path")
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main() -> int:
    """Run GLM setup/doctor CLI."""
    args = _build_parser().parse_args()
    try:
        return int(args.func(args))
    except (SecretStoreError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

