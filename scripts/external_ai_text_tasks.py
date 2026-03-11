#!/usr/bin/env python3
"""External AI text-task preprocessor for ACE-Step caption/lyrics/metadata planning."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from acestep.text_tasks.external_ai_text_tasks import (
    build_acestep_generation_payload,
    request_external_ai_plan,
)
from acestep.text_tasks.secure_secret_store import EncryptedSecretStore, SecretStoreError


DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/chat/completions"
DEFAULT_ZAI_MODEL = "glm-4.5-flash"


def _build_store(path_override: str | None) -> EncryptedSecretStore:
    env_path = os.getenv("ACESTEP_ZAI_SECRET_PATH", "").strip()
    if path_override or env_path:
        final = Path(path_override or env_path).expanduser()
    else:
        final = EncryptedSecretStore.resolve_existing_default_path()
    return EncryptedSecretStore(secret_path=final)


def _resolve_passphrase(raw: str | None, *, confirm: bool = False) -> str:
    if raw:
        return raw
    env_value = os.getenv("ACESTEP_EXTERNAL_AI_STORE_PASSPHRASE", "")
    if env_value:
        return env_value

    first = getpass.getpass("Secret-store passphrase: ")
    if confirm:
        second = getpass.getpass("Confirm passphrase: ")
        if first != second:
            raise SecretStoreError("Passphrases did not match.")
    return first


def _resolve_api_key(raw: str | None) -> str:
    if raw:
        return raw.strip()
    env_value = os.getenv("ACESTEP_ZAI_API_KEY", "").strip()
    if env_value:
        return env_value
    return getpass.getpass("Z.ai API key (input hidden): ").strip()


def _resolve_intent(raw: str | None) -> str:
    if raw and raw.strip():
        return raw.strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            return stdin_text
    raise ValueError("Intent is required. Use --intent or pipe text via stdin.")


def _print_or_write_json(payload: dict, output_path: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")
        return
    print(text)


def _cmd_set_key(args: argparse.Namespace) -> int:
    store = _build_store(args.store_path)
    api_key = _resolve_api_key(args.api_key)
    passphrase = _resolve_passphrase(args.passphrase, confirm=True)
    store.save(secret=api_key, passphrase=passphrase)
    print(f"Stored encrypted external AI API key at: {store.secret_path}")
    return 0


def _cmd_clear_key(args: argparse.Namespace) -> int:
    store = _build_store(args.store_path)
    store.clear()
    print(f"Cleared encrypted key at: {store.secret_path}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    store = _build_store(args.store_path)
    api_key = args.api_key
    if not api_key:
        passphrase = _resolve_passphrase(args.passphrase, confirm=False)
        api_key = store.load(passphrase=passphrase)

    intent = _resolve_intent(args.intent)
    model = args.model or os.getenv("ACESTEP_ZAI_MODEL", DEFAULT_ZAI_MODEL)
    base_url = args.base_url or os.getenv("ACESTEP_ZAI_BASE_URL", DEFAULT_ZAI_BASE_URL)

    plan = request_external_ai_plan(
        api_key=api_key,
        intent=intent,
        model=model,
        base_url=base_url,
        timeout_sec=args.timeout,
        task_focus=args.task_focus,
    )

    payload = (
        build_acestep_generation_payload(plan)
        if args.acestep_payload
        else plan.to_dict()
    )
    _print_or_write_json(payload=payload, output_path=args.out)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    set_key = sub.add_parser(
        "set-key",
        help="Store external AI API key in encrypted user-local storage",
    )
    set_key.add_argument("--api-key", help="API key value (omit to input hidden)")
    set_key.add_argument("--passphrase", help="Secret-store passphrase (omit to prompt)")
    set_key.add_argument("--store-path", help="Override encrypted key file path")
    set_key.set_defaults(func=_cmd_set_key)

    clear_key = sub.add_parser(
        "clear-key",
        help="Delete encrypted external AI key from storage",
    )
    clear_key.add_argument("--store-path", help="Override encrypted key file path")
    clear_key.set_defaults(func=_cmd_clear_key)

    plan = sub.add_parser(
        "plan",
        help="Generate caption/lyrics/metadata plan from intent via external AI",
    )
    plan.add_argument("--intent", help="Natural-language intent (or pipe via stdin)")
    plan.add_argument(
        "--task-focus",
        default="all",
        help="Task focus: all|caption|lyrics|planning",
    )
    plan.add_argument("--model", help=f"Z.ai model (default: {DEFAULT_ZAI_MODEL})")
    plan.add_argument("--base-url", help=f"Z.ai endpoint (default: {DEFAULT_ZAI_BASE_URL})")
    plan.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    plan.add_argument("--api-key", help="API key override (skips encrypted store)")
    plan.add_argument("--passphrase", help="Secret-store passphrase (omit to prompt/env)")
    plan.add_argument("--store-path", help="Override encrypted key file path")
    plan.add_argument(
        "--acestep-payload",
        action="store_true",
        help="Emit ACE-Step request fragment with local LM toggles disabled",
    )
    plan.add_argument("--out", help="Write JSON output to file")
    plan.set_defaults(func=_cmd_plan)

    return parser


def main() -> int:
    """Run external AI text-task CLI."""
    parser = _build_parser()
    args = parser.parse_args()
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
