"""Credential persistence helpers for the external LM setup panel."""

from __future__ import annotations

import os

import gradio as gr

from acestep.text_tasks.external_lm_mode import resolve_external_api_key_for_runtime
from acestep.text_tasks.passphrase_store import store_runtime_passphrase
from acestep.text_tasks.secure_secret_store import SecretStoreError

from .external_lm_setup_defaults import as_markdown_status, resolve_secret_store_for_provider


def store_provider_credentials(
    *,
    profile,
    api_key_value: str,
    passphrase_value: str,
    save_passphrase_to_keyring: bool,
    status_lines: list[str],
) -> tuple[str, dict, dict, dict] | None:
    """Persist provider credentials and update status lines."""

    if api_key_value:
        os.environ[profile.api_key_env] = api_key_value
        if passphrase_value:
            try:
                resolve_secret_store_for_provider(profile.provider_id).save(
                    secret=api_key_value,
                    passphrase=passphrase_value,
                )
                status_lines.append("Encrypted API key stored on disk.")
                os.environ["ACESTEP_GLM_STORE_PASSPHRASE"] = passphrase_value
            except SecretStoreError as exc:
                message = f"Failed to store encrypted API key: {exc}"
                gr.Warning(message)
                return as_markdown_status(message), gr.update(), gr.update(), gr.update()
            if save_passphrase_to_keyring:
                ok, keyring_message = store_runtime_passphrase(passphrase_value)
                status_lines.append(keyring_message)
                if not ok:
                    status_lines.append(
                        "Passphrase keyring persistence failed; runtime still works for this session."
                    )
            else:
                status_lines.append(
                    "Passphrase persistence disabled; available only for the current session."
                )
        elif profile.api_key_required:
            status_lines.append(
                "API key not encrypted: add a passphrase to persist it across restarts."
            )
        return None

    if profile.api_key_required:
        try:
            resolve_external_api_key_for_runtime(profile.provider_id)
            status_lines.append("Using existing provider credentials from env or encrypted store.")
        except SecretStoreError as exc:
            message = f"{profile.label} API key is required: {exc}"
            gr.Warning(message)
            return as_markdown_status(message), gr.update(), gr.update(), gr.update()
    else:
        status_lines.append("Provider does not require an API key by default.")
    return None
