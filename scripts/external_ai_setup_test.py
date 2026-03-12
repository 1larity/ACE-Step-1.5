"""Tests for the external AI setup CLI script."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_SCRIPT_PATH = Path(__file__).with_name("external_ai_setup.py")
_SPEC = importlib.util.spec_from_file_location("external_ai_setup_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load external_ai_setup.py for testing.")
external_ai_setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(external_ai_setup)


class ExternalAiSetupScriptTests(unittest.TestCase):
    """Validate non-sensitive console output from the external AI setup CLI."""

    @patch.object(external_ai_setup, "store_runtime_passphrase", return_value=(True, "stored passphrase in keyring"))
    @patch.object(external_ai_setup, "_resolve_passphrase", return_value="top-secret-passphrase")
    @patch.object(external_ai_setup, "_resolve_api_key", return_value="sk-secret")
    @patch.object(external_ai_setup, "_build_store")
    def test_setup_does_not_print_keyring_message_or_secrets(
        self,
        build_store_mock,
        _resolve_api_key_mock,
        _resolve_passphrase_mock,
        _store_runtime_passphrase_mock,
    ) -> None:
        """Setup should print generic key-storage status without echoing sensitive values."""
        store = build_store_mock.return_value
        store.secret_path = Path("/tmp/external_ai.enc")
        args = argparse.Namespace(
            store_path=None,
            api_key=None,
            passphrase=None,
            save_passphrase=True,
            model=None,
            base_url=None,
        )

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = external_ai_setup._cmd_setup(args)

        output = stream.getvalue()
        self.assertEqual(0, result)
        self.assertIn("Saved runtime passphrase in system key storage.", output)
        self.assertNotIn("stored passphrase in keyring", output)
        self.assertNotIn("top-secret-passphrase", output)
        self.assertNotIn("sk-secret", output)

    @patch.object(external_ai_setup, "resolve_runtime_passphrase", return_value=None)
    @patch.object(external_ai_setup, "_build_store")
    def test_doctor_redacts_secret_lookup_identity(self, build_store_mock, _resolve_runtime_passphrase_mock) -> None:
        """Doctor output should describe secret lookup configuration without echoing raw identifiers."""
        store = build_store_mock.return_value
        store.secret_path = Path("/tmp/external_ai.enc")
        store.exists.return_value = False
        args = argparse.Namespace(store_path=None)

        with patch.dict(
            "os.environ",
            {
                "ACESTEP_EXTERNAL_AI_SECRET_SERVICE": "custom-service",
                "ACESTEP_EXTERNAL_AI_SECRET_USERNAME": "custom-user",
            },
            clear=False,
        ):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = external_ai_setup._cmd_doctor(args)

        output = stream.getvalue()
        self.assertEqual(2, result)
        self.assertIn("Secret lookup identity configured: custom", output)
        self.assertNotIn("custom-service", output)
        self.assertNotIn("custom-user", output)


if __name__ == "__main__":
    unittest.main()
