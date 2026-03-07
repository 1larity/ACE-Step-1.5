"""Unit tests for CLI argument parsing helpers."""

from __future__ import annotations

import argparse
import unittest

from acestep.cli_args import parse_quantization_arg


class ParseQuantizationArgTests(unittest.TestCase):
    """Behavior tests for quantization CLI parsing."""

    def test_returns_none_for_none_aliases(self) -> None:
        """It treats ``none`` aliases as disabled quantization."""
        self.assertIsNone(parse_quantization_arg("none"))
        self.assertIsNone(parse_quantization_arg("None"))
        self.assertIsNone(parse_quantization_arg(" null "))
        self.assertIsNone(parse_quantization_arg(""))

    def test_returns_canonical_quantization_values(self) -> None:
        """It returns supported quantization values in canonical form."""
        self.assertEqual("int8_weight_only", parse_quantization_arg("int8_weight_only"))
        self.assertEqual("int8_weight_only", parse_quantization_arg("INT8_WEIGHT_ONLY"))
        self.assertEqual("int4_weight_only", parse_quantization_arg("int4_weight_only"))
        self.assertEqual("int4_weight_only", parse_quantization_arg(" Int4_Weight_Only "))

    def test_raises_for_invalid_value(self) -> None:
        """It raises ``ArgumentTypeError`` for unsupported values."""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_quantization_arg("fp8")


if __name__ == "__main__":
    unittest.main()
