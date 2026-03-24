"""Bulk caption enhancement review runner for external LM providers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acestep.text_tasks.external_lm_cer import resolve_default_provider, run_cer_campaign
from acestep.text_tasks.external_lm_providers import get_external_provider_profile


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the CER runner."""

    parser = argparse.ArgumentParser(description="Run bulk caption enhancement reviews.")
    parser.add_argument("--provider", type=str, default=resolve_default_provider())
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-limit", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--ollama-max-size-gb", type=float, default=6.0)
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--keyscale", type=str)
    parser.add_argument("--timesignature", type=str)
    parser.add_argument("--lyrics", type=str, default="")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/acestep_external_lm_cer_results.jsonl"),
    )
    return parser.parse_args()


def build_user_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Build fixed CER metadata constraints from CLI args."""

    metadata: dict[str, Any] = {}
    if args.bpm is not None:
        metadata["bpm"] = args.bpm
    if args.duration is not None:
        metadata["duration"] = args.duration
    if args.keyscale:
        metadata["keyscale"] = args.keyscale
    if args.timesignature:
        metadata["timesignature"] = args.timesignature
    return metadata


def main() -> int:
    """Run a CER campaign for the selected provider."""

    args = parse_args()
    provider = (args.provider or "").strip().lower()
    profile = get_external_provider_profile(provider)
    return run_cer_campaign(
        provider=profile.provider_id,
        protocol=profile.protocol,
        prompt_file=args.prompt_file,
        prompt_limit=args.prompt_limit,
        repeat=args.repeat,
        user_metadata=build_user_metadata(args),
        lyrics=args.lyrics,
        sleep_sec=args.sleep,
        output_path=args.output,
        ollama_max_model_gb=args.ollama_max_size_gb,
        models=[item.strip() for item in args.model if item.strip()] or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
