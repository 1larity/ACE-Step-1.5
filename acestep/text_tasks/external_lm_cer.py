"""Bulk caption enhancement review helpers for external LM providers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .external_lm_cer_prompts import DEFAULT_CER_PROMPTS, load_cer_prompts
from .external_lm_cer_support import (
    DEFAULT_OLLAMA_MAX_MODEL_GB,
    append_jsonl,
    configure_external_lm_env,
    discover_provider_models,
    filter_ollama_models_by_size,
    is_model_access_error,
    is_quota_like_api_error,
    resolve_coding_base_url,
    resolve_default_provider,
    resolve_provider_base_url,
)
from .external_lm_ollama_catalog import OllamaModelInfo, list_ollama_models


def run_cer_campaign(
    *,
    provider: str,
    protocol: str,
    prompt_file: Path | None,
    prompt_limit: int,
    repeat: int,
    user_metadata: dict[str, Any],
    lyrics: str,
    sleep_sec: float,
    output_path: Path,
    ollama_max_model_gb: float | None = DEFAULT_OLLAMA_MAX_MODEL_GB,
    models: list[str] | None = None,
    format_fn: Callable[..., Any] | None = None,
    discover_fn: Callable[..., tuple[list[str], str]] = discover_provider_models,
    ollama_catalog_fn: Callable[..., list[OllamaModelInfo]] = list_ollama_models,
) -> int:
    """Run a bulk CER sweep over provider models and write JSONL results."""

    if format_fn is None:
        from .external_lm_tasks import format_sample_with_external_provider

        format_fn = format_sample_with_external_provider

    prompts = load_cer_prompts(prompt_file, prompt_limit)
    base_url = resolve_provider_base_url(provider)
    selected_models, base_url = (models, base_url) if models else discover_fn(
        provider=provider,
        protocol=protocol,
        base_url=base_url,
    )
    model_size_by_name: dict[str, int | None] = {}
    if provider == "ollama" and ollama_max_model_gb is not None:
        catalog = ollama_catalog_fn(base_url=base_url)
        selected_models, skipped_models = filter_ollama_models_by_size(
            models=selected_models,
            catalog=catalog,
            max_model_gb=ollama_max_model_gb,
        )
        model_size_by_name = {item.name: item.size_bytes for item in catalog}
        for skipped_model, size_bytes in skipped_models:
            size_gb = "unknown"
            if size_bytes is not None:
                size_gb = f"{size_bytes / (1024**3):.2f}GB"
            print(
                f"[SKIP] model={skipped_model!r} reported_size={size_gb} exceeds or cannot satisfy "
                f"ollama_max_model_gb={ollama_max_model_gb}",
                flush=True,
            )
    if not selected_models:
        print("CER aborted: no models available after filtering.", flush=True)
        return 1

    if output_path.exists():
        output_path.unlink()

    total_success = 0
    total_failure = 0
    for model in selected_models:
        skip_current_model = False
        configure_external_lm_env(
            provider=provider,
            protocol=protocol,
            model=model,
            base_url=base_url,
        )
        print(
            f"CER model starting: provider={provider} model={model} prompts={len(prompts)} "
            f"repeat={repeat} base_url={base_url}",
            flush=True,
        )
        for prompt in prompts:
            for attempt in range(1, repeat + 1):
                started_at = time.time()
                record = {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "model_size_bytes": model_size_by_name.get(model),
                    "prompt": prompt,
                    "attempt": attempt,
                    "user_metadata": user_metadata,
                }
                try:
                    result = format_fn(
                        caption=prompt,
                        lyrics=lyrics,
                        user_metadata=user_metadata,
                        debug=True,
                    )
                except Exception as exc:
                    coding_base_url = resolve_coding_base_url(provider, base_url)
                    if coding_base_url and coding_base_url != base_url and is_quota_like_api_error(
                        str(exc)
                    ):
                        base_url = coding_base_url
                        configure_external_lm_env(
                            provider=provider,
                            protocol=protocol,
                            model=model,
                            base_url=base_url,
                        )
                        result = format_fn(
                            caption=prompt,
                            lyrics=lyrics,
                            user_metadata=user_metadata,
                            debug=True,
                        )
                        record["endpoint_switched_to_coding"] = True
                    else:
                        elapsed = round(time.time() - started_at, 3)
                        total_failure += 1
                        record.update(
                            {
                                "ok": False,
                                "elapsed_sec": elapsed,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        append_jsonl(output_path, record)
                        print(
                            f"[FAIL] model={model!r} prompt={prompt!r} attempt={attempt} "
                            f"elapsed={elapsed}s error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if is_model_access_error(str(exc)):
                            print(
                                f"[SKIP-MODEL] model={model!r} is unavailable for this account; "
                                "moving to the next model.",
                                flush=True,
                            )
                            skip_current_model = True
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
                        if skip_current_model:
                            break
                        continue

                elapsed = round(time.time() - started_at, 3)
                total_success += 1
                record.update(
                    {
                        "ok": True,
                        "elapsed_sec": elapsed,
                        "caption": result.caption,
                        "bpm": result.bpm,
                        "duration": result.duration,
                        "keyscale": result.keyscale,
                        "language": result.language,
                        "timesignature": result.timesignature,
                        "status_message": result.status_message,
                    }
                )
                append_jsonl(output_path, record)
                print(
                    f"[OK] model={model!r} prompt={prompt!r} attempt={attempt} "
                    f"elapsed={elapsed}s caption={result.caption[:120]!r}",
                    flush=True,
                )
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                if skip_current_model:
                    break
            if skip_current_model:
                break

    print(
        f"CER complete. successes={total_success} failures={total_failure} log={output_path}",
        flush=True,
    )
    return 0 if total_failure == 0 else 1
