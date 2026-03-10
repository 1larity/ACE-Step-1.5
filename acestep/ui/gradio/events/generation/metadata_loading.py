"""Metadata loading and example sampling for generation handlers.

Contains functions for loading generation parameters from JSON files
and sampling random examples from the examples directory.
"""

from __future__ import annotations

import glob
import json
import os
import random

import gradio as gr

from acestep.gpu_config import get_global_gpu_config
from acestep.inference import understand_music
from acestep.text_tasks.external_lm_mode import is_external_lm_active
from acestep.ui.gradio.i18n import t

from .metadata_loading_examples import get_project_root_from, load_json_file
from .metadata_loading_parsing import clamp_batch_size, normalize_optional_text, parse_optional_float, parse_optional_int
from .validation import clamp_duration_to_gpu_limit


def load_metadata(file_obj, llm_handler=None):
    """Load generation parameters from a JSON file.

    Args:
        file_obj: Uploaded file object.
        llm_handler: LLM handler instance (optional, for GPU duration limit check).
    """
    if file_obj is None:
        gr.Warning(t("messages.no_file_selected"))
        return [None] * 37 + [False]
    try:
        filepath = file_obj.name if hasattr(file_obj, 'name') else file_obj
        metadata = load_json_file(filepath)
        vocal_language = metadata.get('vocal_language', 'unknown')
        audio_duration = parse_optional_float(metadata.get('duration', -1), default=-1)
        if audio_duration not in (None, -1):
            audio_duration = clamp_duration_to_gpu_limit(audio_duration, llm_handler)
        elif audio_duration is None:
            audio_duration = -1
        gpu_config = get_global_gpu_config()
        lm_initialized = llm_handler.llm_initialized if llm_handler else False
        max_batch_size = gpu_config.max_batch_size_with_lm if lm_initialized else gpu_config.max_batch_size_without_lm
        batch_size = clamp_batch_size(metadata.get('batch_size', 2), max_batch_size)
        think = metadata.get('thinking', True)
        lm_ok = (llm_handler.llm_initialized if llm_handler else False) or is_external_lm_active()
        audio_codes = metadata.get('audio_codes', '')
        if think and (not lm_ok or (audio_codes and audio_codes.strip())):
            if not lm_ok:
                gr.Warning(t("messages.think_requires_lm"))
            think = False
        gr.Info(t("messages.params_loaded", filename=os.path.basename(filepath)))
        return (
            metadata.get('task_type', 'text2music'), metadata.get('caption', ''), metadata.get('lyrics', ''), vocal_language,
            parse_optional_int(metadata.get('bpm')), metadata.get('keyscale', ''), metadata.get('timesignature', ''),
            audio_duration, batch_size, metadata.get('inference_steps', 8), metadata.get('guidance_scale', 7.0), metadata.get('seed', '-1'),
            False, metadata.get('use_adg', False), metadata.get('cfg_interval_start', 0.0), metadata.get('cfg_interval_end', 1.0),
            metadata.get('shift', 3.0), metadata.get('infer_method', 'ode'), metadata.get('timesteps', '') or '', metadata.get('audio_format', 'flac'),
            metadata.get('lm_temperature', 0.85), metadata.get('lm_cfg_scale', 2.0), metadata.get('lm_top_k', 0), metadata.get('lm_top_p', 0.9),
            metadata.get('lm_negative_prompt', 'NO USER INPUT'), metadata.get('use_cot_metas', True), metadata.get('use_cot_caption', True), metadata.get('use_cot_language', True),
            metadata.get('audio_cover_strength', 1.0), metadata.get('cover_noise_strength', 0.0), think, audio_codes, metadata.get('repainting_start', 0.0),
            metadata.get('repainting_end', -1), metadata.get('track_name'), metadata.get('complete_track_classes', []), metadata.get('instrumental', False), True,
        )
    except json.JSONDecodeError as e:
        gr.Warning(t("messages.invalid_json", error=str(e)))
    except Exception as e:
        gr.Warning(t("messages.load_error", error=str(e)))
    return [None] * 37 + [False]


def _get_project_root() -> str:
    """Return the project root directory (5 levels up from this file)."""
    return get_project_root_from(os.path.abspath(__file__))


def _choose_random_example_file(task_type: str) -> str:
    project_root = _get_project_root()
    examples_dir = os.path.join(project_root, "examples", task_type)
    if not os.path.exists(examples_dir):
        raise FileNotFoundError(f"Examples directory not found: examples/{task_type}/")
    json_files = glob.glob(os.path.join(examples_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in examples/{task_type}/")
    return random.choice(json_files)


def load_random_example(task_type: str, llm_handler=None):
    """Load a random example from the task-specific examples directory."""
    try:
        selected_file = _choose_random_example_file(task_type)
        data = load_json_file(selected_file)
        think_value = data.get('think', True) if isinstance(data.get('think', True), bool) else True
        lm_ok = (llm_handler.llm_initialized if llm_handler else False) or is_external_lm_active()
        if think_value and not lm_ok:
            think_value = False
            gr.Warning(t("messages.think_requires_lm"))
        gr.Info(t("messages.example_loaded", filename=os.path.basename(selected_file)))
        return (
            normalize_optional_text(data.get('caption', data.get('prompt', '')), empty_values={None}),
            normalize_optional_text(data.get('lyrics', ''), empty_values={None}),
            think_value,
            parse_optional_int(data.get('bpm')),
            clamp_duration_to_gpu_limit(parse_optional_float(data.get('duration')), llm_handler) if parse_optional_float(data.get('duration')) is not None else None,
            normalize_optional_text(data.get('keyscale', ''), empty_values={None, 'N/A'}),
            normalize_optional_text(data.get('language', ''), empty_values={None, 'N/A'}),
            normalize_optional_text(data.get('timesignature', ''), empty_values={None, 'N/A'}),
        )
    except FileNotFoundError as e:
        gr.Warning(str(e))
    except json.JSONDecodeError as e:
        filename = os.path.basename(selected_file) if 'selected_file' in locals() else ''
        gr.Warning(t("messages.example_failed", filename=filename, error=str(e)))
    except Exception as e:
        gr.Warning(t("messages.example_error", error=str(e)))
    return "", "", True, None, None, "", "", ""


def sample_example_smart(llm_handler, task_type: str, constrained_decoding_debug: bool = False):
    """Smart sample: use LM if initialized, else fall back to examples directory."""
    if not llm_handler.llm_initialized:
        return load_random_example(task_type)
    try:
        result = understand_music(llm_handler=llm_handler, audio_codes="NO USER INPUT", temperature=0.85, use_constrained_decoding=True, constrained_decoding_debug=constrained_decoding_debug)
        if result.success:
            gr.Info(t("messages.lm_generated"))
            return result.caption, result.lyrics, True, result.bpm, clamp_duration_to_gpu_limit(result.duration, llm_handler), result.keyscale, result.language, result.timesignature
    except Exception:
        pass
    gr.Warning(t("messages.lm_fallback"))
    return load_random_example(task_type)


def load_random_simple_description():
    """Load a random description from the simple_mode examples directory."""
    try:
        selected_file = _choose_random_example_file("simple_mode")
        data = load_json_file(selected_file)
        vocal_language = data.get('vocal_language', 'unknown')
        if isinstance(vocal_language, list):
            vocal_language = vocal_language[0] if vocal_language else 'unknown'
        gr.Info(t("messages.simple_example_loaded", filename=os.path.basename(selected_file)))
        return data.get('description', ''), data.get('instrumental', False), vocal_language
    except FileNotFoundError as e:
        gr.Warning(str(e))
    except json.JSONDecodeError as e:
        filename = os.path.basename(selected_file) if 'selected_file' in locals() else ''
        gr.Warning(t("messages.example_failed", filename=filename, error=str(e)))
    except Exception as e:
        gr.Warning(t("messages.example_error", error=str(e)))
    return gr.update(), gr.update(), gr.update()
