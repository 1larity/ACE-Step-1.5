"""UI helper registration blocks for generation service wiring."""

from __future__ import annotations

from typing import Any, Callable


def register_auto_checkbox_handlers(
    *,
    generation_section: dict[str, Any],
    gen_h: Any,
    auto_checkbox_outputs: list[Any],
) -> None:
    """Register metadata auto-checkbox handlers."""
    auto_field_map = {
        "bpm_auto": ("bpm", "bpm"),
        "key_auto": ("key_scale", "key_scale"),
        "timesig_auto": ("time_signature", "time_signature"),
        "vocal_lang_auto": ("vocal_language", "vocal_language"),
        "duration_auto": ("audio_duration", "audio_duration"),
    }
    for auto_key, (field_name, comp_key) in auto_field_map.items():
        generation_section[auto_key].change(
            fn=lambda checked, fn=field_name: gen_h.on_auto_checkbox_change(checked, fn),
            inputs=[generation_section[auto_key]],
            outputs=[generation_section[comp_key]],
        )
    generation_section["reset_all_auto_btn"].click(fn=gen_h.reset_all_auto, outputs=auto_checkbox_outputs)


def register_visibility_handlers(
    *,
    generation_section: dict[str, Any],
    results_section: dict[str, Any],
    gen_h: Any,
    llm_handler: Any,
    sync_lm_selection_from_init_checkbox: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Register visibility and local/external LM consistency handlers."""
    generation_section["init_llm_checkbox"].change(
        fn=gen_h.update_negative_prompt_visibility,
        inputs=[generation_section["init_llm_checkbox"]],
        outputs=[generation_section["lm_negative_prompt"]],
    )
    generation_section["init_llm_checkbox"].change(
        fn=lambda init_llm, lm_model_path: sync_lm_selection_from_init_checkbox(init_llm, lm_model_path, llm_handler=llm_handler),
        inputs=[generation_section["init_llm_checkbox"], generation_section["lm_model_path"]],
        outputs=[generation_section["init_llm_checkbox"], generation_section["lm_model_path"]],
    )
    generation_section["batch_size_input"].change(
        fn=gen_h.update_audio_components_visibility,
        inputs=[generation_section["batch_size_input"]],
        outputs=[
            results_section["audio_col_1"],
            results_section["audio_col_2"],
            results_section["audio_col_3"],
            results_section["audio_col_4"],
            results_section["audio_row_5_8"],
            results_section["audio_col_5"],
            results_section["audio_col_6"],
            results_section["audio_col_7"],
            results_section["audio_col_8"],
        ],
    )
