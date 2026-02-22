"""Source-audio, track selection, and LM-code hint controls for generation tab."""

from typing import Any

import gradio as gr

from acestep.constants import TRACK_NAMES
from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t


def build_source_track_and_code_controls() -> dict[str, Any]:
    """Create source-audio, track-selector, and LM-code hint controls.

    Args:
        None.

    Returns:
        A component map containing source audio actions, track selectors, and LM code controls.
    """

    with gr.Row(equal_height=True, visible=False) as src_audio_row:
        src_audio = gr.Audio(label=t("generation.source_audio"), type="filepath", scale=10)
        with gr.Column(scale=1, min_width=80):
            analyze_btn = gr.Button(
                t("generation.analyze_btn"),
                variant="secondary",
                size="lg",
            )

    with gr.Group(visible=False) as extract_help_group:
        create_help_button("generation_extract")
    track_name = gr.Dropdown(
        choices=TRACK_NAMES,
        value=None,
        label=t("generation.track_name_label"),
        info=t("generation.track_name_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )
    with gr.Group(visible=False) as complete_help_group:
        create_help_button("generation_complete")
    complete_track_classes = gr.CheckboxGroup(
        choices=TRACK_NAMES,
        label=t("generation.track_classes_label"),
        info=t("generation.track_classes_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )

    with gr.Accordion(
        t("generation.lm_codes_hints"),
        open=False,
        visible=True,
        elem_classes=["has-info-container"],
    ) as text2music_audio_codes_group:
        with gr.Row(equal_height=True):
            lm_codes_audio_upload = gr.Audio(label=t("generation.source_audio"), type="filepath", scale=3)
            text2music_audio_code_string = gr.Textbox(
                label=t("generation.lm_codes_label"),
                placeholder=t("generation.lm_codes_placeholder"),
                lines=6,
                info=t("generation.lm_codes_info"),
                elem_classes=["has-info-container"],
                scale=6,
            )
        with gr.Row():
            convert_src_to_codes_btn = gr.Button(
                t("generation.convert_codes_btn"),
                variant="secondary",
                size="sm",
                scale=1,
            )
            transcribe_btn = gr.Button(
                t("generation.transcribe_btn"),
                variant="secondary",
                size="sm",
                scale=1,
            )

    return {
        "src_audio_row": src_audio_row,
        "src_audio": src_audio,
        "analyze_btn": analyze_btn,
        "extract_help_group": extract_help_group,
        "track_name": track_name,
        "complete_help_group": complete_help_group,
        "complete_track_classes": complete_track_classes,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "lm_codes_audio_upload": lm_codes_audio_upload,
        "text2music_audio_code_string": text2music_audio_code_string,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "transcribe_btn": transcribe_btn,
    }
