"""Simple-mode generation-tab controls."""

from typing import Any

import gradio as gr

from acestep.constants import VALID_LANGUAGES
from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t


def build_simple_mode_controls() -> dict[str, Any]:
    """Create simple-mode prompt controls and sample action controls.

    Args:
        None.

    Returns:
        A component map containing simple-mode query inputs, language toggles, and action buttons.
    """

    with gr.Group(visible=False, elem_classes=["has-info-container"]) as simple_mode_group:
        create_help_button("generation_simple")
        with gr.Row(equal_height=True):
            simple_query_input = gr.Textbox(
                label=t("generation.simple_query_label"),
                placeholder=t("generation.simple_query_placeholder"),
                lines=2,
                info=t("generation.simple_query_info"),
                elem_classes=["has-info-container"],
                scale=9,
            )
            with gr.Column(scale=1):
                simple_vocal_language = gr.Dropdown(
                    choices=[
                        (lang if lang != "unknown" else "Instrumental / auto", lang)
                        for lang in VALID_LANGUAGES
                    ],
                    value="unknown",
                    allow_custom_value=True,
                    label=t("generation.simple_vocal_language_label"),
                    interactive=True,
                    scale=1,
                )
                simple_instrumental_checkbox = gr.Checkbox(
                    label=t("generation.instrumental_label"),
                    value=False,
                    scale=1,
                )
            with gr.Column(scale=1, min_width=80, elem_classes="icon-btn-wrap"):
                random_desc_btn = gr.Button(
                    t("generation.sample_btn"),
                    variant="secondary",
                    size="lg",
                )
        with gr.Row(equal_height=True):
            create_sample_btn = gr.Button(
                t("generation.create_sample_btn"),
                variant="primary",
                size="lg",
            )
    return {
        "simple_mode_group": simple_mode_group,
        "simple_query_input": simple_query_input,
        "simple_vocal_language": simple_vocal_language,
        "simple_instrumental_checkbox": simple_instrumental_checkbox,
        "random_desc_btn": random_desc_btn,
        "create_sample_btn": create_sample_btn,
    }
