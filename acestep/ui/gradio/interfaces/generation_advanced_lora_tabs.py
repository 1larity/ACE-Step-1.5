"""Helpers for tabbed LoRA adapter controls in the generation settings UI."""

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t

MAX_LORA_SLOTS = 8


def _get_lora_slot_label(index: int) -> str:
    """Return the visible label for a LoRA adapter tab."""
    if index == 1:
        return t("generation.lora_accordion_title")
    return f"{t('generation.lora_accordion_title')} {index}"


def _build_lora_slot(index: int) -> dict[str, Any]:
    """Create the component set for a single LoRA adapter slot."""
    tab_id = f"lora-slot-{index}"
    with gr.Tab(
        label=_get_lora_slot_label(index),
        id=tab_id,
        visible=index == 1,
        render_children=True,
    ) as tab:
        with gr.Row():
            lora_path = gr.Textbox(
                label=t("generation.lora_path_label"),
                placeholder=t("generation.lora_path_placeholder"),
                info=t("generation.lora_path_info"),
                scale=3,
            )
            load_lora_btn = gr.Button(t("generation.load_lora_btn"), variant="secondary", scale=1)
            unload_lora_btn = gr.Button(t("generation.unload_lora_btn"), variant="secondary", scale=1)
            remove_lora_btn = gr.Button(
                t("generation.lora_remove_tab_btn"),
                variant="secondary",
                scale=1,
                visible=index > 1,
            )
        with gr.Row():
            use_lora_checkbox = gr.Checkbox(
                label=t("generation.use_lora_label"),
                value=False,
                info=t("generation.use_lora_info"),
                scale=1,
            )
            lora_scale_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label=t("generation.lora_scale_label"),
                info=t("generation.lora_scale_info"),
                scale=2,
            )
        lora_status = gr.Textbox(
            label=t("generation.lora_status_label"),
            value=t("generation.lora_status_default"),
            interactive=False,
            lines=1,
            elem_classes=["no-tooltip"],
        )

    return {
        "tab": tab,
        "tab_id": tab_id,
        "lora_path": lora_path,
        "load_lora_btn": load_lora_btn,
        "unload_lora_btn": unload_lora_btn,
        "remove_lora_btn": remove_lora_btn,
        "use_lora_checkbox": use_lora_checkbox,
        "lora_scale_slider": lora_scale_slider,
        "lora_status": lora_status,
    }


def build_lora_tabbed_controls(max_slots: int = MAX_LORA_SLOTS) -> dict[str, Any]:
    """Create the add button, tabs container, and fixed set of LoRA slots."""
    with gr.Row():
        lora_add_slot_btn = gr.Button(t("generation.lora_add_adapter_btn"), variant="secondary")

    lora_slot_count_state = gr.State(1)
    lora_slot_visibility_state = gr.State([True] + ([False] * (max_slots - 1)))
    lora_slots: list[dict[str, Any]] = []
    lora_tab_items: list[Any] = []
    with gr.Tabs(elem_id="acestep-lora-tabs", elem_classes=["lora-tabs-scroll"]) as lora_tabs:
        for index in range(1, max_slots + 1):
            slot = _build_lora_slot(index)
            lora_slots.append(slot)
            lora_tab_items.append(slot["tab"])

    return {
        "lora_add_slot_btn": lora_add_slot_btn,
        "lora_slot_count_state": lora_slot_count_state,
        "lora_slot_visibility_state": lora_slot_visibility_state,
        "lora_slots": lora_slots,
        "lora_tabs": lora_tabs,
        "lora_tab_items": lora_tab_items,
    }
