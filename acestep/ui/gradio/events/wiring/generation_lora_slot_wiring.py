"""Helpers for multi-slot LoRA event wiring in the generation settings UI."""

from collections.abc import Sequence
from typing import Any

import gradio as gr


DEFAULT_LORA_STATUS = "No LoRA loaded"


def adapter_name_for_slot(slot_index: int) -> str:
    """Return the stable adapter name used for a UI slot."""
    return f"slot_{slot_index + 1}"


def _normalized_visibility_state(visible_slots: Sequence[bool], max_slots: int) -> list[bool]:
    """Return a bounded mutable visibility list with slot 1 always present."""
    normalized = [bool(state) for state in visible_slots[:max_slots]]
    normalized.extend([False] * (max_slots - len(normalized)))
    if normalized:
        normalized[0] = True
    return normalized


def _tab_updates(visible_slots: Sequence[bool]) -> list[dict[str, Any]]:
    """Build Gradio visibility updates for each slot tab."""
    return [gr.update(visible=visible) for visible in visible_slots]


def _selected_tab_id(visible_slots: Sequence[bool], preferred_index: int) -> str:
    """Return the selected tab id closest to the preferred visible slot."""
    if not visible_slots:
        return "lora-slot-1"

    max_index = len(visible_slots) - 1
    candidate = min(max(preferred_index, 0), max_index)
    for index in range(candidate, -1, -1):
        if visible_slots[index]:
            return f"lora-slot-{index + 1}"
    for index, visible in enumerate(visible_slots):
        if visible:
            return f"lora-slot-{index + 1}"
    return "lora-slot-1"


def reveal_next_lora_slot(visible_slots: Sequence[bool]) -> tuple[Any, ...]:
    """Show the next hidden LoRA slot tab and select it."""
    updated_visibility = _normalized_visibility_state(visible_slots, len(visible_slots))
    selected_index = 0
    for index, visible in enumerate(updated_visibility):
        if not visible:
            updated_visibility[index] = True
            selected_index = index
            break
        selected_index = index

    return (
        updated_visibility,
        sum(updated_visibility),
        gr.update(selected=_selected_tab_id(updated_visibility, selected_index)),
        *_tab_updates(updated_visibility),
    )


def _loaded_adapter_names(dit_handler: Any) -> set[str]:
    """Return the set of currently loaded adapter names."""
    status = dit_handler.get_lora_status()
    return {str(name) for name in status.get("adapters", [])}


def _is_success_message(message: str) -> bool:
    """Return whether a handler status message represents a successful operation."""
    stripped = str(message).strip()
    return not stripped.startswith(("?", "??"))


def load_lora_slot(dit_handler: Any, slot_index: int, lora_path: str, scale: float) -> tuple[str, dict[str, Any]]:
    """Load or replace the adapter assigned to a specific UI slot."""
    adapter_name = adapter_name_for_slot(slot_index)
    if adapter_name in _loaded_adapter_names(dit_handler):
        dit_handler.remove_lora(adapter_name)

    message = dit_handler.add_lora(lora_path, adapter_name=adapter_name)
    if not _is_success_message(message):
        return message, gr.update(value=False)

    dit_handler.set_use_lora(True)
    scale_message = dit_handler.set_lora_scale(adapter_name, scale)
    status_message = message if _is_success_message(scale_message) else f"{message} | {scale_message}"
    return status_message, gr.update(value=True)


def unload_lora_slot(dit_handler: Any, slot_index: int) -> tuple[str, dict[str, Any]]:
    """Unload the adapter assigned to a specific UI slot."""
    adapter_name = adapter_name_for_slot(slot_index)
    if adapter_name not in _loaded_adapter_names(dit_handler):
        return "No LoRA adapter loaded in this tab.", gr.update(value=False)

    return dit_handler.remove_lora(adapter_name), gr.update(value=False)


def remove_lora_slot(
    dit_handler: Any,
    slot_index: int,
    visible_slots: Sequence[bool],
) -> tuple[Any, ...]:
    """Remove a secondary LoRA slot, unloading its adapter and hiding the tab."""
    updated_visibility = _normalized_visibility_state(visible_slots, len(visible_slots))
    adapter_name = adapter_name_for_slot(slot_index)

    if adapter_name in _loaded_adapter_names(dit_handler):
        message = dit_handler.remove_lora(adapter_name)
    else:
        message = "LoRA tab removed."

    if slot_index < len(updated_visibility):
        updated_visibility[slot_index] = False

    return (
        updated_visibility,
        sum(updated_visibility),
        gr.update(selected=_selected_tab_id(updated_visibility, slot_index - 1)),
        *_tab_updates(updated_visibility),
        gr.update(value=""),
        gr.update(value=False),
        gr.update(value=1.0),
        gr.update(value=DEFAULT_LORA_STATUS),
        gr.update(value=message),
    )


def toggle_lora_slot_enabled(
    dit_handler: Any,
    slot_index: int,
    scale: float,
    enabled_states: Sequence[bool],
) -> str:
    """Enable or disable a slot by applying its scale or zeroing it out."""
    adapter_name = adapter_name_for_slot(slot_index)
    if adapter_name not in _loaded_adapter_names(dit_handler):
        return "No LoRA adapter loaded in this tab."

    enabled = bool(enabled_states[slot_index]) if slot_index < len(enabled_states) else False
    if enabled:
        dit_handler.set_use_lora(True)
        return dit_handler.set_lora_scale(adapter_name, scale)

    message = dit_handler.set_lora_scale(adapter_name, 0.0)
    if not any(bool(state) for index, state in enumerate(enabled_states) if index != slot_index):
        dit_handler.set_use_lora(False)
    return message


def set_lora_slot_scale(dit_handler: Any, slot_index: int, scale: float, enabled: bool) -> str:
    """Apply scale immediately for enabled slots and preserve disabled-slot values in the UI."""
    adapter_name = adapter_name_for_slot(slot_index)
    if adapter_name not in _loaded_adapter_names(dit_handler):
        return "No LoRA adapter loaded in this tab."

    if not enabled:
        return f"LoRA scale ({adapter_name}): {float(scale):.2f} (stored while disabled)"

    dit_handler.set_use_lora(True)
    return dit_handler.set_lora_scale(adapter_name, scale)


def register_lora_slot_handlers(generation_section: dict[str, Any], dit_handler: Any) -> None:
    """Register multi-slot LoRA UI events against the handler."""
    lora_slots = generation_section["lora_slots"]
    lora_enable_inputs = [slot["use_lora_checkbox"] for slot in lora_slots]
    tab_outputs = [generation_section["lora_slot_visibility_state"], generation_section["lora_slot_count_state"], generation_section["lora_tabs"], *generation_section["lora_tab_items"]]

    generation_section["lora_add_slot_btn"].click(
        fn=reveal_next_lora_slot,
        inputs=[generation_section["lora_slot_visibility_state"]],
        outputs=tab_outputs,
    )

    for slot_index, slot in enumerate(lora_slots):
        slot["load_lora_btn"].click(
            fn=lambda lora_path, scale, handler=dit_handler, slot_index=slot_index: load_lora_slot(
                handler, slot_index, lora_path, scale
            ),
            inputs=[slot["lora_path"], slot["lora_scale_slider"]],
            outputs=[slot["lora_status"], slot["use_lora_checkbox"]],
        )
        slot["unload_lora_btn"].click(
            fn=lambda handler=dit_handler, slot_index=slot_index: unload_lora_slot(handler, slot_index),
            outputs=[slot["lora_status"], slot["use_lora_checkbox"]],
        )
        slot["use_lora_checkbox"].change(
            fn=lambda scale, *enabled_states, handler=dit_handler, slot_index=slot_index: toggle_lora_slot_enabled(
                handler, slot_index, scale, enabled_states
            ),
            inputs=[slot["lora_scale_slider"], *lora_enable_inputs],
            outputs=[slot["lora_status"]],
        )
        slot["lora_scale_slider"].change(
            fn=lambda scale, enabled, handler=dit_handler, slot_index=slot_index: set_lora_slot_scale(
                handler, slot_index, scale, enabled
            ),
            inputs=[slot["lora_scale_slider"], slot["use_lora_checkbox"]],
            outputs=[slot["lora_status"]],
        )
        if slot_index == 0:
            continue
        slot["remove_lora_btn"].click(
            fn=lambda visible_slots, handler=dit_handler, slot_index=slot_index: remove_lora_slot(
                handler, slot_index, visible_slots
            ),
            inputs=[generation_section["lora_slot_visibility_state"]],
            outputs=[
                *tab_outputs,
                slot["lora_path"],
                slot["use_lora_checkbox"],
                slot["lora_scale_slider"],
                slot["lora_status"],
                generation_section["lora_status"],
            ],
        )
