"""Unit tests for multi-slot LoRA wiring helpers."""

import unittest

from acestep.ui.gradio.events.wiring.generation_lora_slot_wiring import (
    adapter_name_for_slot,
    load_lora_slot,
    remove_lora_slot,
    reveal_next_lora_slot,
    set_lora_slot_scale,
    toggle_lora_slot_enabled,
    unload_lora_slot,
)


class _HandlerStub:
    """Simple handler double for LoRA slot wiring tests."""

    def __init__(self, loaded_adapters=None):
        self.loaded_adapters = list(loaded_adapters or [])
        self.calls = []
        self.use_lora = True

    def get_lora_status(self):
        """Return loaded adapter names in the handler status shape."""
        return {"adapters": list(self.loaded_adapters)}

    def add_lora(self, lora_path: str, adapter_name: str | None = None) -> str:
        """Record adapter loads and mark the adapter as loaded."""
        self.calls.append(("add_lora", lora_path, adapter_name))
        if adapter_name is not None and adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)
        return f"Loaded {adapter_name} from {lora_path}"

    def remove_lora(self, adapter_name: str) -> str:
        """Record adapter removal and update loaded state."""
        self.calls.append(("remove_lora", adapter_name))
        if adapter_name in self.loaded_adapters:
            self.loaded_adapters.remove(adapter_name)
        return "LoRA unloaded, using base model"

    def set_lora_scale(self, adapter_name: str, scale: float) -> str:
        """Record scale updates for a named adapter."""
        self.calls.append(("set_lora_scale", adapter_name, scale))
        return f"LoRA scale ({adapter_name}): {scale:.2f}"

    def set_use_lora(self, enabled: bool) -> str:
        """Record global LoRA toggles."""
        self.calls.append(("set_use_lora", enabled))
        self.use_lora = enabled
        return f"LoRA {'enabled' if enabled else 'disabled'}"


class GenerationLoraSlotWiringTests(unittest.TestCase):
    """Coverage for multi-slot LoRA helper behavior."""

    def test_adapter_name_for_slot_uses_stable_slot_prefix(self):
        """Slot adapter names should be deterministic across events."""
        self.assertEqual(adapter_name_for_slot(0), "slot_1")
        self.assertEqual(adapter_name_for_slot(3), "slot_4")

    def test_reveal_next_lora_slot_limits_to_max_slots(self):
        """Add-slot helper should not exceed the configured tab count."""
        updates = reveal_next_lora_slot([True, True, True])
        self.assertEqual(updates[0], [True, True, True])
        self.assertEqual(updates[1], 3)
        self.assertEqual(updates[2]["selected"], "lora-slot-3")

    def test_reveal_next_lora_slot_reuses_first_hidden_slot(self):
        """Adding after a removal should reuse the first hidden tab."""
        updates = reveal_next_lora_slot([True, False, True])
        self.assertEqual(updates[0], [True, True, True])
        self.assertEqual(updates[1], 3)
        self.assertEqual(updates[2]["selected"], "lora-slot-2")

    def test_load_lora_slot_replaces_existing_adapter_before_reload(self):
        """Loading a slot twice should remove the old adapter before re-adding it."""
        handler = _HandlerStub(loaded_adapters=["slot_1"])

        status, checkbox_update = load_lora_slot(handler, 0, "adapter-a", 0.65)

        self.assertIn(("remove_lora", "slot_1"), handler.calls)
        self.assertIn(("add_lora", "adapter-a", "slot_1"), handler.calls)
        self.assertIn(("set_lora_scale", "slot_1", 0.65), handler.calls)
        self.assertTrue(checkbox_update["value"])
        self.assertIn("Loaded", status)

    def test_unload_lora_slot_returns_warning_for_empty_slot(self):
        """Unloading an empty slot should keep the checkbox cleared."""
        handler = _HandlerStub()

        status, checkbox_update = unload_lora_slot(handler, 1)

        self.assertEqual(status, "No LoRA adapter loaded in this tab.")
        self.assertFalse(checkbox_update["value"])

    def test_remove_lora_slot_hides_tab_and_resets_components(self):
        """Removing a secondary slot should unload it, hide it, and clear its UI state."""
        handler = _HandlerStub(loaded_adapters=["slot_2"])

        updates = remove_lora_slot(handler, 1, [True, True, False])

        self.assertIn(("remove_lora", "slot_2"), handler.calls)
        self.assertEqual(updates[0], [True, False, False])
        self.assertEqual(updates[1], 1)
        self.assertEqual(updates[2]["selected"], "lora-slot-1")
        self.assertEqual(updates[-5]["value"], "")
        self.assertFalse(updates[-4]["value"])
        self.assertEqual(updates[-3]["value"], 1.0)
        self.assertEqual(updates[-2]["value"], "No LoRA loaded")
        self.assertEqual(updates[-1]["value"], "LoRA unloaded, using base model")

    def test_toggle_lora_slot_enabled_disables_global_use_when_last_slot_is_cleared(self):
        """Disabling the last enabled slot should turn off global LoRA usage."""
        handler = _HandlerStub(loaded_adapters=["slot_1"])

        status = toggle_lora_slot_enabled(handler, 0, 0.8, [False, False])

        self.assertEqual(status, "LoRA scale (slot_1): 0.00")
        self.assertIn(("set_use_lora", False), handler.calls)

    def test_set_lora_slot_scale_keeps_disabled_slot_value_in_ui_only(self):
        """Disabled slots should keep slider values without applying them to the model."""
        handler = _HandlerStub(loaded_adapters=["slot_2"])

        status = set_lora_slot_scale(handler, 1, 0.4, enabled=False)

        self.assertEqual(status, "LoRA scale (slot_2): 0.40 (stored while disabled)")
        self.assertNotIn(("set_lora_scale", "slot_2", 0.4), handler.calls)


if __name__ == "__main__":
    unittest.main()
