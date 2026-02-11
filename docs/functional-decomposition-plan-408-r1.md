# Functional Decomposition Plan Update (Discussion #408)

Date: 2026-02-11
Source plan: https://github.com/ace-step/ACE-Step-1.5/discussions/408
Compared PR: https://github.com/ace-step/ACE-Step-1.5/pull/431

## 1) Progress vs plan (where you got to)

Status: Partial execution of the "Decompose handler.py incrementally" phase, with additional LoRA-domain extraction completed.

Completed from plan intent:
- Started handler monolith split behind a stable facade (`acestep/core/generation/handler/__init__.py`).
- Extracted LoRA responsibility from `acestep/handler.py` into focused modules.
- Added progress module (`acestep/core/generation/handler/progress.py`) as planned target capability.
- Preserved behavior while refactoring and added tests for service and integration paths.

Not started yet from original sequence:
- UI wiring decomposition (`ui/gradio/events/*_wiring.py`).
- API server decomposition (`acestep/api/http/*`, `acestep/api/jobs/*`).
- Inference/logits decomposition (`core/generation/inference/*`, `core/generation/logits/*`).
- LLM runtime split (`core/llm/runtime/*`).

## 2) Required variance integrated into the plan

Reason for variance:
- During PR #431, LoRA behavior regressions required immediate stabilization while decomposing.
- This introduced a service-oriented LoRA domain not explicitly represented in the original structure tree.

### 2.1 Updated structure tree (variance-aware)

acestep/
|- core/
|  |- generation/
|  |  |- handler/
|  |  |  |- __init__.py
|  |  |  |- lora_manager.py
|  |  |  |- progress.py
|  |  |  \- lora/
|  |  |     |- adapter_discovery.py
|  |  |     |- controls.py
|  |  |     |- lifecycle.py
|  |  |     |- registry_builder.py
|  |  |     |- registry_state.py
|  |  |     \- scale_apply.py
|  |  \- logits/                         # Planned, not yet executed
|  |- lora/                              # Variance: promoted reusable LoRA domain
|  |  |- __init__.py
|  |  |- introspection.py
|  |  |- registry.py
|  |  |- scaling.py
|  |  \- service.py
|  \- llm/                               # Planned, not yet executed
|- api/                                   # Planned, not yet executed
|- ui/                                    # Planned, not yet executed
|- dataset/                               # Planned, not yet executed
\- training/
   \- lora/                               # Keep for training-specific LoRA config/apply/save_load

### 2.2 Updated breakdown tree (variance-aware)

handler.py decomposition now proceeds in two LoRA layers:

1) Handler adapters (`acestep/core/generation/handler/lora/*`)
- UI/runtime-facing lifecycle and control glue.
- Minimal orchestration and state handoff.

2) Reusable LoRA domain service (`acestep/core/lora/*`)
- Introspection, deterministic registry, scaling semantics.
- Service facade consumed by handler adapters.

This replaces the earlier single-node assumption (`handler/lora_manager.py` only).

## 3) Revised segment order (rolling strategy)

Segment A (done): LoRA extraction from handler + stabilization fixes.
Segment B (done): Add handler progress module and callback path foundation.
Segment C (next): Continue handler split (`init_service.py`, `diffusion.py`, `decode.py`, `io_audio.py`) while preserving facade.
Segment D: UI wiring split for generation/training handlers.
Segment E: API monolith split.
Segment F: Inference/logits split.
Segment G: LLM runtime split and provider contracts.

Safety gate for each segment remains:
- Gradio startup.
- API `/health`.
- One representative generation path.
- Import smoke for touched modules.
- Touched-subsystem tests.

## 4) Explicit PR #431 alignment summary

Aligned:
- "Decompose `handler.py` incrementally behind stable facade."
- "Avoid behavior changes without tests."
- "Enable groundwork for future multi-LoRA."

Diverged (intentional, required):
- Introduced `acestep/core/lora/` domain earlier than original tree.
- Prioritized LoRA stability/regression fixes over pure move-only refactor.
- Expanded handler LoRA internals to multiple focused files, not just `lora_manager.py`.

## ReRevisions

1. Added explicit `acestep/core/lora/` domain to structure tree as first-class reusable module set.
2. Updated handler decomposition model from single `lora_manager.py` node to two-layer adapter+service architecture.
3. Inserted progress module completion as an early completed segment instead of a later optional extraction.
4. Reordered execution sequence so remaining handler split happens before UI/API decomposition.
5. Clarified that `training/lora/*` remains training-specific while generation/runtime LoRA logic lives under `core/lora/*`.
