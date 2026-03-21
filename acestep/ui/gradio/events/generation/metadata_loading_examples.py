"""Example-loading helpers for generation metadata actions."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any


def get_project_root_from(current_file: str) -> str:
    """Return the ACE-Step project root from a generation event module path."""
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        )
    )


def choose_random_example_file(*, project_root: str, task_type: str) -> str:
    """Return a random example JSON path for a task type."""
    examples_dir = os.path.join(project_root, "examples", task_type)
    if not os.path.exists(examples_dir):
        raise FileNotFoundError(f"Examples directory not found: examples/{task_type}/")
    json_files = glob.glob(os.path.join(examples_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in examples/{task_type}/")
    return random.choice(json_files)


def load_json_file(path: str) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
