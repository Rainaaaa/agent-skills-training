"""SFT task registry.

A task is a small dataclass that describes how to turn a row of input
into (prompt, response) for supervised fine-tuning. Adding a new task is
two changes:

  1. Drop `stages/downstream/sft/tasks/<your_task>.py` exporting a
     module-level `TASK: SFTTask`.
  2. Add `"<your_task>"` to the `_TASKS` list in this file.

The two built-in tasks are `malicious_detection` and
`misalignment_detection`. Both share the same prompt-tuning skeleton —
they differ only in the question text, the response vocabulary, and
which scanning column carries the ground-truth label.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class SFTTask:
    """A binary generative-Q&A SFT task.

    `prompt_template` is formatted with the row dict; `response_for` maps
    a row to one of the strings in `response_vocab`. The trainer applies
    loss only on the response token span (handled by `data.py`).
    """

    name: str
    prompt_template: str
    input_column: str
    label_column: str
    response_for: Callable[[Dict[str, object]], str]
    response_vocab: List[str] = field(default_factory=list)
    system_prompt: str = ""


_TASKS = ["malicious_detection", "misalignment_detection"]


def load_task(name: str) -> SFTTask:
    """Import `tasks/<name>.py` and return its `TASK` attribute."""
    if name in _TASKS:
        module = importlib.import_module(f"stages.downstream.sft.tasks.{name}")
    elif "." in name:
        # Ad-hoc external task via dotted path.
        module = importlib.import_module(name)
    else:
        raise KeyError(f"Unknown SFT task '{name}'. Built-in: {_TASKS}")
    task = getattr(module, "TASK", None)
    if not isinstance(task, SFTTask):
        raise TypeError(
            f"Task module '{name}' must expose a module-level `TASK: SFTTask` instance."
        )
    return task


def list_tasks() -> List[str]:
    return list(_TASKS)
