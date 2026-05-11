"""SFT task — binary malicious-skill detection.

Reads the scanning verdict (or human-reviewed label) from the
`overall_class` column and asks the model: *"Is this skill malicious?"*.

Expected response vocabulary: `{"malicious", "safe"}`.

Label mapping:
  - `MALICIOUS` (or `SUSPICIOUS`, configurable) → "malicious"
  - `SAFE` / unset                              → "safe"
  - `ERROR` rows are dropped at the data-loader level (no gold label).
"""

from __future__ import annotations

from stages.downstream.sft.tasks import SFTTask


SYSTEM_PROMPT = (
    "You are a precise security classifier. Given the contents of a Claude "
    "Code skill package, decide whether it is malicious. Reply with a single "
    "word — either `malicious` or `safe` — and nothing else."
)


PROMPT_TEMPLATE = (
    "Classify the following Claude Code skill package as `malicious` or `safe`.\n"
    "\n"
    "=== SKILL ===\n"
    "{skill_text}\n"
    "=== END SKILL ===\n"
    "\n"
    "Answer (one word, `malicious` or `safe`):"
)


_MAL_LABELS = {"MALICIOUS", "SUSPICIOUS"}


def _response(row):
    label = (row.get("overall_class") or row.get("malicious_label") or "").upper().strip()
    return "malicious" if label in _MAL_LABELS else "safe"


TASK = SFTTask(
    name="malicious_detection",
    prompt_template=PROMPT_TEMPLATE,
    input_column="skill_text",
    label_column="overall_class",
    response_for=_response,
    response_vocab=["malicious", "safe"],
    system_prompt=SYSTEM_PROMPT,
)
