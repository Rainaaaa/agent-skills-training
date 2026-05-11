"""SFT task — binary alignment detection.

Reads the scanning verdict (or human-reviewed label) from the
`alignment_class` column and asks the model: *"Is this skill aligned
with its description?"*.

Expected response vocabulary: `{"aligned", "misaligned"}`.

Label mapping:
  - `MISALIGNED`     → "misaligned"
  - `ALIGNED` / unset → "aligned"
  - `ERROR` rows are dropped at the data-loader level.

This matches the binary alignment axis used by agent-skills-scanning's
`alignment` scanner (see scanners/alignment/README.md).
"""

from __future__ import annotations

from stages.downstream.sft.tasks import SFTTask


SYSTEM_PROMPT = (
    "You are a careful auditor. Given the contents of a Claude Code skill "
    "package, decide whether the body matches the description (aligned) or "
    "diverges from it (misaligned). Reply with a single word — either "
    "`aligned` or `misaligned` — and nothing else."
)


PROMPT_TEMPLATE = (
    "Decide whether the following skill is `aligned` (description matches "
    "body) or `misaligned` (description and body disagree, or the body does "
    "something the description omits).\n"
    "\n"
    "=== SKILL ===\n"
    "{skill_text}\n"
    "=== END SKILL ===\n"
    "\n"
    "Answer (one word, `aligned` or `misaligned`):"
)


def _response(row):
    label = (row.get("alignment_class") or row.get("alignment_label") or "").upper().strip()
    return "misaligned" if label == "MISALIGNED" else "aligned"


TASK = SFTTask(
    name="misalignment_detection",
    prompt_template=PROMPT_TEMPLATE,
    input_column="skill_text",
    label_column="alignment_class",
    response_for=_response,
    response_vocab=["aligned", "misaligned"],
    system_prompt=SYSTEM_PROMPT,
)
