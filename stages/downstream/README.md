# Downstream — SFT, Inference, Baseline Eval

The final stage. Three independent entry points, all sharing the
backbone-loading + LoRA + auto-resume machinery from `common/`.

| Entry point                              | What it does |
| ---------------------------------------- | ------------ |
| `stages.downstream.sft.train`            | Generative-Q&A SFT for a registered classification task (`malicious_detection`, `misalignment_detection`, or your own). Loss masked on the prompt; only the one-word answer is supervised. |
| `stages.downstream.inference`            | Batch-predict + classification metrics (accuracy / precision / recall / F1) on a held-out test set. Uses the same task plug-in as SFT so a single task definition powers training + eval. |
| `stages.downstream.eval_baseline`        | CLM perplexity / bits-per-byte on the pretraining test split. Useful as a sanity check before claiming the trained model is meaningfully better than the raw backbone. |

## SFT — task-parameterized

Built-in tasks (see `sft/tasks/`):

- `malicious_detection` — reads `overall_class`, answers `malicious` / `safe`.
- `misalignment_detection` — reads `alignment_class`, answers `aligned` / `misaligned`.

Adding a new task is a single Python file:

```python
# stages/downstream/sft/tasks/my_task.py
from stages.downstream.sft.tasks import SFTTask

PROMPT = "Classify the skill as `foo` or `bar`.\n\n{skill_text}\n\nAnswer:"

def _response(row):
    return "foo" if row.get("my_label") == "FOO" else "bar"

TASK = SFTTask(
    name="my_task",
    prompt_template=PROMPT,
    input_column="skill_text",
    label_column="my_label",
    response_for=_response,
    response_vocab=["foo", "bar"],
)
```

Then add `"my_task"` to `_TASKS` in `stages/downstream/sft/tasks/__init__.py`.
The SFT trainer and `inference.py` both pick it up via `data.task: my_task`
in the config.

## Examples

```bash
# Train SFT for malicious detection (uses the base backbone)
python -m stages.downstream.sft.train \
    --config stages/downstream/configs/sft_malicious_example.yaml

# Train SFT for misalignment detection, starting from a CPT+HCL backbone
AGENTSKILLS_PHASE1_ADAPTER=/path/to/full_cpt/final_model \
AGENTSKILLS_PHASE2_ADAPTER=/path/to/pl_hcl/final_model \
python -m stages.downstream.sft.train \
    --config stages/downstream/configs/sft_misalignment_example.yaml

# Evaluate the SFT adapter on the real-world test set
AGENTSKILLS_SFT_ADAPTER=/path/to/sft_malicious/final_model \
python -m stages.downstream.inference \
    --config stages/downstream/configs/inference_example.yaml

# Perplexity sanity check
python -m stages.downstream.eval_baseline \
    --config stages/downstream/configs/eval_baseline_example.yaml
```

## Outputs

### SFT
```
<run.output_dir>/
├── train.log
├── metrics.jsonl
├── checkpoint-<N>/
├── final_model/                 # LoRA adapter — load into inference
├── train_results.json
├── validation_results.json
└── test_results.json
```

### Inference
```
<run.output_dir>/
├── predictions_<task>.jsonl     # one row per skill: {gold, pred, scores}
└── metrics_<task>.json          # {accuracy, precision, recall, f1, …}
```

## Input data conventions

The SFT data is **flat** — one row per skill with at minimum:

- the input column declared by `task.input_column` (default `skill_text`)
- the label column declared by `task.label_column` (default `overall_class`
  or `alignment_class`)

Where this comes from is up to you, but the natural sources are:

1. **Synthetic labels** — verdict columns from agent-skills-scanning's
   `unified_results.csv` joined back to the skill text. Good for
   training; cheap.
2. **Human-reviewed labels** — the corpus pointed at by agent-skills-
   preparation's `--scan-results`. Best for the final evaluation pass
   because the labels are gold.
