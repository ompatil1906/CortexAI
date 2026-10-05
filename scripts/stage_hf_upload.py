#!/usr/bin/env python3
"""Stage the Hugging Face upload tree under hf_upload/.

Builds two self-contained, self-describing directories:

  hf_upload/SupportLM_V1.0.2/            the adapter repo
  hf_upload/SupportLM-triage-dataset/    the labelled splits, as a Benchmark

Everything is generated here, including the two READMEs and eval.yaml. They used
to be written by hand and then silently deleted on the next run, because the
script rmtree'd its own output directory. If a file belongs in an upload, it has
to be emitted by this script.

The dataset is a cut-down export, not the 1.3 GB source corpus: only the
4,000/600/600 rows actually used, with the system prompt stripped. An
``eval.yaml`` registers the test split so scores can be submitted as Hub Eval
Results instead of being self-reported prose.

Run:  python scripts/stage_hf_upload.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE = REPO_ROOT / "hf_upload"
MODEL_DIR = STAGE / "SupportLM_V1.0.2"
DATASET_DIR = STAGE / "SupportLM-triage-dataset"
DATA_DIR = DATASET_DIR / "data"

MODEL_REPO = "Ompatil19/SupportLM_V1.0.2"
DATASET_REPO = "Ompatil19/SupportLM-triage-dataset"

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "results.png",
)

TARGET_FIELDS = ("emotion", "urgency", "summary", "next_action", "suggested_reply")
KEEP_META = ("uid", "split", "domain", "company", "category", "intent", "label_source")

SPLITS = ("train", "valid", "test")

EVAL_YAML = """\
name: SupportLM triage v1.0.2
description: >-
  Held-out customer support tickets with five-field triage labels (emotion,
  urgency, summary, next_action, suggested_reply), used to evaluate the
  SupportLM V1.0.2 LoRA adapter. Splits are disjoint by ticket uid.
evaluation_framework: inspect-ai

tasks:
  - id: triage_v1
    config: default
    split: test

    field_spec:
      input: user_text
      target: labels
      choices: []

    solvers:
      - name: generate

    scorers:
      - name: exact_match
        args:
          metric_name: accuracy
      - name: rouge
        args:
          metric_name: rouge
"""


def stage_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = REPO_ROOT / "peft" / "final"
    for name in ADAPTER_FILES:
        src = source / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, MODEL_DIR / name)
    total = sum(p.stat().st_size for p in MODEL_DIR.iterdir()) / 1e6
    print(f"model   : {len(ADAPTER_FILES)} files, {total:.1f} MB -> {MODEL_DIR.name}/")


def stage_dataset_rows() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = REPO_ROOT / "data" / "labeled" / f"{split}.jsonl"
        out = DATA_DIR / f"{split}.jsonl"
        written = 0
        with src.open() as fin, out.open("w") as fout:
            for line in fin:
                row = json.loads(line)
                target = row["target"]
                slim = {key: row.get(key) for key in KEEP_META}
                # Published ids carry the SupportLM namespace. The numeric suffix
                # is preserved so a row stays traceable across the rename.
                slim["uid"] = re.sub(r"^cortex-", "supportlm-", slim["uid"] or "")
                slim["user_text"] = row["user_text"]
                slim["labels"] = {field: target.get(field) for field in TARGET_FIELDS}
                fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
                written += 1
        print(f"dataset : {split:5s} {written:5d} rows, {out.stat().st_size / 1e6:5.1f} MB")

    verify_no_leakage()


def verify_no_leakage() -> None:
    """A benchmark with overlapping splits is worthless, so this is a hard gate."""
    uids = {
        split: {
            json.loads(line)["uid"] for line in (DATA_DIR / f"{split}.jsonl").open()
        }
        for split in SPLITS
    }
    bad = {
        (a, b): len(uids[a] & uids[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        if uids[a] & uids[b]
    }
    if bad:
        raise SystemExit(f"split leakage detected, refusing to publish: {bad}")
    if not all(u.startswith("supportlm-") for u in uids["test"]):
        raise SystemExit("uid rename did not apply")
    print("dataset : split leakage check passed (0 overlap by uid)")


DATASET_README = """\
---
license: cc0-1.0
task_categories:
- text-classification
- text-generation
language:
- en
tags:
- synthetic
- customer-support
- triage
- supportlm
- benchmark
pretty_name: SupportLM triage v1.0.2
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/valid.jsonl
  - split: test
    path: data/test.jsonl
---

# SupportLM triage v1.0.2

Held-out customer support tickets with five-field triage labels, published so the
[SupportLM V1.0.2](https://huggingface.co/{MODEL_REPO}) adapter's scores are
reproducible rather than self-reported.

**Synthetic data.** Every ticket was LLM-generated (`llm_generated_v6_augmented`)
for this project. It has never seen real support traffic, and the synthetic-to-real
gap is unmeasured.

## Splits

| Split | Rows | Purpose |
|---|---|---|
| `train` | 4,000 | the rows used for fine-tuning |
| `validation` | 600 | checkpoint selection |
| `test` | 600 | held out; all reported metrics come from here |

Splits are **disjoint by `uid`**, verified at export time with a hard gate that
refuses to stage if any overlap appears. Near-duplicate tickets cannot straddle
the train/test boundary by construction.

## Schema

```json
{{
  "uid": "supportlm-001268",
  "split": "test",
  "domain": "healthcare",
  "company": "MediCare Plus",
  "category": "prescription",
  "intent": "dosage_question",
  "label_source": "offline",
  "user_text": "I'm experiencing severe anxiety ...",
  "labels": {{
    "emotion": "Anxious",
    "urgency": "Critical",
    "summary": "I require this refill immediately ...",
    "next_action": "CLINICAL_ESCALATION",
    "suggested_reply": "Ms. Dubois, I understand your distress ..."
  }}
}}
```

| Field | Type | Notes |
|---|---|---|
| `uid` | string | stable ticket id; the split key |
| `split` | string | `train` / `valid` / `test` |
| `domain` | string | `ecommerce`, `healthcare`, `finance`, `saas` |
| `company` | string | fictional brand, e.g. `ShopNova` |
| `category`, `intent` | string | 22 categories, 87 intents in the source corpus |
| `label_source` | string | `offline` (rule-based) or `llm` (Gemini-written) |
| `user_text` | string | the single customer message |
| `labels.emotion` | enum | `Angry`, `Frustrated`, `Confused`, `Anxious`, `Neutral`, `Polite`, `Happy` |
| `labels.urgency` | enum | `Low`, `Medium`, `High`, `Critical` |
| `labels.summary` | string | <=25 words, one sentence |
| `labels.next_action` | enum | 1 of 42 fixed constants (`PROCESS_REFUND`, `ESCALATE_HUMAN`, ...) |
| `labels.suggested_reply` | string | reference reply, verbatim from the corpus - never generated by the model |

### Label provenance, per field

- `emotion`, `urgency`, `next_action` - rule-based from the corpus metadata, so
  they are deterministic and consistent.
- `summary` - **14 rows written by Gemini, the rest by a grounded extractive
  fallback**, because free-tier API quota was exhausted mid-run. `label_source`
  marks which rows are which. This field mostly tests the summariser, not the model.
- `suggested_reply` - the corpus's own `chosen_response`, copied verbatim.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{DATASET_REPO}")
print(ds["test"][0]["labels"])
```

```bash
curl -LO https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/data/test.jsonl
```

## Benchmark

`eval.yaml` registers the `test` split as a benchmark task (`triage_v1`) so scores
can be submitted through Hub Eval Results and appear on a leaderboard. It uses
`evaluation_framework: inspect-ai`.

Baseline for comparison, measured on this exact split:

| Metric | Base Qwen2.5-1.5B | + SupportLM V1.0.2 |
|---|---|---|
| Emotion accuracy | 55.7% | 99.0% |
| Urgency accuracy | 19.8% | 94.8% |
| Next-action accuracy | 1.2% | 56.3% |
| Reply ROUGE-L | 0.141 | 0.735 |
| Reply invents amounts | 0.2% | 58.8% |

The last row is why this dataset is public: a reproducible benchmark should make
the bad numbers as easy to find as the good ones.

## Caveats for benchmark users

- **`next_action` is imbalanced.** 23.2% of the test split is a single action
  (`EXPLAIN_PRODUCT`), and urgency is 55.8% `High`. A majority-class baseline
  scores higher on `next_action` than the model does. Report macro-F1 or
  per-class numbers alongside accuracy.
- **Reply fabrication rises with ROUGE-L.** The tuned model invents amounts in
  58.8% of replies versus 0.2% for the base model. ROUGE-L cannot distinguish a
  correct draft from a fluent invention, so score grounding separately.
- **Synthetic, single-turn, English-only, one fictional brand per domain.** It
  measures schema learning, not production fit.
- Not adversarially tested. Ticket text is untrusted input.

## License

CC0-1.0 (public domain), matching the source corpus metadata.
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    stage_model()
    stage_dataset_rows()
    (DATASET_DIR / "eval.yaml").write_text(EVAL_YAML)
    (DATASET_DIR / "README.md").write_text(DATASET_README)
    print("dataset : eval.yaml, README.md generated")
    print(f"\nstaged under {STAGE}")


if __name__ == "__main__":
    main()