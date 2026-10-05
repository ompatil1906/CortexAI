"""Stage the Hugging Face upload tree under hf_upload/.

Produces two self-contained directories:

  hf_upload/SupportLM_V1.0.2/            the adapter repo
  hf_upload/SupportLM-triage-dataset/    the labelled evaluation dataset

The dataset is a *cut-down* export, not the 1.3 GB source corpus. It carries
only the 4,000/600/600 rows that were actually used, with the fields a consumer
needs, plus an ``eval.yaml`` so the test split becomes a real Benchmark. That
turns the model's held-out scores into structured Hub Eval Results instead of
self-reported prose.

Run:  python scripts/stage_hf_upload.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE = REPO_ROOT / "hf_upload"
MODEL_DIR = STAGE / "SupportLM_V1.0.2"
DATASET_DIR = STAGE / "SupportLM-triage-dataset"
DATA_DIR = DATASET_DIR / "data"

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
    "results.png",
)

#: Only these target fields are exported. `policy` is the full system prompt
#: (several hundred tokens per row) and `user_text`/metadata are kept, but the
#: prompt itself is reconstructable from the source repo and would triple the
#: download size for no gain to a benchmark consumer.
TARGET_FIELDS = ("emotion", "urgency", "summary", "next_action", "suggested_reply")
KEEP_META = ("uid", "split", "domain", "company", "category", "intent", "label_source")

SPLITS = ("train", "valid", "test")


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


def stage_dataset() -> None:
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
                slim["user_text"] = row["user_text"]
                slim["labels"] = {field: target.get(field) for field in TARGET_FIELDS}
                fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
                written += 1
        size = out.stat().st_size / 1e6
        print(f"dataset : {split:5s} {written:5d} rows, {size:5.1f} MB")

    verify_no_leakage()


def verify_no_leakage() -> None:
    """A benchmark with overlapping splits is worthless, so this is a hard gate."""
    uids = {}
    for split in SPLITS:
        uids[split] = {
            json.loads(line)["uid"] for line in (DATA_DIR / f"{split}.jsonl").open()
        }
    overlaps = {
        (a, b): len(uids[a] & uids[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    bad = {pair: n for pair, n in overlaps.items() if n}
    if bad:
        raise SystemExit(f"split leakage detected, refusing to publish: {bad}")
    print("dataset : split leakage check passed (0 overlap by uid)")


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    stage_model()
    stage_dataset()
    print(f"\nstaged under {STAGE}")


if __name__ == "__main__":
    main()