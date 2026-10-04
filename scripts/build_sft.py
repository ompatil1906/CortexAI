"""Convert labelled tickets into mlx-lm chat-format SFT files.

Emits ``{"messages": [...]}`` per line, which is what
``mlx_lm.tuner.datasets.ChatDataset`` reads (``venv/.../mlx_lm/tuner/datasets.py:41``).
Training uses ``--mask-prompt`` so the loss covers only the assistant JSON.

Also reports a token-length histogram against the real tokenizer, because
``--max-seq-length`` has to be chosen from data rather than guessed.

Usage:
    python scripts/build_sft.py
    python scripts/build_sft.py --tokenizer mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__:
    from .prompts import build_messages
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompts import build_messages

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = REPO_ROOT / "data" / "labeled"
SFT_DIR = REPO_ROOT / "data" / "sft"
SPLITS = ("train", "valid", "test")


def load_schema() -> list[str]:
    path = LABELED_DIR / "schema.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run scripts/build_labels.py first")
    return json.loads(path.read_text())["fields"]


def write_split(split: str, out_dir: Path, fields: list[str]) -> tuple[int, list[dict]]:
    source = LABELED_DIR / f"{split}.jsonl"
    if not source.exists():
        raise SystemExit(f"missing {source} - run scripts/build_labels.py first")

    out_path = out_dir / f"{split}.jsonl"
    kept = 0
    kept_rows: list[dict] = []

    with source.open() as src, out_path.open("w") as sink:
        for line in src:
            row = json.loads(line)
            messages = build_messages(
                ticket=row["user_text"],
                target=row["target"],
                company=row["company"],
                fields=fields,
            )
            sink.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1
            if split == "test":
                kept_rows.append(row)

    return kept, kept_rows


def report_lengths(rows: list[dict], tokenizer_name: str, fields: list[str], max_seq_length: int = 0) -> dict:
    """Token-length stats under the real chat template."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers unavailable; skipping token-length report")
        return {}

    print(f"\nloading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    full_lengths: list[int] = []
    completion_lengths: list[int] = []

    for index, row in enumerate(rows):
        if index % 200 == 0:
            print(f"  tokenizing {index}/{len(rows)}", flush=True)
        messages = build_messages(
            row["user_text"], row["target"], row["company"], fields=fields
        )
        # transformers>=5 returns a BatchEncoding unless return_dict=False, which would
        # make len() count dict keys instead of tokens. mlx_lm's own ChatDataset passes
        # the same flag for exactly this reason.
        full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        full_lengths.append(len(full))
        completion_lengths.append(len(full) - len(prompt_only))

    def stats(values: list[int], name: str) -> dict:
        ordered = sorted(values)
        result = {
            f"{name}_p50": ordered[len(ordered) // 2],
            f"{name}_p90": ordered[int(len(ordered) * 0.9)],
            f"{name}_p99": ordered[int(len(ordered) * 0.99)],
            f"{name}_max": ordered[-1],
        }
        print(f"  {name}: " + "  ".join(f"{k.split('_', 1)[1]}={v}" for k, v in result.items()))
        return result

    if full_lengths and max(full_lengths) < 16:
        raise SystemExit(
            f"tokenizer produced implausible lengths (max={max(full_lengths)}); "
            "the chat template is probably not being applied"
        )

    print(f"\ntoken lengths over {len(rows)} test rows:")
    result = {}
    result.update(stats(full_lengths, "full"))
    result.update(stats(completion_lengths, "completion"))
    result["suggested_max_seq_length"] = int(
        sorted(full_lengths)[int(len(full_lengths) * 0.995)]
    ) + 32

    # mlx_lm truncates over-length sequences silently apart from a warning, which
    # would train the model to emit unterminated JSON. Make that impossible to miss.
    result["max_seq_length"] = max_seq_length
    result["rows_over_max_seq_length"] = sum(1 for length in full_lengths if length > max_seq_length)
    if max_seq_length:
        over = result["rows_over_max_seq_length"]
        share = over / len(full_lengths)
        status = "OK" if over == 0 else "TRUNCATION"
        print(
            f"\n  max_seq_length={max_seq_length}: {over} of {len(full_lengths)} rows exceed it "
            f"({share:.2%}) [{status}]"
        )
        if over:
            print(
                f"    !! raise --max-seq-length (>= {result['suggested_max_seq_length']}) or drop "
                "those rows; truncated targets teach incomplete JSON"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled-dir", type=Path, default=LABELED_DIR)
    parser.add_argument("--out-dir", type=Path, default=SFT_DIR)
    parser.add_argument(
        "--tokenizer",
        default="mlx-community/Phi-4-mini-instruct-4bit",
        help="model repo used for the chat template",
    )
    parser.add_argument("--skip-token-report", action="store_true")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="training max_seq_length; used to flag rows that mlx_lm would truncate",
    )
    args = parser.parse_args()

    fields = load_schema()
    print(f"target fields: {fields}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_rows: list[dict] = []
    sizes = {}
    for split in SPLITS:
        kept, test_rows = write_split(split, args.out_dir, fields)
        sizes[split] = kept
        print(f"  {split:5s} -> {args.out_dir / f'{split}.jsonl'}  rows={kept:,}")

    summary = {"fields": fields, "sizes": sizes}
    if test_rows and not args.skip_token_report:
        summary["token_stats"] = report_lengths(test_rows, args.tokenizer, fields, args.max_seq_length)

    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()