"""Merge the pool with labels, preferring LLM labels and falling back to offline rules.

Label provenance is tracked per field so the README and ``evaluate.py`` can state
exactly where each label came from. Current state of the project:

============  ==========================  ==================================
field         source                     note
============  ==========================  ==================================
emotion       ``sentiment`` remap        free; the corpus ships this field
urgency       metadata score             free; ``scripts/taxonomy.py``
next_action   intent -> action map      free; covers all 87 intents
suggested_reply  ``chosen_response``     free; the corpus ships this field
summary       LLM, else extractive     free fallback; ``scripts/summarize.py``
============  ==========================  ==================================

``summary`` comes from an LLM batch when ``label_llm.py`` has been run, and
otherwise from a local extractive summariser. The fallback exists because the
free Gemini tier allows only ~20 requests/minute and reports "retry in 12h" once
exhausted, which makes it unusable as a build dependency for thousands of rows.
The fallback quotes the ticket verbatim, so it is grounded by construction; it is
less fluent than an LLM summary, and LLM labels override it wherever they exist.
The model and the app both read the field list from ``data/labeled/schema.json``,
so switching between the two needs no code change beyond rerunning this script.

Usage:
    python scripts/build_labels.py
    python scripts/build_labels.py --require-llm     # fail if summaries are missing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__:
    from .summarize import summarize
    from .taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summarize import summarize
    from taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "data" / "labeled"
BATCH_DIR = REPO_ROOT / "data" / "llm_batches"

SPLITS = ("train", "valid", "test")

#: Field order in the emitted JSON target. ``summary`` is optional.
BASE_FIELDS = ("emotion", "urgency", "next_action", "suggested_reply")
SUMMARY_FIELD = "summary"


def load_llm_labels(split: str, batch_dir: Path) -> dict[str, dict]:
    path = batch_dir / f"{split}.labeled.jsonl"
    if not path.exists():
        return {}
    labels: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line from an interrupted run
            labels[row["uid"]] = row["labels"]
    return labels


def offline_labels(record: dict) -> dict:
    return {
        "emotion": record["emotion_deterministic"],
        "urgency": record["urgency_deterministic"],
        "next_action": record["action_deterministic"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--require-llm", action="store_true", help="fail instead of dropping summary")
    args = parser.parse_args()

    schema_seen: list[str] | None = None
    provenance: dict[str, Counter] = {}
    out_dir = args.pool_dir

    for split in SPLITS:
        pool_path = args.pool_dir / f"{split}_pool.jsonl"
        if not pool_path.exists():
            raise SystemExit(f"missing {pool_path} - run scripts/prepare_data.py first")

        llm_labels = load_llm_labels(split, args.batch_dir)
        coverage = len(llm_labels)

        fields = list(BASE_FIELDS)
        # ``summary`` comes from the LLM batch when there is one, and from the local
        # extractive summariser otherwise. Either way the schema is stable across
        # splits, so the model always learns the same five fields.
        fields.insert(fields.index("urgency") + 1, SUMMARY_FIELD)
        if schema_seen is None:
            schema_seen = fields
        elif fields != schema_seen:
            raise SystemExit(
                "inconsistent label schema across splits: "
                f"{schema_seen} vs {fields}. Finish labelling every split before building."
            )

        out_path = out_dir / f"{split}.jsonl"
        kept = 0
        stats = Counter()

        with pool_path.open() as source, out_path.open("w") as sink:
            for line in source:
                record = json.loads(line)
                base = offline_labels(record)
                source_kind = "offline"
                target = dict(base)

                if llm_labels:
                    override = llm_labels.get(record["uid"])
                    if override:
                        target.update({k: v for k, v in override.items() if k in fields})
                        source_kind = "llm"
                    else:
                        stats["rows_without_llm_labels"] += 1

                # Local fallback for the one field an API call would normally supply.
                # Extractive, so it cannot invent content; see scripts/summarize.py.
                if not target.get(SUMMARY_FIELD):
                    target[SUMMARY_FIELD] = summarize(record["user_text"])
                    if source_kind == "llm":
                        source_kind = "llm+offline_summary"

                target["suggested_reply"] = record["agent_reply"]

                # Guard the enums; a bad value would silently poison training.
                if target["emotion"] not in EMOTIONS:
                    stats["bad_emotion"] += 1
                    target["emotion"] = "Neutral"
                if target["urgency"] not in URGENCIES:
                    stats["bad_urgency"] += 1
                    target["urgency"] = "Medium"
                if target["next_action"] not in NEXT_ACTIONS:
                    stats["bad_next_action"] += 1
                    target["next_action"] = "ESCALATE_HUMAN"

                ordered = {field: target[field] for field in fields}
                sink.write(
                    json.dumps(
                        {
                            "uid": record["uid"],
                            "split": split,
                            "label_source": source_kind,
                            "domain": record["domain"],
                            "company": record["company"],
                            "category": record["category"],
                            "intent": record["intent"],
                            "user_text": record["user_text"],
                            "policy": record["policy"],
                            "target": ordered,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
                stats[f"source_{source_kind}"] += 1
                for name in ordered:
                    provenance.setdefault(name, Counter())[source_kind] += 1

        print(f"{split:5s} -> {out_path.relative_to(REPO_ROOT)}  rows={kept:,}  llm_coverage={coverage:,}")
        if stats:
            for key, value in sorted(stats.items()):
                if value:
                    print(f"        {key}: {value}")

    assert schema_seen is not None
    schema_path = out_dir / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "fields": schema_seen,
                "enums": {
                    "emotion": list(EMOTIONS),
                    "urgency": list(URGENCIES),
                    "next_action": list(NEXT_ACTIONS),
                },
                "has_summary": SUMMARY_FIELD in schema_seen,
            },
            indent=2,
        )
    )

    print(f"\nschema: {schema_seen}")
    print(f"wrote {schema_path.relative_to(REPO_ROOT)}")

    print("\nlabel provenance by field:")
    for field in schema_seen:
        counts = provenance.get(field, Counter())
        detail = " ".join(f"{kind}={count:,}" for kind, count in sorted(counts.items())) or "corpus field"
        print(f"    {field:16s} {detail}")

    if SUMMARY_FIELD not in schema_seen:
        message = (
            "No LLM labels found and 'summary' is absent, so the model is being trained "
            "on a 4-field target. Run scripts/label_llm.py to add it."
        )
        if args.require_llm:
            raise SystemExit(message)
        print(f"\nNOTE: {message}")


if __name__ == "__main__":
    main()