"""Rebuild a leak-free, English-only, single-turn pool from the source corpus.

Why this exists rather than reusing ``dataset/{train,validation,test}_complete.jsonl``
as shipped:

*   **The shipped splits leak.** 2,196 ``conversation_id`` values and 5,298 unique
    first-user-message hashes appear in both train and test. Any metric computed on
    them is inflated.
*   **The shipped train file is heavily augmented.** 450,724 rows collapse to
    200,654 unique first-user messages, so the same customer message (sometimes with
    a different gold reply) appears many times.

Output is a deduplicated pool split by *conversation group*, so no conversation and
no customer message can ever appear in two splits. The split assignment is hashed
from the conversation id and stratification happens only *within* a split, which
makes leakage structurally impossible rather than merely unlikely.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --train-size 5000 --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .taxonomy import (
        NEXT_ACTIONS,
        SENTIMENT_TO_EMOTION,
        action_for,
        urgency_from_metadata,
    )
else:  # invoked directly as `python scripts/prepare_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from taxonomy import (
        NEXT_ACTIONS,
        SENTIMENT_TO_EMOTION,
        action_for,
        urgency_from_metadata,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "dataset"
OUT_DIR = REPO_ROOT / "data" / "labeled"

SOURCE_FILES = ("train_complete.jsonl", "validation_complete.jsonl", "test_complete.jsonl")

COMPANY_RE = re.compile(r"customer support agent for ([^,\.]+)")
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Aggressive on purpose: the augmented corpus rewrites the same ticket many
    times, so near-duplicates that differ only in casing or punctuation must
    collapse to the same hash.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return WS_RE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def split_bucket(conversation_id: str) -> float:
    """Stable pseudo-random position in [0, 1) for a conversation id."""
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


def agent_reply_of(record: dict) -> str | None:
    reply = record.get("chosen_response")
    if isinstance(reply, str) and reply.strip():
        return reply.strip()
    turns = record.get("conversation") or []
    for turn in reversed(turns):
        if turn.get("role") == "agent" and (turn.get("text") or "").strip():
            return turn["text"].strip()
    return None


def build_pool(min_quality: int) -> tuple[list[dict], dict]:
    """Stream every source file, filter, and deduplicate by customer-message hash."""
    seen_hashes: set[str] = set()
    pool: list[dict] = []
    stats: Counter = Counter()

    for filename in SOURCE_FILES:
        path = SOURCE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"missing source file: {path}")

        with path.open() as handle:
            for line in handle:
                stats["rows_read"] += 1
                record = json.loads(line)

                if record.get("language") != "en":
                    stats["dropped_not_english"] += 1
                    continue

                turns = record.get("conversation") or []
                roles = [t.get("role") for t in turns]
                if roles != ["user", "agent"]:
                    stats["dropped_not_single_turn"] += 1
                    continue

                if (record.get("quality_score") or 0) < min_quality:
                    stats["dropped_low_quality"] += 1
                    continue

                user_text = (turns[0].get("text") or "").strip()
                if not user_text:
                    stats["dropped_empty_user_text"] += 1
                    continue

                reply = agent_reply_of(record)
                if not reply:
                    stats["dropped_no_agent_reply"] += 1
                    continue

                # Normalized placeholder + whitespace variants must not both survive,
                # otherwise the model sees the same ticket twice with different gold.
                digest = text_hash(user_text)
                if digest in seen_hashes:
                    stats["dropped_duplicate_message"] += 1
                    continue
                seen_hashes.add(digest)

                conversation_id = record.get("conversation_id") or record["id"]
                system_prompt = (record.get("system_prompt") or {}).get("text") or ""
                company_match = COMPANY_RE.search(system_prompt)
                category = record.get("category", "")
                intent = record.get("intent", "")
                sentiment = record.get("sentiment", "")

                pool.append(
                    {
                        "uid": f"cortex-{len(pool):06d}",
                        "src_id": record.get("id"),
                        "conversation_id": conversation_id,
                        "user_text": user_text,
                        "user_hash": digest,
                        "agent_reply": reply,
                        "domain": record.get("domain", ""),
                        "company": (company_match.group(1).strip() if company_match else "the company"),
                        "policy": system_prompt,
                        "category": category,
                        "intent": intent,
                        "channel": record.get("channel", ""),
                        "sentiment": sentiment,
                        "resolution": record.get("resolution", ""),
                        "difficulty": record.get("difficulty", 0),
                        "emotion_deterministic": SENTIMENT_TO_EMOTION.get(sentiment),
                        "urgency_deterministic": urgency_from_metadata(
                            record.get("resolution", ""),
                            record.get("difficulty", 0),
                            sentiment,
                            category,
                        ),
                        "action_deterministic": action_for(category, intent),
                    }
                )
                stats["kept"] += 1

    return pool, dict(stats)


def assign_splits(pool: list[dict], ratios: dict[str, float]) -> None:
    """Stamp each record with a split derived from its conversation group."""
    for record in pool:
        position = split_bucket(record["conversation_id"])
        cumulative = 0.0
        assigned = "train"
        for name, ratio in ratios.items():
            cumulative += ratio
            if position < cumulative:
                assigned = name
                break
        record["split"] = assigned


def stratify_sample(
    pool: list[dict], split: str, target: int, rng: random.Random
) -> list[dict]:
    """Proportional stratified sample by (category, sentiment) within one split.

    Stratifying *within* a split keeps the group-level split assignment intact,
    so this cannot reintroduce leakage.
    """
    members = [r for r in pool if r["split"] == split]
    if target >= len(members):
        return members

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in members:
        buckets[(record["category"], record["sentiment"])].append(record)

    for records in buckets.values():
        rng.shuffle(records)

    total = len(members)
    sampled: list[dict] = []
    # Largest-remainder allocation keeps the category mix proportional.
    remainders = []
    for key, records in buckets.items():
        exact = target * len(records) / total
        take = int(exact)
        remainders.append((exact - take, key, records, take))
        sampled.extend(records[:take])

    shortfall = target - len(sampled)
    remainders.sort(key=lambda item: item[0], reverse=True)
    for _, _, records, take in remainders:
        if shortfall <= 0:
            break
        extra = min(shortfall, len(records) - take)
        sampled.extend(records[take : take + extra])
        shortfall -= extra

    rng.shuffle(sampled)
    return sampled


def assert_no_leakage(splits: dict[str, list[dict]]) -> dict:
    """Fail loudly if any conversation id or customer-message hash crossed splits."""
    conv_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    violations: list[str] = []

    for name, records in splits.items():
        for record in records:
            cid = record["conversation_id"]
            if cid in conv_owner and conv_owner[cid] != name:
                violations.append(f"conversation {cid} in {conv_owner[cid]} and {name}")
            conv_owner[cid] = name

            digest = record["user_hash"]
            if digest in hash_owner and hash_owner[digest] != name:
                violations.append(f"message {digest} in {hash_owner[digest]} and {name}")
            hash_owner[digest] = name

    if violations:
        raise AssertionError(
            f"leakage detected ({len(violations)} violations), first 5:\n  "
            + "\n  ".join(violations[:5])
        )

    return {
        "unique_conversations": len(conv_owner),
        "unique_customer_messages": len(hash_owner),
        "violations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-size", type=int, default=25_000)
    parser.add_argument("--valid-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--min-quality", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-only", action="store_true", help="profile without writing splits")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building pool ...", flush=True)
    pool, filter_stats = build_pool(args.min_quality)
    print(f"  pool: {len(pool):,} unique single-turn English tickets", flush=True)
    for key, value in sorted(filter_stats.items()):
        print(f"    {key:28s} {value:>9,}")

    ratios = {
        "train": args.train_size / (args.train_size + args.valid_size + args.test_size),
        "valid": args.valid_size / (args.train_size + args.valid_size + args.test_size),
        "test": args.test_size / (args.train_size + args.valid_size + args.test_size),
    }
    assign_splits(pool, ratios)

    targets = {"train": args.train_size, "valid": args.valid_size, "test": args.test_size}
    splits = {name: stratify_sample(pool, name, targets[name], rng) for name in targets}

    leakage = assert_no_leakage(splits)

    print("\nsplit sizes:")
    for name, records in splits.items():
        print(f"    {name:6s} {len(records):>7,}")

    print("\nsentiment mix (deterministic emotion validator):")
    for name, records in splits.items():
        counts = Counter(r["sentiment"] for r in records)
        mix = " ".join(f"{k}={v / len(records):.0%}" for k, v in counts.most_common())
        print(f"    {name:6s} {mix}")

    print("\nurgency mix (deterministic, metadata-derived):")
    for name, records in splits.items():
        counts = Counter(r["urgency_deterministic"] for r in records)
        mix = " ".join(f"{k}={v / len(records):.0%}" for k, v in counts.most_common())
        print(f"    {name:6s} {mix}")

    print("\nnext_action mix (deterministic, intent-derived), top 6:")
    counts = Counter(r["action_deterministic"] for r in pool)
    for action, count in counts.most_common(6):
        print(f"    {action:24s} {count / len(pool):>6.1%}")
    print(f"    distinct actions used: {len(counts)} of {len(NEXT_ACTIONS)}")

    missing_action = sum(1 for r in pool if r["action_deterministic"] is None)
    print(f"\nrecords with no deterministic next-action mapping: {missing_action:,}")
    print(f"unique companies: {sorted({r['company'] for r in pool})}")
    print(f"leakage check: {leakage['violations']} violations")

    report = {
        "filter_stats": filter_stats,
        "pool_size": len(pool),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "min_quality": args.min_quality,
        "seed": args.seed,
        "leakage": leakage,
        "sentiment_mix": {
            name: dict(Counter(r["sentiment"] for r in records))
            for name, records in splits.items()
        },
        "category_mix": {
            name: dict(Counter(r["category"] for r in records))
            for name, records in splits.items()
        },
        "domains": dict(Counter(r["domain"] for r in pool)),
        "companies": sorted({r["company"] for r in pool}),
    }

    if args.report_only:
        print("\n--report-only: nothing written")
        return

    for name, records in splits.items():
        path = out_dir / f"{name}_pool.jsonl"
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(records):,} rows)")

    report_path = out_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()