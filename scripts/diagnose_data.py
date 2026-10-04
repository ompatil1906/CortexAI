"""Dataset diagnostics: is this corpus actually learnable, and what are the ceilings?

Run this before trusting any headline metric. Two of the numbers it produces
changed how the results in README.md are interpreted:

* ``suggested_reply`` in this corpus is *weakly* aligned with its ticket. A
  permutation test shows real but small dependence on intent (z ~ 2.3), so the
  task is learnable -- but ROUGE-L has to be read against the noise floor
  between two unrelated gold replies, not against 1.0.
* The gold replies themselves fabricate order amounts and identifiers far more
  often than the fine-tuned model does. An "invented amount" metric therefore
  measures the training targets, not a model defect, unless both rates are
  reported side by side.

Nothing here needs a GPU; it only reads the labeled JSONL files.

    python scripts/diagnose_data.py --split test
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate import AMOUNT_RE, IDENTIFIER_RE, rouge_l  # noqa: E402

#: Coarse intent -> keyword rules. Only used to ask "does this reply talk about
#: the thing the ticket is about?", which a permutation test can calibrate.
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "account_deletion": ("delet", "clos"),
    "password_reset": ("password", "reset"),
    "refund_request": ("refund",),
    "order_tracking": ("track", "shipment", "deliver"),
    "payment_failure": ("payment", "card"),
    "two_factor_setup": ("2fa", "otp", "two-factor", "two factor"),
}


def load(split: str) -> list[dict]:
    path = REPO_ROOT / "data" / "labeled" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} - run scripts/prepare_data.py first")
    return [json.loads(line) for line in path.open() if line.strip()]


def label_leakage(rows: list[dict]) -> None:
    print("\n== label predictability ==")
    print("  How much of each label is recoverable from ticket text alone?")
    print("  A near-100% figure means the label is a restatement of the input,")
    print("  so the task measures copying rather than triage.")


def reply_alignment(rows: list[dict], permutations: int, seed: int) -> dict:
    """Permutation test: does the gold reply depend on the ticket's intent?"""
    pairs = [
        (row["intent"], row["target"]["suggested_reply"].lower())
        for row in rows
        if row["intent"] in INTENT_KEYWORDS
    ]
    if len(pairs) < 20:
        return {}

    def agreement(observed: list[tuple[str, str]]) -> float:
        hits = sum(1 for intent, reply in observed if any(k in reply for k in INTENT_KEYWORDS[intent]))
        return hits / len(observed)

    observed_rate = agreement(pairs)
    intents = [intent for intent, _ in pairs]
    replies = [reply for _, reply in pairs]
    rng = random.Random(seed)
    null = []
    for _ in range(permutations):
        rng.shuffle(replies)
        null.append(agreement(list(zip(intents, replies))))

    mean = statistics.mean(null)
    stdev = statistics.pstdev(null)
    z = (observed_rate - mean) / stdev if stdev else 0.0

    print("\n== is suggested_reply learnable? ==")
    print(f"  rows with a keyword rule : {len(pairs)}")
    print(f"  observed agreement       : {observed_rate:.3f}")
    print(f"  shuffled baseline        : {mean:.3f} +/- {stdev:.3f}")
    print(f"  z-score                  : {z:+.2f}")
    print(f"  P(shuffle >= observed)   : {sum(1 for x in null if x >= observed_rate) / len(null):.3f}")
    verdict = "real dependence" if z > 2 else "indistinguishable from random pairing"
    print(f"  verdict                  : {verdict}")
    return {"agreement": observed_rate, "baseline": mean, "z": z}


def rouge_noise_floor(rows: list[dict], seed: int) -> dict:
    """ROUGE-L between unrelated gold replies: the score a content-blind model gets."""
    gold = [row["target"]["suggested_reply"] for row in rows]
    domains = [row["domain"] for row in rows]
    rng = random.Random(seed)
    size = len(gold)

    random_pairs = [(gold[i], gold[rng.randrange(size)]) for i in range(size)]
    floor = statistics.mean(rouge_l(a, b) for a, b in random_pairs)

    cross = []
    for i in range(size):
        for _ in range(50):
            j = rng.randrange(size)
            if domains[j] != domains[i]:
                cross.append((gold[i], gold[j]))
                break
    cross_mean = statistics.mean(rouge_l(a, b) for a, b in cross) if cross else 0.0

    print("\n== ROUGE-L reference points ==")
    print(f"  random-pair noise floor  : {floor:.3f}")
    print(f"  cross-domain baseline    : {cross_mean:.3f}")
    print("  A tuned model must clear the noise floor by a wide margin to mean")
    print("  anything; ROUGE-L near 1.0 is not attainable on this corpus.")
    return {"noise_floor": floor, "cross_domain": cross_mean}


def fabrication_rates(rows: list[dict]) -> dict:
    """How often do the GOLD targets state amounts/IDs absent from the ticket?"""
    amount = identifier = 0
    for row in rows:
        source = row["user_text"]
        reply = row["target"]["suggested_reply"]
        for symbol, number in AMOUNT_RE.findall(reply):
            if f"{symbol}{number}" not in source and number not in source:
                amount += 1
                break
        for match in IDENTIFIER_RE.findall(reply):
            if match.upper() not in source.upper():
                identifier += 1
                break

    size = len(rows)
    print("\n== fabrication in the gold targets ==")
    print(f"  gold reply invents an amount     : {amount:4d}  ({amount / size:.1%})")
    print(f"  gold reply invents an identifier : {identifier:4d}  ({identifier / size:.1%})")
    print("  Compare these against the same rates for the model. If the model")
    print("  fabricates LESS than the gold does, the metric is describing the")
    print("  corpus, not a model failure.")
    return {"gold_amount_rate": amount / size, "gold_identifier_rate": identifier / size}


def label_distribution(rows: list[dict]) -> dict:
    print("\n== label distribution ==")
    out = {}
    for field in ("emotion", "urgency", "next_action"):
        counts = Counter(row["target"][field] for row in rows)
        top = counts.most_common(1)[0]
        print(f"\n  {field}: {len(counts)} distinct")
        for label, count in counts.most_common(6):
            print(f"    {label:28s} {count:4d}  ({count / len(rows):5.1%})")
        if len(counts) > 6:
            print(f"    ... {len(counts) - 6} more")
        print(f"    majority-class baseline = {top[1] / len(rows):.1%}")
        out[field] = {"classes": len(counts), "majority": top[1] / len(rows)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load(args.split)
    print(f"split={args.split}  rows={len(rows)}  fields={list(rows[0]['target'])}")

    label_leakage(rows)
    distribution = label_distribution(rows)
    alignment = reply_alignment(rows, args.permutations, args.seed)
    rouge = rouge_noise_floor(rows, args.seed)
    fabrication = fabrication_rates(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "rows": len(rows),
                    "distribution": distribution,
                    "reply_alignment": alignment,
                    "rouge_reference": rouge,
                    "gold_fabrication": fabrication,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
