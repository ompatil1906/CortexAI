"""Evaluate CortexAI on the held-out test split: base model vs fine-tuned.

Reports, per system:

* accuracy **and macro-F1** for emotion, urgency and next_action
* combined emotion+urgency exact match
* per-class breakdown and confusion matrices
* reply quality proxies that need no LLM judge: ROUGE-L against the gold agent
  reply, plus two grounding checks that matter more than overlap for a support tool --
  unterminated JSON, and identifiers or amounts that appear in the reply but in
  neither the ticket nor the gold reply

Macro-F1 is reported alongside accuracy on purpose. Urgency is 53% ``High``, so
accuracy alone would mostly measure class imbalance.

Usage:
    python scripts/evaluate.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt-mode full
    python scripts/evaluate.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter adapters/final
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

if __package__:
    from .prompts import build_user_message, full_system, short_system
    from .taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompts import build_user_message, full_system, short_system
    from taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = REPO_ROOT / "data" / "labeled"
EVAL_DIR = REPO_ROOT / "eval"

PLACEHOLDER_RE = re.compile(r"\[([A-Z_]{3,})\]")
#: Digit runs with an order-ish shape. Used to detect values the model invented.
IDENTIFIER_RE = re.compile(r"\b(?:order|invoice|ticket|case|ref|return)\s*(?:id|no\.?|number|#)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{4,})", re.I)
AMOUNT_RE = re.compile(r"([$€£])\s?(\d[\d,]*(?:\.\d{2})?)")

#: Honorific + surname, e.g. "Ms. Silva". The corpus injects customer names into
#: tickets, so a reply addressing someone is normally correct -- but when the ticket
#: names nobody, any honorific in the reply is fabricated. Inventing a name is a
#: trust-breaking error in a customer-facing tool and is invisible to the
#: identifier/amount checks, so it gets its own metric.
NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Sir|Madam)\.?\s+([A-Z][a-zA-Z'-]+)")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> float:
    """Macro-F1 over classes that actually occur in the gold labels.

    Averaging over the whole declared enum would score every absent class as 0.0
    and understate the model: the offline labels only ever use 5 of the 7 emotions
    and 39 of the 42 actions, so two thirds of the denominator would be classes the
    model was never asked to predict. Restricted to gold-supported classes this is
    the standard macro average; ``per_class`` uses the same rule.
    """
    scores = []
    for label in labels:
        if not any(g == label for g in gold):
            continue
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def per_class(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> dict:
    out = {}
    for label in labels:
        support = sum(1 for g in gold if g == label)
        if not support:
            continue
        hits = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        out[label] = {
            "support": support,
            "accuracy": hits / support,
            "predicted": sum(1 for p in pred if p == label),
        }
    return out


def confusion(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in index and p in index:
            matrix[index[g]][index[p]] += 1
    return matrix


def render_confusion(matrix: list[list[int]], labels: tuple[str, ...]) -> str:
    present = [i for i, row in enumerate(matrix) if sum(row)]
    width = max(len(labels[i]) for i in present) if present else 8
    lines = ["gold \\ pred".ljust(width + 12) + " ".join(f"{labels[i][:9]:>9}" for i in present)]
    for i in present:
        row = " ".join(f"{matrix[i][j]:>9}" for j in present)
        lines.append(labels[i].ljust(width + 12) + row)
    return "\n".join(lines)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if token == other else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def rouge_l(gold: str, pred: str) -> float:
    gold_tokens = gold.lower().split()
    pred_tokens = pred.lower().split()
    if not gold_tokens or not pred_tokens:
        return 0.0
    # F1 over the longest common subsequence.
    lcs = lcs_length(gold_tokens, pred_tokens)
    if not lcs:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def parse_target(text: str) -> dict | None:
    """Recover the JSON object from a completion, tolerating stray prose."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None  # unbalanced braces => truncated generation


def load_model(model: str, adapter: str | None):
    from mlx_lm import load

    loaded, tokenizer = load(model)
    if adapter:
        from mlx_lm.tuner.utils import load_adapters

        loaded = load_adapters(loaded, adapter)
        print(f"loaded adapter: {adapter}")
    return loaded, tokenizer


def run_inference(rows: list[dict], fields: list[str], model: str, adapter: str | None, prompt_mode: str) -> list[dict]:
    from mlx_lm import generate
    import mlx.core as mx

    loaded, tokenizer = load_model(model, adapter)
    outputs = []
    started = time.time()

    for index, row in enumerate(rows):
        system = full_system(row["company"], fields) if prompt_mode == "full" else short_system(fields)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_message(row["user_text"])},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        raw = generate(loaded, tokenizer, prompt=prompt, max_tokens=320, verbose=False)
        outputs.append({"raw": raw, "parsed": parse_target(raw)})
        if (index + 1) % 50 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(rows)}  {rate:.1f} rows/s", flush=True)
        mx.clear_cache()

    return outputs


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(rows: list[dict], outputs: list[dict], fields: list[str]) -> dict:
    gold = [row["target"] for row in rows]
    parsed = [out["parsed"] for out in outputs]

    parse_failures = sum(1 for p in parsed if p is None)
    # A failed parse still counts as a wrong answer for every field, never as a skip.
    results: dict = {
        "n": len(rows),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows),
    }

    enum_fields = [
        ("emotion", EMOTIONS),
        ("urgency", URGENCIES),
        ("next_action", tuple(NEXT_ACTIONS)),
    ]
    for name, labels in enum_fields:
        if name not in fields:
            continue
        # Coerce to str: a model can emit an explicit JSON null for a key, in which
        # case .get(name, "") returns None rather than the default, and sorting or
        # comparing None against str then raises. An absent or null label is simply
        # a wrong answer, and "" records it as such.
        preds = [str((p or {}).get(name) or "") for p in parsed]
        golds = [g[name] for g in gold]
        results[name] = {
            "accuracy": sum(1 for g, p in zip(golds, preds) if g == p) / len(golds),
            "macro_f1": macro_f1(golds, preds, labels),
            "majority_baseline": max(Counter(golds).values()) / len(golds),
            "per_class": per_class(golds, preds, labels),
            "confusion": confusion(golds, preds, labels),
            "off_enum_predictions": sorted({p for p in preds if p not in labels}),
        }

    if "summary" in fields:
        # Free text, so no accuracy. Two things matter: does it overlap the gold gist,
        # and does it stay grounded in the ticket? A summary that invents an order
        # number is worse than a bland one, so fabrication is scored explicitly.
        rouge, empty, invented, lengths = [], 0, 0, []
        for row, out, target in zip(rows, outputs, gold):
            summary = (out["parsed"] or {}).get("summary")
            if not isinstance(summary, str) or not summary.strip():
                empty += 1
                continue
            rouge.append(rouge_l(target["summary"], summary))
            lengths.append(len(summary.split()))
            source = row["user_text"]
            for match in IDENTIFIER_RE.findall(summary):
                if match.upper() not in source.upper():
                    invented += 1
                    break
            else:
                for symbol, number in AMOUNT_RE.findall(summary):
                    if f"{symbol}{number}" not in source and number not in source:
                        invented += 1
                        break
        results["summary"] = {
            "rouge_l": sum(rouge) / len(rouge) if rouge else 0.0,
            "missing_or_empty": empty,
            "invented_detail_rows": invented,
            "invented_detail_rate": invented / len(rows),
            "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
            "gold_mean_words": sum(len(g["summary"].split()) for g in gold) / len(gold),
        }

    if "emotion" in fields and "urgency" in fields:
        pair_gold = [f"{g['emotion']}|{g['urgency']}" for g in gold]
        pair_pred = [
            f"{(p or {}).get('emotion') or ''}|{(p or {}).get('urgency') or ''}" for p in parsed
        ]
        results["emotion_urgency_exact_match"] = sum(
            1 for g, p in zip(pair_gold, pair_pred) if g == p
        ) / len(pair_gold)

    if "suggested_reply" in fields:
        rouge, unterminated, invented_ids, invented_amounts, length_ratio = [], 0, 0, 0, []
        invented_names = 0
        for row, out, target in zip(rows, outputs, gold):
            reply = (out["parsed"] or {}).get("suggested_reply")
            if not isinstance(reply, str):
                unterminated += 1
                continue
            rouge.append(rouge_l(target["suggested_reply"], reply))

            gold_text = f"{row['user_text']} {target['suggested_reply']}"
            for match in IDENTIFIER_RE.findall(reply):
                if match.upper() not in gold_text.upper():
                    invented_ids += 1
                    break
            for match in AMOUNT_RE.findall(reply):
                token = f"{match[0]}{match[1]}"
                if token not in gold_text and match[1] not in gold_text:
                    invented_amounts += 1
                    break
            # Only a problem when the ticket itself names nobody to address.
            if not NAME_RE.search(row["user_text"]):
                for surname in NAME_RE.findall(reply):
                    if surname not in gold_text:
                        invented_names += 1
                        break
            length_ratio.append(len(reply.split()) / max(1, len(target["suggested_reply"].split())))

        results["suggested_reply"] = {
            "rouge_l": sum(rouge) / len(rouge) if rouge else 0.0,
            "missing_or_truncated": unterminated,
            "invented_identifier_rows": invented_ids,
            "invented_amount_rows": invented_amounts,
            "invented_identifier_rate": invented_ids / len(rows),
            "invented_amount_rate": invented_amounts / len(rows),
            "invented_name_rows": invented_names,
            "invented_name_rate": invented_names / len(rows),
            "mean_length_ratio": sum(length_ratio) / len(length_ratio) if length_ratio else 0.0,
        }

    return results


def print_report(name: str, results: dict) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  rows={results['n']}  unparseable={results['parse_failures']} ({results['parse_failure_rate']:.1%})")
    for field in ("emotion", "urgency", "next_action"):
        if field not in results:
            continue
        block = results[field]
        print(
            f"  {field:12s} acc={block['accuracy']:6.1%}  macro-F1={block['macro_f1']:6.1%}  "
            f"(majority {block['majority_baseline']:.1%})"
        )
        if block["off_enum_predictions"]:
            print(f"      off-enum predictions: {block['off_enum_predictions'][:5]}")
    if "emotion_urgency_exact_match" in results:
        print(f"  {'emotion+urgency':12s} exact match={results['emotion_urgency_exact_match']:6.1%}")
    if "summary" in results:
        block = results["summary"]
        print(
            f"  {'summary':12s} ROUGE-L={block['rouge_l']:.3f}  "
            f"words={block['mean_words']:.0f} (gold {block['gold_mean_words']:.0f})  "
            f"invented_detail={block['invented_detail_rate']:.1%}  empty={block['missing_or_empty']}"
        )
    if "suggested_reply" in results:
        block = results["suggested_reply"]
        print(
            f"  {'reply':12s} ROUGE-L={block['rouge_l']:.3f}  len_ratio={block['mean_length_ratio']:.2f}  "
            f"invented_id={block['invented_identifier_rate']:.1%}  invented_amount={block['invented_amount_rate']:.1%}"
            f"  invented_name={block.get('invented_name_rate', 0.0):.1%}"
        )
    for field in ("emotion", "urgency"):
        if field in results:
            print(f"\n  --- {field} per class ---")
            for label, stats in sorted(results[field]["per_class"].items(), key=lambda kv: -kv[1]["support"]):
                print(f"      {label:12s} n={stats['support']:4d}  recall={stats['accuracy']:6.1%}  predicted={stats['predicted']:4d}")
            print("\n  " + render_confusion(results[field]["confusion"], EMOTIONS if field == "emotion" else URGENCIES).replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        help="defaults to the base model this project fine-tunes",
    )
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prompt-mode",
        choices=("short", "full"),
        default="short",
        help="'full' spells out the label space; that is the base model's best case",
    )
    parser.add_argument("--tag", default=None, help="name used in eval/<tag>.json")
    parser.add_argument("--sample-output", type=int, default=0, help="print N raw completions")
    args = parser.parse_args()

    schema = json.loads((LABELED_DIR / "schema.json").read_text())
    fields = schema["fields"]
    rows = [json.loads(line) for line in (LABELED_DIR / f"{args.split}.jsonl").open()]
    if args.limit:
        rows = rows[: args.limit]

    tag = args.tag or (f"{'tuned' if args.adapter else 'base'}_{args.prompt_mode}_{args.split}")
    print(f"evaluating: model={args.model} adapter={args.adapter} prompt={args.prompt_mode} rows={len(rows)}")

    outputs = run_inference(rows, fields, args.model, args.adapter, args.prompt_mode)
    results = score(rows, outputs, fields)
    print_report(f"{args.model}" + (f" + {args.adapter}" if args.adapter else "") + f"  [prompt={args.prompt_mode}]", results)

    if args.sample_output:
        print(f"\n  --- sample completions ---")
        for row, out in list(zip(rows, outputs))[: args.sample_output]:
            print(f"\n  TICKET: {row['user_text'][:160]}")
            print(f"  GOLD   : {json.dumps(row['target'], ensure_ascii=False)[:260]}")
            print(f"  PRED   : {out['raw'][:260]}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{tag}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {EVAL_DIR / f'{tag}.json'}")


if __name__ == "__main__":
    main()