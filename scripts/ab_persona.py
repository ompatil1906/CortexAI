"""A/B the assistant persona in the system prompt against the held-out test set.

Every SFT row begins with "You are CortexAI, a customer support triage
assistant." The published card says SupportLM. That is a distribution shift the
adapter has never seen, so rather than assume it is harmless or assume it is
fatal, measure it: run the full test split under both personas and compare.

    python scripts/ab_persona.py

Exit code is 0 either way -- this reports, it does not gate. The result decides
what the model card is allowed to claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OLD = "CortexAI"
NEW = "SupportLM"

FIELDS = ("emotion", "urgency", "next_action")


def short_system(persona: str, fields: list[str]) -> str:
    """Mirrors prompts.short_system, with the persona name as the only variable."""
    from prompts import short_system as build

    return build(fields).replace(f"You are {OLD},", f"You are {persona},")


def parse_target(text: str) -> dict | None:
    from inference import parse_target as parse

    return parse(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 600 test rows")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "persona_ab.json")
    args = ap.parse_args()

    fields = json.loads((REPO_ROOT / "data" / "labeled" / "schema.json").read_text())["fields"]
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "data" / "labeled" / "test.jsonl").open()
    ]
    if args.limit:
        rows = rows[: args.limit]

    from mlx_lm import load, generate
    from mlx_lm.tuner.utils import load_adapters

    base, tok = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    base = load_adapters(base, str(REPO_ROOT / "adapters" / "final"))

    results: dict[str, dict] = {}
    for persona in (OLD, NEW):
        system = short_system(persona, fields)
        assert persona in system, f"persona substitution failed for {persona}"
        started = time.time()
        hits: Counter = Counter()
        parsed_n = 0
        for row in rows:
            prompt = tok.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"<ticket>\n{row['user_text']}\n</ticket>"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            raw = generate(base, tok, prompt=prompt, max_tokens=320, verbose=False)
            parsed = parse_target(raw)
            if parsed is None:
                continue
            parsed_n += 1
            for f in FIELDS:
                if parsed.get(f) == row["target"][f]:
                    hits[f] += 1
        n = len(rows)
        results[persona] = {
            "n": n,
            "parsed": parsed_n,
            "parse_rate": round(parsed_n / n, 4),
            **{f: round(hits[f] / n, 4) for f in FIELDS},
            "seconds": round(time.time() - started, 1),
        }
        print(f"{persona:10s} {results[persona]}")

    old, new = results[OLD], results[NEW]
    deltas = {f: round(new[f] - old[f], 4) for f in FIELDS}
    report = {
        "rows": len(rows),
        "metrics": "accuracy vs gold, unparseable counted as incorrect",
        "results": results,
        "delta_new_minus_old": deltas,
        "max_regression": min(deltas.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print("\ndelta (SupportLM minus CortexAI):", deltas)
    worst = min(deltas.values())
    if worst >= -0.01:
        print("verdict: persona swap is safe within noise; the card may use SupportLM.")
    elif worst >= -0.05:
        print("verdict: mild degradation. Use SupportLM but document the trained persona.")
    else:
        print("verdict: material degradation. Keep the trained persona in the card.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()