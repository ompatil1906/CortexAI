"""Generate emotion / urgency / summary / next_action labels with the OpenAI API.

``suggested_reply`` is *not* generated here -- the source corpus already ships a
high-quality agent reply (``chosen_response``), filtered on ``quality_score``.
Only the four fields that do not exist in the corpus are labelled.

Design notes:

*   **Structured outputs.** Every request uses a ``json_schema`` response format,
    so a malformed-JSON retry loop is unnecessary at25K scale.
*   **Resumable.** Results are appended to a shard file keyed by ``uid``; rerunning
    skips anything already labelled. Interrupt and resume freely.
*   **Deterministic.** ``temperature=0`` plus a fixed seed, so a rerun reproduces
    the same labels.
*   **Validated.** Enum values are checked against ``scripts/taxonomy.py`` after the
    fact; a violation falls back to the deterministic intent mapping rather than
    poisoning the training set.
*   **No new dependencies.** Uses ``urllib`` from the standard library.

Usage:
    python scripts/label_llm.py --split train --limit 200          # smoke batch
    python scripts/label_llm.py --split train --workers 32          # full run
    python scripts/label_llm.py --split test --report               # audit only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__:
    from .taxonomy import (
        EMOTIONS,
        NEXT_ACTIONS,
        URGENCIES,
        URGENCY_RUBRIC,
    )
    from .llm_client import (
        PROVIDERS,
        backoff,
        build_headers,
        build_request,
        infer_provider,
        is_retryable,
        parse_response,
        post_json,
        resolve_key,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES, URGENCY_RUBRIC
    from llm_client import (
        PROVIDERS,
        backoff,
        build_headers,
        build_request,
        infer_provider,
        is_retryable,
        parse_response,
        post_json,
        resolve_key,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "data" / "labeled"
BATCH_DIR = REPO_ROOT / "data" / "llm_batches"
API_URL = "https://api.openai.com/v1/chat/completions"

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

RUBRIC_TEXT = "\n".join(f"- **{name}**: {text}" for name, text in URGENCY_RUBRIC.items())
EMOTION_LIST = ", ".join(EMOTIONS)
URGENCY_LIST = ", ".join(URGENCIES)
ACTION_LINES = "\n".join(f"{key}: {text}" for key, text in NEXT_ACTIONS.items())

SYSTEM_PROMPT = f"""You are a senior customer-support triage annotator. You label customer support tickets for a machine-learning dataset. You are precise, literal, and you never invent information that is not written in the ticket.

Label each ticket on four axes.

## 1. emotion
The emotional tone of the customer, choosing exactly one of: {EMOTION_LIST}.
Use Angry for open hostility, shouting, accusations or threats. Frustrated for
repeated annoyance or dissatisfaction that stays civil. Anxious for worry about
consequences or pressure about timing. Confused for requests for clarification or
messages that do not follow the expected flow. Polite for courteous, mild, neutral-
in-tone requests. Neutral for flat, businesslike, emotionless messages. Happy for
genuine satisfaction, gratitude or praise.
Judge the customer's tone, not the agent's.

## 2. urgency
Triage priority, choosing exactly one of: {URGENCY_LIST}.

RUBRIC:
{RUBRIC_TEXT}

This is the most important rule: judge urgency ONLY from what the customer actually
wrote. Ignore how small or large the issue sounds in the abstract, and never
speculate about business impact that the ticket does not state. If the customer
states no deadline, no consequence, and no urgency language at all, the answer is
Low. Most tickets are Low. Choosing Low is correct and expected; do not inflate
urgency.

## 3. summary
One sentence, at most 25 words, stating what the customer wants and why it matters
to them. Do not include emotion or urgency wording, and do not add facts that are
not in the ticket.

## 4. next_action
The single next step the support agent should take, chosen from this fixed list.
Pick the one action that resolves the customer's actual request.

{ACTION_LINES}

Use the closest match when nothing fits exactly. If the requester is not the
verified account holder, or asks to confirm or change account details belonging to
someone else, choose VERIFY_IDENTITY."""

USER_TEMPLATE = """<company>{company}</company>
<industry>{domain}</industry>
<company_policy>
{policy}
</company_policy>
<ticket>
{message}
</ticket>

Label this ticket."""

def build_user_prompt(record: dict) -> str:
    return USER_TEMPLATE.format(
        company=record["company"],
        domain=record["domain"],
        policy=record["policy"],
        message=record["user_text"],
    )


RESPONSE_SCHEMA = {
    "name": "ticket_triage",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotion": {"type": "string", "enum": list(EMOTIONS)},
            "urgency": {"type": "string", "enum": list(URGENCIES)},
            "summary": {"type": "string"},
            "next_action": {"type": "string", "enum": list(NEXT_ACTIONS)},
        },
        "required": ["emotion", "urgency", "summary", "next_action"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.retries = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, usage: dict | None) -> None:
        with self.lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)

    def bump(self, field: str) -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)


def call_api(
    api_key: str,
    provider: str,
    model: str,
    record: dict,
    max_retries: int,
    stats: Stats,
) -> dict | None:
    url, payload = build_request(
        provider, model, SYSTEM_PROMPT, build_user_prompt(record), RESPONSE_SCHEMA, seed=42
    )
    headers = build_headers(provider, api_key, model)

    for attempt in range(max_retries + 1):
        status, body = post_json(url, headers, payload)
        if status == 200 and body is not None:
            parsed, usage = parse_response(provider, body)
            if parsed is not None:
                stats.record(usage)
                return parsed
            # 200 but unusable body: usually a truncated or schema-violating reply.
            stats.bump("retries")
            print(f"  [{record['uid']}] unparseable body, retry {attempt + 1}", flush=True)
            time.sleep(backoff(attempt))
            continue

        if not is_retryable(status):
            # Surface the provider's own explanation when there is one; "HTTP 400"
            # on its own cannot be acted on.
            api_detail = ""
            if isinstance(body, dict):
                message = body.get("error", {})
                if isinstance(message, dict):
                    api_detail = message.get("message", "")
                elif isinstance(message, str):
                    api_detail = message
            detail = f"HTTP {status}"
            if status == 429:
                detail = "HTTP 429 quota/rate limit exhausted - check API billing or raise the quota"
            elif status == 400:
                detail = "HTTP 400 bad request"
            elif status == 401:
                detail = "HTTP 401 unauthorized - the API key was rejected"
            elif status == 403:
                detail = "HTTP 403 forbidden - the key lacks access to this model"
            if api_detail:
                detail += f": {api_detail}"
            print(f"  [{record['uid']}] {detail} (not retryable)", flush=True)
            stats.bump("failures")
            return None

        stats.bump("retries")
        wait = backoff(attempt)
        print(f"  [{record['uid']}] HTTP {status}, retry {attempt + 1} in {wait:.1f}s", flush=True)
        time.sleep(wait)

    stats.bump("failures")
    return None


def validate(labels: dict, record: dict) -> tuple[dict, list[str]]:
    """Repair enum violations against the deterministic fallback."""
    problems: list[str] = []
    if labels.get("emotion") not in EMOTIONS:
        problems.append(f"emotion={labels.get('emotion')!r}")
        labels["emotion"] = record.get("emotion_deterministic") or "Neutral"
    if labels.get("urgency") not in URGENCIES:
        problems.append(f"urgency={labels.get('urgency')!r}")
        labels["urgency"] = "Low"
    if labels.get("next_action") not in NEXT_ACTIONS:
        problems.append(f"next_action={labels.get('next_action')!r}")
        labels["next_action"] = record.get("action_deterministic") or "ESCALATE_HUMAN"
    if not isinstance(labels.get("summary"), str) or not labels["summary"].strip():
        problems.append("summary empty")
        labels["summary"] = record["user_text"][:200]
    return labels, problems


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                continue  # tolerate a torn final line from an interrupt
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--out-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--model", default="", help="blank = provider default")
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="", help="blank = infer from model")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--report", action="store_true", help="audit existing labels, make no API calls")
    args = parser.parse_args()

    provider = args.provider
    if not provider:
        provider = infer_provider(args.model) if args.model else None
    if provider is None:
        parser.error("pass --model or --provider")

    model = args.model or PROVIDERS[provider]["default_model"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.split}.labeled.jsonl"

    if args.report:
        report(out_path)
        return

    api_key = resolve_key(provider, args.api_key_file)

    pool_path = args.pool_dir / f"{args.split}_pool.jsonl"
    records = [json.loads(line) for line in pool_path.open()]
    done = load_done(out_path)
    todo = [r for r in records if r["uid"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"split={args.split} provider={provider} model={model}")
    print(f"  pool={len(records):,}  already_done={len(done):,}  to_label={len(todo):,}")
    if not todo:
        print("  nothing to do")
        return

    stats = Stats()
    started = time.time()
    repaired: list[str] = []
    written = 0
    mode = "a" if out_path.exists() else "w"

    with out_path.open(mode) as sink:

        def handle(record: dict) -> None:
            nonlocal written
            labels = call_api(api_key, provider, model, record, args.max_retries, stats)
            if labels is None:
                return
            labels, problems = validate(labels, record)
            if problems:
                repaired.append(f"{record['uid']}:{';'.join(problems)}")
            sink.write(json.dumps({**record, "labels": labels}, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                sink.flush()
                elapsed = time.time() - started
                rate = written / elapsed
                remaining = (len(todo) - written) / rate if rate else 0
                print(
                    f"  {written:,}/{len(todo):,}  {rate:.1f} rows/s  "
                    f"eta {remaining / 60:.1f} min  "
                    f"tokens {stats.prompt_tokens / 1e6:.2f}M in / {stats.completion_tokens / 1e6:.2f}M out",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            futures = [pool_exec.submit(handle, record) for record in todo]
            for future in as_completed(futures):
                future.result()

    elapsed = time.time() - started
    print(f"\nwrote {written:,} labels to {out_path.relative_to(REPO_ROOT)} in {elapsed / 60:.1f} min")
    print(f"  api calls={stats.calls:,} retries={stats.retries:,} failures={stats.failures:,}")
    print(f"  tokens: {stats.prompt_tokens / 1e6:.2f}M prompt / {stats.completion_tokens / 1e6:.2f}M completion")
    if repaired:
        print(f"  enum violations repaired: {len(repaired)} (first 5: {repaired[:5]})")

    report(out_path)


def report(out_path: Path) -> None:
    """Audit labels: class balance plus agreement with the deterministic validators."""
    if not out_path.exists():
        raise SystemExit(f"nothing to audit: {out_path} does not exist")

    rows = [json.loads(line) for line in out_path.open() if line.strip()]
    if not rows:
        raise SystemExit(f"{out_path} is empty - no labels to audit")
    labels = [r["labels"] for r in rows]
    print(f"auditing {len(rows):,} labels from {out_path.name}\n")

    for field, enum in (("emotion", EMOTIONS), ("urgency", URGENCIES)):
        counts = Counter(l[field] for l in labels)
        print(f"{field} distribution:")
        for name in enum:
            share = counts.get(name, 0) / len(labels)
            bar = "#" * int(share * 40)
            print(f"    {name:11s} {counts.get(name, 0):>6,}  {share:>6.1%} {bar}")
        unknown = set(counts) - set(enum)
        if unknown:
            print(f"    !! off-enum values: {unknown}")
        print()

    # Agreement with the corpus-derived validators quantifies label noise: emotion
    # against the `sentiment` remap, next_action against the intent mapping.
    pairs = [
        (l["emotion"], r["emotion_deterministic"]) for l, r in zip(labels, rows) if r["emotion_deterministic"]
    ]
    if pairs:
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        print(f"emotion vs sentiment-remap agreement:     {agree:.1%}  (n={len(pairs):,})")

    pairs = [
        (l["next_action"], r["action_deterministic"])
        for l, r in zip(labels, rows)
        if r["action_deterministic"]
    ]
    if pairs:
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        print(f"next_action vs intent-map agreement:      {agree:.1%}  (n={len(pairs):,})")

    lengths = sorted(len(l["summary"].split()) for l in labels)
    print(f"summary words: p50={lengths[len(lengths) // 2]} p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]}")

    actions = Counter(l["next_action"] for l in labels)
    print(f"\ndistinct next_action labels used: {len(actions)} of {len(NEXT_ACTIONS)}")
    print("least common:", ", ".join(f"{k}({v})" for k, v in actions.most_common()[-5:]))


if __name__ == "__main__":
    main()