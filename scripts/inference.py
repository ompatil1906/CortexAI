"""Shared inference wrapper used by the Gradio app and the evaluation script.

Keeps the demo and the metrics honest: both go through :func:`CortexAI.solve`,
which uses the same prompt module and the same JSON parsing as
``scripts/evaluate.py``. A number shown in the UI is produced by the code path
that produced the reported metrics.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prompts import build_user_message, full_system, short_system  # noqa: E402
from taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "data" / "labeled" / "schema.json"

URGENCY_COLORS = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}

#: The four fictional companies in the source corpus, one per industry.
COMPANIES: dict[str, str] = {
    "ShopNova (ecommerce)": "ecommerce",
    "CloudSync Inc (saas)": "saas",
    "MediCare Plus (healthcare)": "healthcare",
    "TrustBank Financial (finance)": "finance",
}

EXAMPLES: list[tuple[str, str]] = [
    (
        "Rage about a late order",
        "This is the third time I'm contacting you about order [ORDER_ID] and it STILL hasn't arrived. "
        "I needed it for a conference that starts tomorrow morning. Your delivery estimate said Thursday "
        "and it's now Friday night. I want this delivered today or I want a full refund immediately. "
        "This is completely unacceptable.",
    ),
    (
        "Refund status chase",
        "Hi, I returned my laptop three weeks ago and you confirmed the return was received. "
        "I've been waiting for the refund ever since. Could you check where the money is? "
        "I need it before the end of the month. Order number is [ORDER_ID].",
    ),
    (
        "Cannot log in",
        "Hello, I'm trying to log into my account but I keep getting an error saying my account is locked. "
        "I never changed my password. I have tried three times and reset my password twice. "
        "Can someone please help me get back in today?",
    ),
    (
        "Polite pre-purchase question",
        "Hi! I'm considering the Galaxy S24 but I'm not sure whether the battery lasts a full day of "
        "video calls. Could you tell me what the actual screen size is in inches? "
        "No rush at all, just curious before I decide.",
    ),
    (
        "Duplicate charge",
        "I was charged twice for the same order [ORDER_ID] this morning. My bank says two pending "
        "transactions for the same amount. Please reverse one immediately, I don't have the money "
        "for double and my rent is due Friday.",
    ),
    (
        "Possible card fraud",
        "Someone has been making purchases on my card that I didn't make. There are three charges from "
        "another country in the last hour. Please freeze the account right now and tell me what I "
        "should do.",
    ),
]


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
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return coerce(parsed)
    return None


def coerce(parsed: dict) -> dict:
    """Snap labels onto the canonical enums; report the raw value if unrecognised."""
    for field, allowed in (("emotion", EMOTIONS), ("urgency", URGENCIES), ("next_action", NEXT_ACTIONS)):
        value = parsed.get(field)
        if isinstance(value, str) and value not in allowed:
            # Models occasionally emit the option's prose instead of its key.
            for key in allowed:
                if value.strip().lower().startswith(key.lower()):
                    parsed[field] = key
                    break
            else:
                for key in allowed:
                    if key.lower() in value.lower():
                        parsed[field] = key
                        break
    return parsed


class CortexAI:
    """Lazily-loaded MLX model plus its LoRA adapter."""

    def __init__(self, model: str, adapter: str | None = None, long_prompt: bool = False):
        self.model_id = model
        self.adapter = adapter
        self.long_prompt = long_prompt
        self._loaded = None
        self._tokenizer = None

    def load(self) -> None:
        if self._loaded is not None:
            return
        from mlx_lm import load

        print(f"loading {self.model_id}" + (f" + adapter {self.adapter}" if self.adapter else ""))
        self._loaded, self._tokenizer = load(self.model_id)
        if self.adapter:
            from mlx_lm.tuner.utils import load_adapters

            self._loaded = load_adapters(self._loaded, self.adapter)
        print("ready")

    @property
    def fields(self) -> list[str]:
        return json.loads(SCHEMA_PATH.read_text())["fields"]

    def solve(self, ticket: str, company: str = "ShopNova") -> dict:
        self.load()
        from mlx_lm import generate
        import mlx.core as mx

        fields = self.fields
        system = full_system(company, fields) if self.long_prompt else short_system(fields)
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_message(ticket)},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        raw = generate(self._loaded, self._tokenizer, prompt=prompt, max_tokens=320, verbose=False)
        mx.clear_cache()
        parsed = parse_target(raw)
        return {"raw": raw, "parsed": parsed, "error": None if parsed else "could not parse JSON"}


def render(result: dict) -> dict:
    """Turn a raw solve() result into display strings for the UI."""
    parsed = result.get("parsed")
    if not parsed:
        return {
            "emotion": "—",
            "urgency": "—",
            "summary": "",
            "next_action": "",
            "suggested_reply": "",
            "error": result.get("error") or "no output",
            "raw": result.get("raw", ""),
        }

    urgency = parsed.get("urgency", "—")
    return {
        "emotion": parsed.get("emotion", "—"),
        "urgency": f"{URGENCY_COLORS.get(urgency, '')} {urgency}".strip(),
        "summary": parsed.get("summary", ""),
        "next_action": parsed.get("next_action", "—"),
        "suggested_reply": parsed.get("suggested_reply", ""),
        "error": None,
        "raw": result.get("raw", ""),
    }