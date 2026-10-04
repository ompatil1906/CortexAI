"""Prompt construction shared by training, evaluation and the demo app.

There are deliberately **two** system prompts, because the base model and the
fine-tuned model need different things:

``SHORT_SYSTEM``
    Names the JSON keys and the grounding rule. ~50 tokens. This is what the
    fine-tuned model is trained on and what the app sends at inference: after
    fine-tuning the label space is in the weights, so re-stating all 39
    next_action definitions on every request would be pure cost.

``full_system(company)``
    Additionally spells out every emotion, urgency and next_action option with
    its definition. ~700 tokens. This is the base model's only way of knowing
    the label space, so it is the baseline's *best* configuration.

Reporting the base model under ``full_system`` and the tuned model under
``SHORT_SYSTEM`` is the honest comparison: it gives the base model every
advantage a prompted model could have, and still costs the tuned model nothing.
"""

from __future__ import annotations

import json

from taxonomy import EMOTIONS, NEXT_ACTIONS, URGENCIES, URGENCY_RUBRIC

#: Documented target contract. The *actual* field list is schema-driven, read from
#: ``data/labeled/schema.json``: ``summary`` comes from an LLM batch when one exists
#: and from the local extractive summariser otherwise, so in practice the five-field
#: list below is always emitted. A prompt naming a key the model is never trained to
#: emit is worse than no prompt at all, so the rule text is filtered by ``fields``.
OUTPUT_FIELDS: tuple[str, ...] = ("emotion", "urgency", "summary", "next_action", "suggested_reply")

FIELD_HELP: dict[str, str] = {
    "emotion": f"the tone of the customer, one of {', '.join(EMOTIONS)}",
    "urgency": f"triage priority, one of {', '.join(URGENCIES)}",
    "summary": "one sentence, at most 25 words, stating what the customer wants",
    "next_action": "the single next step the agent should take, from the fixed action list",
    "suggested_reply": (
        "a short, professional and empathetic reply the agent can send as-is"
    ),
}


def _field_lines(fields: tuple[str, ...] | list[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {name}" for name in fields)


def short_system(fields: tuple[str, ...] | list[str]) -> str:
    """~60 tokens. Used for fine-tuning and for tuned-model inference."""
    return "\n".join(
        [
            "You are CortexAI, a customer support triage assistant.",
            "",
            "Read the customer's support message and reply with a single JSON object "
            "and nothing else.",
            "Keys, in this order:",
            *(f"- {name}: {FIELD_HELP[name]}" for name in fields),
            "",
            "Never invent specifics that the customer did not provide. If an order number, "
            "date, amount or address is needed but missing, ask for it in the reply and "
            "write it as [ORDER_ID], [DATE] or [AMOUNT] rather than guessing.",
        ]
    )


def _rules_for(fields: tuple[str, ...] | list[str]) -> str:
    """Field-specific rules, filtered to the fields actually being emitted.

    Emitting "summary is one sentence..." for a 4-field schema would describe an
    output the model is never trained to produce, so each rule is conditional.
    """
    rules = [
        "- emotion is the tone of the customer, not the tone you should reply in.",
        "- urgency is triage priority inferred only from what the customer actually wrote.\n"
        "  If no deadline, consequence or urgency is stated, urgency is Low.",
    ]
    if "summary" in fields:
        rules.append("- summary is one sentence, at most 25 words, stating what the customer wants.")
    if "next_action" in fields:
        rules.append("- next_action must be exactly one of the options below.")
    if "suggested_reply" in fields:
        rules.append(
            "- suggested_reply is the reply the agent sends to the customer: empathetic, specific,\n"
            "  and never inventing order numbers, dates, amounts or policy details that the ticket\n"
            "  does not contain. Ask for missing details and use [ORDER_ID], [DATE] or [AMOUNT]."
        )
    return "\n".join(rules)


def full_system(company: str, fields: tuple[str, ...] | list[str]) -> str:
    """~700 tokens. Spells out the whole label space; used only for the base baseline."""
    rubric = "\n".join(f"  {name}: {text}" for name, text in URGENCY_RUBRIC.items())
    actions = "\n".join(f"  - {key}: {text}" for key, text in NEXT_ACTIONS.items())
    return f"""You are CortexAI, a customer support triage assistant for {company}.

Read the customer's support message and reply with a single JSON object and nothing else,
with exactly these keys:

{_field_lines(fields)}

Rules:
{_rules_for(fields)}

emotion options: {", ".join(EMOTIONS)}
  - Angry: open hostility, accusations or threats
  - Frustrated: repeated annoyance that stays civil
  - Anxious: worry about consequences or pressure about timing
  - Confused: needs clarification, or the message does not follow the expected flow
  - Polite: courteous and mild in tone
  - Neutral: flat and businesslike
  - Happy: genuine satisfaction, gratitude or praise

urgency options:
{rubric}

next_action options:
{actions}"""


def build_user_message(ticket: str) -> str:
    """The model input. Only the customer's own words, nothing else."""
    return f"<ticket>\n{ticket.strip()}\n</ticket>"


def target_to_json(target: dict, fields: tuple[str, ...] | list[str]) -> str:
    """Serialise the assistant target with a fixed key order.

    ``ensure_ascii=False`` keeps names and currency literal ("Müller", "€892.99") rather
    than escaped, which shortens the target and keeps the demo output readable.
    """
    return json.dumps(
        {field: target[field] for field in fields}, ensure_ascii=False
    )


def build_messages(
    ticket: str,
    target: dict | None,
    company: str,
    fields: tuple[str, ...] | list[str],
    long_prompt: bool = False,
) -> list[dict]:
    system = full_system(company, fields) if long_prompt else short_system(fields)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(ticket)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target_to_json(target, fields)})
    return messages