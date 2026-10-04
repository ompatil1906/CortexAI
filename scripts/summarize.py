"""Deterministic, offline ticket summaries.

Why this exists
---------------
``summary`` is a required output field, and an LLM-written summary needs an API key
and a quota. The Gemini free tier allows only ~20 requests per minute and reports
"retry in 12h" once exhausted, which makes it unusable as a build dependency for a
4,000-row training set. So the summary is derived locally instead.

The approach is **extractive with intent-aware sentence selection**: split the
ticket into sentences, score each one by how strongly it signals the customer's
actual problem (issue keywords, request verbs, urgency markers), and return the
best sentence, lightly trimmed. This is not free-form generation, so it cannot
hallucinate -- the output is always a verbatim span of the input.

The trade-off, stated plainly: these summaries are extractive, so they read like
"Refused to log in after password reset." rather than a fluent paraphrase. They are
accurate and grounded, which is what matters for triage, but they are not as
readable as an LLM's. ``scripts/label_llm.py`` replaces them with LLM summaries
when a key and quota are available; ``build_labels.py`` prefers those and falls
back here.
"""

from __future__ import annotations

import re

#: Greeting prefixes that carry no triage signal and should not open a summary.
GREETING = re.compile(
    r"^(hi|hello|hey|yo|greetings|good\s+(morning|afternoon|evening)|dear\s+\S+|"
    r"to\s+whom\s+it\s+may\s+concern)\b[\s,.!:-]*",
    re.IGNORECASE,
)

#: Praise with no ask. "I wanted to say thank you for the wonderful experience..."
#: is a sentiment signal, already carried by the `emotion` field, and is useless as
#: a summary. The emotion label is where this information belongs.
PRAISE = re.compile(
    r"\b(thank you|thanks|grateful|appreciat\w*|wonderful experience|"
    r"pleased|wonderful|great experience|love (it|my|the)|fantastic|excellent service)\b",
    re.IGNORECASE,
)

#: Filler that restates the reader's role instead of describing the problem.
#: "I'm a first-time buyer." / "As a busy professional, ..." carry no triage signal
#: and score badly as a summary even though they are syntactically fine.
FILLER = re.compile(
    r"\b(first-time (buyer|online buyer)|i'?m an? (elderly|senior|busy|new)\b|"
    r"as an? (elderly|senior|busy|professional|new)\b|i'?ve been (a|an) \w+ customer|"
    r"i'?m not (very )?tech(nical)?|not tech-savvy|i'?m writing (because|to))\b",
    re.IGNORECASE,
)

#: Politeness padding that trails a ticket without describing the problem.
CLOSING = re.compile(
    r"(thanks?( you)?( so much)?|thank you|cheers|best regards|kind regards|"
    r"appreciate it|looking forward to your (reply|response)|"
    r"please (let me know|respond|reply))[.!\s]*$",
    re.IGNORECASE,
)

#: Problem-bearing language. Weighted so that explicit requests outrank mood.
ISSUE_TERMS: dict[str, float] = {
    # concrete failures
    "not working": 3.0, "does not work": 3.0, "doesn't work": 3.0, "won't work": 3.0,
    "broken": 3.0, "crash": 3.0, "crashes": 3.0, "error": 2.5, "failed": 2.5,
    "fails": 2.5, "cannot": 2.5, "can't": 2.5, "unable": 2.5, "stuck": 2.5,
    "not received": 3.0, "never arrived": 3.0, "still waiting": 3.0, "no response": 3.0,
    "missing": 2.0, "delayed": 2.0, "overcharged": 3.0, "charged twice": 3.0,
    "duplicate charge": 3.0, "wrong item": 2.5, "damaged": 2.5, "defective": 2.5,
    "locked out": 3.0, "locked": 2.0, "rejected": 2.0, "declined": 2.5,
    "invalid": 2.0, "expired": 2.0, "unauthorized": 2.5, "fraud": 3.5,
    "unrecognised": 2.5, "unrecognized": 2.5, "suspicious": 2.5,
    # explicit asks
    "please": 1.0, "need": 1.5, "want": 1.2, "require": 1.5, "requesting": 1.8,
    "how do i": 2.0, "how can i": 2.0, "where can i": 2.0, "help me": 1.8,
    "refund": 2.5, "return": 2.0, "replace": 2.5, "cancel": 2.2, "upgrade": 1.8,
    "track": 1.8, "resend": 2.0, "reset": 1.8, "escalate": 2.5, "complaint": 1.8,
    # urgency / impact
    "urgent": 2.0, "immediately": 2.2, "asap": 2.2, "today": 1.5, "right away": 2.2,
    "deadline": 1.8, "emergency": 3.0, "hospital": 3.0, "medical": 3.0,
    "fined": 2.0, "legal": 2.5, "cancel my": 2.2, "lost": 1.8, "stolen": 3.0,
}

#: Placeholders the corpus uses in place of real identifiers. Kept verbatim so a
#: summary never implies a concrete order number that the ticket did not contain.
PLACEHOLDER = re.compile(r"(\[[A-Z_]+\]|CC\$[\d.]+|[$€£]\s?[\d,]+(?:\.\d{2})?)")

#: First name / surname style tokens the corpus injects mid-sentence ("iPhone 15 Ricci").
#: The corpus splices a name into the middle of otherwise normal text. Anchoring on
#: the brand keeps legitimate capitalised words ("Order ID", "Galaxy S24") intact.
#: Python's ``re`` forbids variable-width lookbehind, so the brand is captured and
#: re-emitted by the replacement instead.
NAME_NOISE = re.compile(
    r"\b(iPhone|Galaxy|MacBook|Nintendo|Samsung|Pixel|Watch)\s+[A-Z][a-z]{2,12}\b"
)

#: "Hello, I'm Lisa Anderson and I'm a first-time buyer." The self-introduction is
#: preamble; the complaint follows the "and". Splitting on the conjunction recovers
#: the informative half instead of emitting "I'm and I'm a first-time buyer."
INTRO = re.compile(r"^(?:hi|hello|hey)?,?\s*i'?m\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+and\s+", re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip())]
    return [part for part in parts if part]


def strip_intro(sentence: str) -> str:
    """Drop a leading self-introduction, keeping the clause after "and"."""
    stripped = INTRO.sub("", sentence.strip())
    return stripped if len(stripped.split()) >= 4 else sentence


def score_sentence(sentence: str, position: int, total: int) -> float:
    """Higher is more likely to state the customer's actual problem."""
    lowered = sentence.lower()
    score = sum(weight for term, weight in ISSUE_TERMS.items() if term in lowered)

    # Leading sentences carry the complaint; trailing ones are usually thanks.
    score += max(0.0, 1.6 - 0.55 * position)

    # A very long sentence is usually background, not the point.
    words = len(sentence.split())
    if words > 45:
        score -= 0.8
    if words < 4:
        score -= 1.5

    # A sentence that is pure greeting/politeness is never the summary.
    stripped = GREETING.sub("", sentence).strip()
    if not stripped or CLOSING.search(stripped):
        score -= 3.0

    # Praise with no request. Penalised hard so the summary states the ask rather
    # than the gratitude, unless nothing else in the ticket is informative. On a
    # thank-you ticket there is no ask, so the least-bad sentence still wins -- but
    # a later sentence describing the actual experience is preferred over the opener.
    if PRAISE.search(lowered) and score < 2.0:
        score -= 2.5

    # Self-description filler ("I'm a first-time buyer") is never the summary.
    if FILLER.search(lowered) and not PLACEHOLDER.search(sentence):
        score -= 2.0

    # Prefer sentences that name a concrete artefact.
    if PLACEHOLDER.search(sentence):
        score += 0.5
    if total and position == total - 1 and score > 1.0:
        score -= 0.3  # mild bias away from "thanks" endings
    return score


def tidy(sentence: str) -> str:
    """Trim a chosen span to one clean, self-contained line."""
    text = strip_intro(GREETING.sub("", sentence.strip()))
    text = NAME_NOISE.sub(r"\1", text)
    text = CLOSING.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    if not text:
        text = sentence.strip()

    # Cap the length at a clause boundary so the summary stays scannable.
    if len(text.split()) > 32:
        cut = " ".join(text.split()[:32])
        for separator in (", ", "; ", " and ", " but "):
            if separator in cut:
                cut = cut.rsplit(separator, 1)[0]
                break
        text = cut.rstrip(" ,;:-")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def summarize(user_text: str, max_sentences: int = 1) -> str:
    """Return a grounded one-sentence gist of the ticket.

    Falls back to a truncated prefix when the text is a single fragment, so the
    result is never empty for a non-empty ticket.
    """
    text = (user_text or "").strip()
    if not text:
        return ""

    sentences = split_sentences(text)
    if not sentences:
        return tidy(text[:240])

    scored = [
        (score_sentence(sentence, index, len(sentences)), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    best = sorted(scored, key=lambda item: (-item[0], item[1]))

    # One sentence is the target; a second only helps when the first scores poorly
    # (for example a pure greeting), and never when the best score is already clear.
    chosen = [best[0][2]]
    if len(best) > 1 and best[0][0] < 3.0 and best[1][0] >= 2.0:
        chosen.append(best[1][2])

    return tidy(" ".join(chosen[:max_sentences])) if max_sentences == 1 else tidy(" ".join(chosen))
