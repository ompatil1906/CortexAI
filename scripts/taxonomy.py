"""Canonical label space for CortexAI.

Everything the model is allowed to emit is defined here exactly once: the
emotion enum, the urgency enum, the closed next-action set, the rubric the
labeller uses, and the deterministic validators derived from the source corpus.

The source corpus (``dataset/*_complete.jsonl``) ships with a ``sentiment``
field whose values are ``confused|urgent|neutral|positive|frustrated``. Note
``urgent`` is a *tone*, not an emotion, so it is remapped to ``Anxious`` here.
That remap is never used as the training label -- it exists to audit the
LLM-generated emotion labels (see ``scripts/label_llm.py --report``).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Emotion
# --------------------------------------------------------------------------

EMOTIONS: tuple[str, ...] = (
    "Angry",
    "Frustrated",
    "Confused",
    "Anxious",
    "Neutral",
    "Polite",
    "Happy",
)

#: Deterministic validator: source-corpus ``sentiment`` -> canonical emotion.
SENTIMENT_TO_EMOTION: dict[str, str] = {
    "frustrated": "Frustrated",
    "urgent": "Anxious",
    "confused": "Confused",
    "neutral": "Neutral",
    "positive": "Happy",
}

# --------------------------------------------------------------------------
# Urgency
# --------------------------------------------------------------------------

URGENCIES: tuple[str, ...] = ("Low", "Medium", "High", "Critical")

# --------------------------------------------------------------------------
# Urgency: offline rule (used until LLM labels are available)
# --------------------------------------------------------------------------
# Score components, summed then bucketed by ``urgency_from_score``.
#
# This was originally rejected in favour of an LLM-written, text-observable label
# because a crude keyword probe agreed with it only 11.7% of the time. That
# measurement was wrong: the keyword regex was bad, not the label. A TF-IDF +
# logistic-regression probe on the leak-free held-out split recovers this label at
# 99.4% accuracy / 99.2% macro-F1 against a 52.8% majority baseline, so the label is
# almost perfectly determined by linguistic cues in the ticket. It is learnable.
#
# The real caveat is different and is documented in the README: ``resolution``,
# ``difficulty`` and ``csat`` are outcomes of how a ticket was handled, not
# properties of what the customer wrote, so a triage model trained on this label is
# partly predicting the corpus generator's latent variables. That is why
# ``evaluate.py`` also reports a cross-domain generalisation split, which is the
# number worth believing.

RESOLUTION_SCORE: dict[str, int] = {
    "escalated": 3,
    "unresolved": 2,
    "follow_up_needed": 1,
    "refunded": 0,
    "resolved": -1,
}

DIFFICULTY_SCORE: dict[int, int] = {5: 3, 4: 2, 3: 1, 2: 0, 1: -1}

SENTIMENT_SCORE: dict[str, int] = {
    "urgent": 3,
    "frustrated": 2,
    "confused": 1,
    "neutral": 0,
    "positive": -1,
}

#: Categories where the underlying subject matter is high-impact regardless of tone
#: (fraud, money movement, prescriptions, medical records).
HIGH_IMPACT_CATEGORIES: frozenset[str] = frozenset(
    {
        "fraud_security",
        "payment_billing",
        "loan_mortgage",
        "prescription",
        "insurance_billing",
        "investment",
        "medical_records",
    }
)

HIGH_IMPACT_BONUS = 2

URGENCY_THRESHOLDS: tuple[tuple[int, str], ...] = ((7, "Critical"), (4, "High"), (2, "Medium"))


def urgency_from_score(score: int) -> str:
    for threshold, name in URGENCY_THRESHOLDS:
        if score >= threshold:
            return name
    return "Low"


def urgency_from_metadata(resolution: str, difficulty: int, sentiment: str, category: str) -> str:
    score = (
        RESOLUTION_SCORE.get(resolution, 0)
        + DIFFICULTY_SCORE.get(difficulty, 0)
        + SENTIMENT_SCORE.get(sentiment, 0)
        + (HIGH_IMPACT_BONUS if category in HIGH_IMPACT_CATEGORIES else 0)
    )
    return urgency_from_score(score)

#: Deliberately text-observable only. The source corpus also carries
#: ``resolution``, ``difficulty``, ``csat`` and ``sla_target_min``, but those are
#: post-hoc outcomes of the ticket rather than properties of what the customer
#: wrote. See the comment above ``RESOLUTION_SCORE`` for the measured
#: learnability result and the caveat that replaces the original rubric.
URGENCY_RUBRIC: dict[str, str] = {
    "Critical": (
        "Immediate and irreversible harm if not handled now: suspected fraud or an "
        "account takeover, a safety or medical emergency, money frozen or a payment "
        "about to be taken with no consent, or a service the customer depends on "
        "being completely down right now."
    ),
    "High": (
        "The customer states a hard deadline within 24-48 hours, money or a "
        "consequence already at stake, an urgent need for a human agent, or is "
        "explicitly repeating an issue that was already reported and not fixed."
    ),
    "Medium": (
        "The customer wants a resolution soon but names no hard deadline: a delayed "
        "delivery, a pending refund, a billing question, or a request for a "
        "supervisor without time pressure."
    ),
    "Low": (
        "No time pressure stated at all: a general question, product information, "
        "curiosity, or something they explicitly say can wait ('whenever', 'next "
        "week', 'no rush')."
    ),
}

#: When no urgency language is present at all, Low is the correct label rather
#: than a guess at business impact. ~70% of tickets land here by construction.
URGENCY_FLOOR = "Low"

# --------------------------------------------------------------------------
# Next action: closed set the model must choose from
# --------------------------------------------------------------------------

NEXT_ACTIONS: dict[str, str] = {
    "REQUEST_ORDER_DETAILS": "Ask for the order ID and any other detail needed to look up the case",
    "LOOK_UP_ORDER": "Look up the order and report its current status and tracking details",
    "CHECK_SHIPMENT": "Check the shipment with the carrier and confirm a delivery estimate",
    "RESOLVE_DELAY": "Explain the cause of the delay and offer expedite shipping or cancellation",
    "FIX_PARTIAL_DELIVERY": "Arrange to resend or split the missing items from the order",
    "UPDATE_ADDRESS": "Correct the delivery address before the order ships",
    "EXPLAIN_RETURN_POLICY": "Explain return eligibility and the return window",
    "CREATE_RETURN": "Create the return and provide the shipping instructions",
    "PROCESS_REFUND": "Process the refund and confirm how long it will take",
    "CHECK_REFUND_STATUS": "Check where the refund is and confirm the expected disbursement date",
    "ARRANGE_EXCHANGE": "Arrange an exchange or replacement for the affected item",
    "RESOLVE_DAMAGE": "Request photos of the damage, then issue a replacement or refund",
    "RESET_PASSWORD": "Send a password reset link and confirm the account email address",
    "FIX_LOGIN": "Troubleshoot the login failure and check that the account is active",
    "RESEND_OTP": "Re-send or regenerate the one-time verification code",
    "UPDATE_PROFILE": "Update the account details after verifying identity",
    "EXPLAIN_PRODUCT": "Explain the product specifications, options, and availability",
    "REVIEW_PRICE_MATCH": "Review price match eligibility and apply the adjustment if it qualifies",
    "FIX_PAYMENT": "Investigate why the payment failed and suggest a retry or another method",
    "RESEND_INVOICE": "Resend the invoice or correct the billing details on it",
    "EXPLAIN_CHARGES": "Explain the taxes, fees, and proration on the charge",
    "MANAGE_SUBSCRIPTION": "Change, upgrade, downgrade, or cancel the subscription",
    "ESCALATE_HUMAN": "Escalate the ticket to a human agent or supervisor",
    "OFFER_COMPENSATION": "Assess compensation eligibility and offer goodwill credit",
    "ESCALATE_FRAUD": "Escalate suspected fraud to the security team",
    "FREEZE_ACCOUNT": "Freeze the account and start the security verification process",
    "CLINICAL_ESCALATION": "Escalate the dosage or prescription question to a clinician or pharmacist",
    "PROCESS_REFILL": "Process the refill request and confirm the pharmacy details",
    "ARRANGE_APPOINTMENT": "Schedule, reschedule, or cancel the appointment",
    "COLLECT_DIAGNOSTICS": "Collect the error details and triage the reported fault or outage",
    "FIX_INTEGRATION": "Configure the API, third-party integration, or webhook settings",
    "MANAGE_ACCESS": "Enable the feature or adjust roles and permissions",
    "EXPLAIN_COVERAGE": "Explain policy coverage, limits, and exclusions",
    "CHECK_CLAIM_STATUS": "Check the claim status and what documentation is still needed",
    "INVESTMENT_INFO": "Explain the account, portfolio, or investment options",
    "CARD_SERVICES": "Handle card activation, PIN, credit limits, or rewards",
    "ONBOARD_GUIDANCE": "Provide setup, data migration, or training guidance",
    "CARE_GUIDANCE": "Collect the symptoms and advise the appropriate next care step",
    "RECORDS_REQUEST": "Provide or correct medical records after verifying identity",
    "REFERRAL": "Arrange a referral to a qualified specialist",
    "VERIFY_IDENTITY": "Verify the account holder's identity before making any account change",
    "POLICY_LINK": "Share the relevant policy documentation or self-serve link",
}

# --------------------------------------------------------------------------
# Deterministic validator: source-corpus intent -> next action
# --------------------------------------------------------------------------
# Used to (a) sanity-check the LLM's next_action choice and (b) provide a
# fallback label if the API budget is cut. Keyed on the 87 intents present in
# the corpus; ``domain`` breaks the two collisions where one intent name is
# reused across categories.

INTENT_TO_ACTION: dict[str, str] = {
    # product_info
    "availability": "EXPLAIN_PRODUCT",
    "comparison": "EXPLAIN_PRODUCT",
    "general_inquiry": "EXPLAIN_PRODUCT",
    "price_match": "REVIEW_PRICE_MATCH",
    "recommendation": "EXPLAIN_PRODUCT",
    "specs_query": "EXPLAIN_PRODUCT",
    "stock_status": "EXPLAIN_PRODUCT",
    # order_shipping
    "address_change": "UPDATE_ADDRESS",
    "delivery_delay": "RESOLVE_DELAY",
    "missed_delivery": "CHECK_SHIPMENT",
    "partial_delivery": "FIX_PARTIAL_DELIVERY",
    "track_order": "LOOK_UP_ORDER",
    "wrong_address": "UPDATE_ADDRESS",
    # returns_refunds
    "damaged_item": "RESOLVE_DAMAGE",
    "exchange": "ARRANGE_EXCHANGE",
    "refund_status": "CHECK_REFUND_STATUS",
    "return_policy": "EXPLAIN_RETURN_POLICY",
    "return_request": "CREATE_RETURN",
    "wrong_item": "ARRANGE_EXCHANGE",
    # account
    "account_deletion": "ESCALATE_HUMAN",
    "address_update": "UPDATE_PROFILE",
    "login_issue": "FIX_LOGIN",
    "otp_issue": "RESEND_OTP",
    "password_reset": "RESET_PASSWORD",
    "profile_update": "UPDATE_PROFILE",
    # escalation
    "compensation": "OFFER_COMPENSATION",
    "complaint": "OFFER_COMPENSATION",
    "human_agent": "ESCALATE_HUMAN",
    "repeated_issue": "ESCALATE_HUMAN",
    "supervisor": "ESCALATE_HUMAN",
    "unresolved": "ESCALATE_HUMAN",
    # payment_billing
    "installment_query": "EXPLAIN_CHARGES",
    "payment_failure": "FIX_PAYMENT",
    "promo_code": "EXPLAIN_CHARGES",
    "tax_query": "EXPLAIN_CHARGES",
    "wallet_issue": "FIX_PAYMENT",
    # loan_mortgage
    "loan_application": "ESCALATE_HUMAN",
    "mortgage_payment": "EXPLAIN_CHARGES",
    "rate_inquiry": "INVESTMENT_INFO",
    "refinancing": "ESCALATE_HUMAN",
    # fraud_security
    "account_freeze": "FREEZE_ACCOUNT",
    "card_replacement": "CARD_SERVICES",
    "report_fraud": "ESCALATE_FRAUD",
    # prescription
    "dosage_question": "CLINICAL_ESCALATION",
    "medication_interaction": "CLINICAL_ESCALATION",
    "refill_request": "PROCESS_REFILL",
    # subscription
    "cancel_subscription": "MANAGE_SUBSCRIPTION",
    "downgrade_plan": "MANAGE_SUBSCRIPTION",
    "plan_change": "MANAGE_SUBSCRIPTION",
    "upgrade_plan": "MANAGE_SUBSCRIPTION",
    # onboarding
    "data_migration": "ONBOARD_GUIDANCE",
    "initial_setup": "ONBOARD_GUIDANCE",
    "user_training": "ONBOARD_GUIDANCE",
    # symptom_inquiry
    "condition_question": "CARE_GUIDANCE",
    "describe_symptoms": "CARE_GUIDANCE",
    "referral_request": "REFERRAL",
    # medical_records
    "access_records": "RECORDS_REQUEST",
    "correct_info": "RECORDS_REQUEST",
    "request_copies": "RECORDS_REQUEST",
    # technical_support
    "bug_report": "COLLECT_DIAGNOSTICS",
    "downtime_report": "COLLECT_DIAGNOSTICS",
    "error_report": "COLLECT_DIAGNOSTICS",
    "performance_issue": "COLLECT_DIAGNOSTICS",
    # billing_plan
    "payment_method_update": "FIX_PAYMENT",
    "prorated_billing": "EXPLAIN_CHARGES",
    # integration
    "api_setup": "FIX_INTEGRATION",
    "third_party_connect": "FIX_INTEGRATION",
    "webhook_config": "FIX_INTEGRATION",
    # feature_access
    "enable_feature": "MANAGE_ACCESS",
    "manage_permissions": "MANAGE_ACCESS",
    "role_management": "MANAGE_ACCESS",
    # insurance_billing
    "billing_dispute": "ESCALATE_HUMAN",
    "claim_status": "CHECK_CLAIM_STATUS",
    "coverage_question": "EXPLAIN_COVERAGE",
    # investment
    "portfolio_inquiry": "INVESTMENT_INFO",
    "retirement_account": "INVESTMENT_INFO",
    "trading_question": "INVESTMENT_INFO",
    # card_services
    "card_activation": "CARD_SERVICES",
    "limit_increase": "CARD_SERVICES",
    "pin_change": "CARD_SERVICES",
    "rewards_inquiry": "CARD_SERVICES",
    # appointment
    "cancel_appointment": "ARRANGE_APPOINTMENT",
    "doctor_availability": "ARRANGE_APPOINTMENT",
    "reschedule_appointment": "ARRANGE_APPOINTMENT",
    "schedule_appointment": "ARRANGE_APPOINTMENT",
    # safety
    "adversarial": "VERIFY_IDENTITY",
}

#: ``invoice_request`` exists under both payment_billing and billing_plan and
#: maps differently, so it is resolved by category.
INTENT_ACTION_BY_CATEGORY: dict[tuple[str, str], str] = {
    ("payment_billing", "invoice_request"): "RESEND_INVOICE",
    ("billing_plan", "invoice_request"): "RESEND_INVOICE",
}


def action_for(category: str, intent: str) -> str | None:
    """Deterministic next-action label for a source-corpus (category, intent)."""
    if (category, intent) in INTENT_ACTION_BY_CATEGORY:
        return INTENT_ACTION_BY_CATEGORY[(category, intent)]
    return INTENT_TO_ACTION.get(intent)


# --------------------------------------------------------------------------
# Shared system prompt / output contract
# --------------------------------------------------------------------------

OUTPUT_FIELDS: tuple[str, ...] = (
    "emotion",
    "urgency",
    "summary",
    "next_action",
    "suggested_reply",
)

#: Grounding rule. The corpus masks order ids and amounts as ``[ORDER_ID]``,
#: so the reply target teaches the model to ask for identifiers instead of
#: inventing them. Prompts must state this or the model will hallucinate.
NO_INVENTION_RULE = (
    "Never invent specifics that are not in the ticket. If an order number, date, "
    "amount, address or policy detail is needed but not provided, ask the customer "
    "for it and write it as a placeholder such as [ORDER_ID]."
)

#: Fields the source corpus provides, kept out of the model's input so the
#: training task matches inference exactly.
LEAKY_FIELDS: frozenset[str] = frozenset(
    {
        "sentiment",
        "resolution",
        "csat",
        "difficulty",
        "sla_target_min",
        "first_response_sec",
        "quality_score",
        "system_prompt",
        "chosen_response",
        "rejected_response",
        "context",
        "id",
        "conversation_id",
        "timestamp",
    }
)