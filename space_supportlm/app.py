"""SupportLM V1.0.2 — live demo of a LoRA support-triage adapter.

Paste a customer ticket, get five structured fields back. Uncheck "apply the
fine-tune" to see what the base Qwen2.5-1.5B does with the same ticket, which is
the clearest way to see what the 4,000-row fine-tune actually bought.

The adapter runs through `disable_adapter()` rather than a second copy of the
base model, so the toggle costs no extra VRAM.

ZeroGPU notes: `spaces` must be imported before torch, the model is loaded and
moved to CUDA at module scope (ZeroGPU streams the weights in on first
@spaces.GPU entry), and only the generation function is decorated.
"""

import json

import spaces  # must precede torch / transformers
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "Ompatil19/SupportLM_V1.0.2"
MAX_NEW_TOKENS = 320

FIELDS = ("emotion", "urgency", "summary", "next_action", "suggested_reply")
FIELD_LABELS = {
    "emotion": "tone of the customer",
    "urgency": "triage priority",
    "summary": "one sentence, at most 25 words, stating what the customer wants",
    "next_action": "the single next step the agent should take, from the fixed action list",
    "suggested_reply": "a short, professional and empathetic reply the agent can send as-is",
}
EMOTIONS = ("Angry", "Frustrated", "Confused", "Anxious", "Neutral", "Polite", "Happy")
URGENCIES = ("Low", "Medium", "High", "Critical")

SHORT_SYSTEM = "\n".join(
    [
        "You are SupportLM, a customer support triage assistant.",
        "",
        "Read the customer's support message and reply with a single JSON object "
        "and nothing else.",
        "Keys, in this order:",
        *(f"- {name}: {FIELD_LABELS[name]}" for name in FIELDS),
        "",
        "Never invent specifics that the customer did not provide. If an order number, "
        "date, amount or address is needed but missing, ask for it in the reply and "
        "write it as [ORDER_ID], [DATE] or [AMOUNT] rather than guessing.",
    ]
)

# The base model was never fine-tuned on this label space, so it needs the rubric
# spelled out. Same role as scripts/prompts.py full_system().
LONG_SYSTEM = """You are SupportLM, a customer support triage assistant.

Read the customer's support message and reply with a single JSON object and nothing else,
with exactly these keys: {fields}

Rules:
- emotion is the tone of the customer, not the tone you should reply in.
- urgency is how fast a human must act, not how upset the customer is.
- summary is one sentence of at most 25 words, stating what the customer wants.
- next_action is exactly one of the fixed action names listed below.
- suggested_reply is addressed to the customer and can be sent as-is.

emotion options: {emotions}
urgency options: {urgencies}

next_action options (choose exactly one):
{actions}

Never invent specifics that the customer did not provide. If an order number, date,
amount or address is needed but missing, ask for it in suggested_reply and write it
as [ORDER_ID], [DATE] or [AMOUNT] rather than guessing."""

BASE_ACTIONS = [
    "PROCESS_REFUND", "TRACK_PACKAGE", "CANCEL_ORDER", "UPDATE_ADDRESS",
    "EXPLAIN_PRODUCT", "RESET_PASSWORD", "ESCALATE_HUMAN", "OPEN_TICKET",
    "CLINICAL_ESCALATION", "SCHEDULE_CALLBACK",
]

EXAMPLES = [
    ["I'm so disappointed! I've been trying to set up 2FA on my Samsung account for two days now, and the OTP never arrives. My order number is [ORDER_ID], and I require this fixed immediately."],
    ["I'm experiencing severe anxiety and panic attacks after running out of my Xanax prescription. My psychiatrist is out of town until next week. I require this refill immediately before I have a severe episode."],
    ["I'm confused about the fees associated with my investment advisory services. I thought I was paying a flat management fee, but my recent statement shows additional charges for 'trading activity' that nobody explained to me."],
    ["I attempted to enable single sign-on for our company yesterday, but the system keeps rejecting our SAML configuration. Our IT team spent hours on this and we are now blocked from onboarding new staff. Can someone please walk us through it?"],
    ["Thanks so much for sorting out my refund so quickly, I really appreciate the help. This is the second time you've turned a nightmare into a non-issue."],
]

# --- loaded once at module scope so ZeroGPU streams weights in on first call ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def _extract(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a completion."""
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
    return None


@spaces.GPU(duration=30)
def triage(ticket: str, tuned: bool = True) -> tuple[str, str, str, str, str, str, str]:
    """Triage one customer support ticket into five structured fields.

    Args:
        ticket: The customer's message, pasted verbatim.
        tuned: Apply the SupportLM LoRA adapter. When False the base model is
            run instead, with the full label rubric in the prompt.

    Returns:
        emotion, urgency, summary, next_action, suggested_reply, raw JSON,
        and a short note when the output could not be parsed.
    """
    ticket = (ticket or "").strip()
    if not ticket:
        return ("",) * 6 + ("Paste a ticket first.",)

    if tuned:
        system = SHORT_SYSTEM
        context = model  # adapter live
        note = ""
    else:
        system = LONG_SYSTEM.format(
            fields="\n".join(f"  - {name}" for name in FIELDS),
            emotions=", ".join(EMOTIONS),
            urgencies=", ".join(URGENCIES),
            actions="\n".join(f"  - {a}" for a in BASE_ACTIONS),
        )
        context = model.disable_adapter()
        note = "Base model (adapter off)."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"<ticket>\n{ticket}\n</ticket>"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with context:
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0][input_ids["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    parsed = _extract(completion)
    if parsed is None:
        return (
            "—", "—", "—", "—", "—", completion,
            f"{note} Could not parse JSON from this completion.".strip(),
        )

    def get(field: str) -> str:
        value = parsed.get(field)
        return str(value) if value not in (None, "") else "—"

    return (
        get("emotion"),
        get("urgency"),
        get("summary"),
        get("next_action"),
        get("suggested_reply"),
        json.dumps(parsed, indent=2, ensure_ascii=False),
        note,
    )


EXAMPLES = gr.Examples(
    EXAMPLES,
    inputs=[gr.Textbox(label="Customer ticket", lines=7, scale=4)],
    outputs=[
        gr.Textbox(label="emotion"),
        gr.Textbox(label="urgency"),
        gr.Textbox(label="summary", lines=3),
        gr.Textbox(label="next_action"),
        gr.Textbox(label="suggested_reply", lines=5),
        gr.JSON(label="raw model output"),
        gr.Markdown(),
    ],
    fn=triage,
    cache_examples=True,
    cache_mode="lazy",
    label="Tickets from the published test split",
)

with gr.Blocks(title="SupportLM V1.0.2", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SupportLM V1.0.2\n"
        "Paste a customer ticket. A LoRA adapter on "
        "**Qwen2.5-1.5B-Instruct** returns five fields as JSON: "
        "`emotion`, `urgency`, `summary`, `next_action`, `suggested_reply`.\n\n"
        "[model](https://huggingface.co/Ompatil19/SupportLM_V1.0.2) · "
        "[benchmark dataset](https://huggingface.co/datasets/Ompatil19/SupportLM-triage-dataset) · "
        "trained on synthetic tickets"
    )

    with gr.Row():
        ticket_input = gr.Textbox(
            label="Customer ticket",
            placeholder="Paste the customer's message here…",
            lines=7,
            scale=4,
        )
        with gr.Column(scale=1):
            tuned_toggle = gr.Checkbox(
                value=True,
                label="Apply the fine-tune",
                info="Off = base model, full rubric in prompt",
            )
            submit = gr.Button("Triage ticket", variant="primary")

    gr.Markdown("## Result")
    with gr.Row():
        emotion_out = gr.Textbox(label="emotion")
        urgency_out = gr.Textbox(label="urgency")
    with gr.Row():
        summary_out = gr.Textbox(label="summary", lines=2)
    with gr.Row():
        action_out = gr.Textbox(label="next_action")
    with gr.Row():
        reply_out = gr.Textbox(label="suggested_reply", lines=4)
    with gr.Accordion("Raw output", open=False):
        raw_out = gr.Code(label="model completion", language="json")
    note_out = gr.Markdown()

    submit.click(
        fn=triage,
        inputs=[ticket_input, tuned_toggle],
        outputs=[
            emotion_out, urgency_out, summary_out, action_out, reply_out,
            raw_out, note_out,
        ],
    )
    ticket_input.submit(
        fn=triage,
        inputs=[ticket_input, tuned_toggle],
        outputs=[
            emotion_out, urgency_out, summary_out, action_out, reply_out,
            raw_out, note_out,
        ],
    )

    gr.Markdown("## Try these")
    EXAMPLES

    gr.Markdown(
        "---\n"
        "**Read this before trusting a `suggested_reply`.** On held-out data the "
        "adapter drafts fluent, on-topic replies and *invents amounts in 58.8% of "
        "them* — against 0.2% for the base model. Higher ROUGE-L, more grounding "
        "errors. Check every number before sending. "
        "[Full caveats](https://huggingface.co/Ompatil19/SupportLM_V1.0.2)"
    )

demo.launch(mcp_server=True)