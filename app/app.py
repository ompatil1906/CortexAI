"""CortexAI demo: a fine-tuned support-ticket triage assistant.

Runs the 4-bit base model plus its LoRA adapter locally through MLX, so nothing
leaves the machine. Switch between the base model and the fine-tuned model to see
the difference the fine-tune made on the same ticket.

    python app/app.py
    python app/app.py --share          # public gradio.space link
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gradio as gr  # noqa: E402

from inference import COMPANIES, EXAMPLES, SCHEMA_PATH, CortexAI, render  # noqa: E402
from taxonomy import NEXT_ACTIONS  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_ADAPTER = "adapters/final"

CACHE: dict[str, CortexAI] = {}

CARD_CSS = """
.headline { margin-bottom: 2px; }
.sub { color: #6b7280; font-size: 0.95rem; margin-top: -6px; margin-bottom: 14px; }
.badge-row { display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-weight: 600; font-size: 0.95rem; border: 1px solid transparent;
}
.badge-emotion { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
.badge-urgency { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
.badge-action  { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.output-label { font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: #6b7280; margin-bottom: 4px; }
.field { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px;
    background: #fafafa; margin-bottom: 12px; }
details { margin-top: 8px; }
footer { display: none !important; }
"""


def get_engine(key: str, model: str, adapter: str | None, long_prompt: bool) -> CortexAI:
    """One loaded model per distinct setting; loading 1.5B twice would thrash RAM."""
    if key not in CACHE:
        CACHE[key] = CortexAI(model, adapter, long_prompt)
    return CACHE[key]


def classify(ticket: str, company_label: str, use_tuned: bool, adapter: str, model: str):
    if not ticket or not ticket.strip():
        return None, "Paste a support message first.", "", "", None, ""

    company = company_label.split(" (")[0] if company_label else "ShopNova"
    adapter_path = str(REPO_ROOT / adapter) if adapter else None

    if use_tuned and adapter_path and not Path(adapter_path).exists():
        return None, (
            f"No adapter at {adapter_path}. Train one first:\n"
            f"  python scripts/train_lora.py --config configs/final.json"
        ), "", "", None, ""

    engine = get_engine(
        f"{model}|{adapter_path if use_tuned else ''}|{use_tuned}",
        model,
        adapter_path if use_tuned else None,
        long_prompt=not use_tuned,
    )
    try:
        result = engine.solve(ticket, company)
    except Exception as exc:  # a model failure must not kill the UI
        return None, f"{type(exc).__name__}: {exc}", "", "", None, ""

    view = render(result)
    if view["error"]:
        return None, view["error"], "", "", None, view["raw"]

    badges = (
        f'<div class="badge-row">'
        f'<span class="badge badge-emotion">{view["emotion"]}</span>'
        f'<span class="badge badge-urgency">Urgency: {view["urgency"]}</span>'
        f'<span class="badge badge-action">{view["next_action"].replace("_", " ").title()}</span>'
        f"</div>"
    )
    action_text = NEXT_ACTIONS.get(view["next_action"], view["next_action"])
    reply = view["suggested_reply"]

    return (
        badges,
        None,
        view["summary"],
        f"{view['next_action']}\n\n{action_text}",
        reply,
        view["raw"],
    )


def load_example(name: str) -> str:
    for label, text in EXAMPLES:
        if label == name:
            return text
    return EXAMPLES[0][1] if EXAMPLES else ""


def build_ui(adapter_default: str, model_default: str) -> gr.Blocks:
    fields = json.loads(SCHEMA_PATH.read_text())["fields"]
    with gr.Blocks(title="CortexAI", css=CARD_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            '<div class="headline"><h1 style="margin:0">CortexAI</h1></div>'
            '<div class="sub">Support ticket triage assistant &mdash; a 4-bit Qwen2.5-1.5B '
            "fine-tuned with LoRA on synthetic customer-support tickets. Runs locally on Apple "
            "Silicon via MLX.</div>"
        )

        with gr.Row():
            with gr.Column(scale=5):
                company = gr.Dropdown(
                    choices=list(COMPANIES), value=list(COMPANIES)[0], label="Company"
                )
                ticket = gr.Textbox(
                    label="Customer support message",
                    placeholder="Paste the customer's message here...",
                    lines=9,
                )
                with gr.Row():
                    solve = gr.Button("Solve ticket", variant="primary", scale=2)
                    clear = gr.Button("Clear", scale=1)
                with gr.Accordion("Try an example", open=True):
                    example = gr.Dropdown(
                        choices=[label for label, _ in EXAMPLES],
                        value=EXAMPLES[0][0],
                        label="Example tickets",
                    )
                    load_btn = gr.Button("Load example")
                use_tuned = gr.Checkbox(
                    value=True,
                    label="Use fine-tuned model (uncheck to see the base model)",
                )
                with gr.Accordion("Model paths", open=False):
                    adapter = gr.Textbox(value=adapter_default, label="Adapter path")
                    model = gr.Textbox(value=model_default, label="Base model")

            with gr.Column(scale=6):
                badges = gr.HTML()
                error = gr.Markdown()
                with gr.Column():
                    gr.HTML('<div class="output-label">Problem summary</div>')
                    summary = gr.Textbox(lines=2, interactive=False)
                    gr.HTML('<div class="output-label">Recommended next action</div>')
                    action = gr.Textbox(lines=3, interactive=False)
                    gr.HTML('<div class="output-label">Suggested agent reply</div>')
                    reply = gr.Textbox(lines=8, interactive=False)
                    copy_reply = gr.Button("Copy reply")
                raw = gr.JSON(label="Raw model output", visible=False)

        outputs = [badges, error, summary, action, reply, raw]
        solve.click(classify, inputs=[ticket, company, use_tuned, adapter, model], outputs=outputs)
        ticket.submit(classify, inputs=[ticket, company, use_tuned, adapter, model], outputs=outputs)
        clear.click(lambda: ("", None, "", "", "", ""), outputs=[ticket, error, summary, action, reply, raw])
        load_btn.click(load_example, inputs=[example], outputs=[ticket])
        copy_reply.click(lambda: gr.update(value=reply.value), outputs=reply)

        gr.HTML(
            f"<div class='sub'>Outputs: <code>{', '.join(fields)}</code>. "
            "Emotion and urgency are read from the customer's words only; the model is "
            "instructed never to invent order numbers, dates or amounts that the ticket "
            "does not contain.</div>"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.adapter, args.model)
    demo.launch(share=args.share, server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()