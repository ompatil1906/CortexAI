---
title: SupportLM V1.0.2 Demo
emoji: 🎫
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: LoRA support-ticket triage on Qwen2.5-1.5B
python_version: "3.12"
startup_duration_timeout: 30m
---

# SupportLM V1.0.2 — demo Space

Live demo of [SupportLM V1.0.2](https://huggingface.co/Ompatil19/SupportLM_V1.0.2),
a LoRA adapter fine-tuned on 4,000 synthetic customer support tickets to turn a
ticket message into five structured fields.

Paste a ticket on the left, press **Triage ticket**, and read `emotion`,
`urgency`, `summary`, `next_action` and `suggested_reply` back. Uncheck **Apply
the fine-tune** to run the base Qwen2.5-1.5B on the same input — the adapter is
switched off with PEFT's `disable_adapter()`, so the toggle costs no extra VRAM.

## What it runs on

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` in bf16 |
| Adapter | `r=16`, `alpha=512`, LoRA on q/k/v/o/gate/up/down, layers 12–27 |
| Hardware | ZeroGPU (`zero-a10g`), allocated per request |
| Output | one JSON object, 5 keys, fixed order |

The Space is stateless, so every click re-queues a GPU allocation. Generation is
capped at 320 new tokens, which covers the five fields; longer tickets get a
truncated `suggested_reply` rather than an error.

## Honest limits

- **The `suggested_reply` field invents specifics.** 58.8% of tuned replies
  contain a fabricated amount versus 0.2% for the base model. That is the price
  of ROUGE-L going 0.141 → 0.735, and it is the reason the demo shows the raw
  JSON. Never send a reply without reading it.
- **`next_action` is imbalanced** — 23.2% of the test split is
  `EXPLAIN_PRODUCT`, so accuracy alone flatters a majority-class predictor.
- **Synthetic, single-turn, English, one fictional brand per domain.** The model
  has never seen real support traffic and the synthetic-to-real gap is unmeasured.
- The adapter was trained with a different assistant name in its system prompt;
  this Space uses `SupportLM`. Predictions matched in a small spot check but the
  swap was not measured across the full test set.

## Reproduce locally

```bash
pip install -r requirements.txt peft
python app.py
```

Benchmark and full metrics: [Ompatil19/SupportLM-triage-dataset](https://huggingface.co/datasets/Ompatil19/SupportLM-triage-dataset).