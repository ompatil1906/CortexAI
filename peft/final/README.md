---
license: apache-2.0
library_name: peft
tags:
  - peft
  - lora
  - text-generation
  - customer-support
  - triage
  - qwen2.5
base_model: Qwen/Qwen2.5-1.5B-Instruct
---

# CortexAI — support ticket triage (LoRA adapter)

A LoRA adapter that turns [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
into a structured customer-support triage assistant. Given one support message it
returns a single JSON object with five fields.

**Adapter only.** This repository contains the 10.6M-parameter LoRA delta
(~40 MB). You load it alongside the base model; the base weights are not included.

## Usage

```bash
pip install transformers peft torch
```

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16),
    "your-org/cortexai-support-triage",   # or a local path
).eval()

SYSTEM = (
    "You are CortexAI, a customer support triage assistant.\n\n"
    "Read the customer's support message and reply with a single JSON object "
    "and nothing else.\nKeys, in this order:\n"
    "- emotion\n- urgency\n- summary\n- next_action\n- suggested_reply\n\n"
    "Never invent specifics that the customer did not provide. If an order number, "
    "date, amount or address is needed but missing, ask for it in the reply and "
    "write it as [ORDER_ID], [DATE] or [AMOUNT] rather than guessing."
)

prompt = tok.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            "Customer: My payment was declined for order [ORDER_ID] and I need "
            "this fixed today, it is urgent."
        )},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
out = model.generate(ids, max_new_tokens=320, do_sample=False)
print(json.loads(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
```

The short prompt above is what the adapter was trained on. Substituting a
different instruction format, or dropping the field list, will degrade results.

## Output format

```json
{
  "emotion": "Angry | Frustrated | Confused | Anxious | Neutral | Polite | Happy",
  "urgency": "Low | Medium | High | Critical",
  "summary": "one sentence, at most 25 words, stating what the customer wants",
  "next_action": "one of 42 fixed ACTION_CONSTANT values",
  "suggested_reply": "short, professional, empathetic reply to send as-is"
}
```

`next_action` is drawn from a closed list of 42 constants (`FIX_PAYMENT`,
`CHECK_SHIPMENT`, `PROCESS_REFUND`, `ESCALATE_HUMAN`, …). See the source repo's
`data/labeled/schema.json` for the full enumeration.

The adapter is instructed to write unknown specifics as `[ORDER_ID]`, `[DATE]`
or `[AMOUNT]` rather than inventing them.

## Measured performance

600 held-out synthetic tickets, greedy decoding, single run. Base column is the
same 600 tickets with no adapter.

| Metric | Base | + this adapter |
|---|---|---|
| Valid JSON | 96.8% | 99.8% |
| Emotion accuracy | 55.7% | **99.0%** |
| Urgency accuracy | 19.8% | **94.8%** |
| Next-action accuracy | 1.2% | **56.3%** |
| Emotion + urgency exact | — | **94.5%** |
| Summary ROUGE-L | 0.291 | **0.864** |
| Reply ROUGE-L | 0.141 | **0.735** |

Read `next_action` as the honest number: at 56% it is the weakest field, and the
task taxonomy is genuinely imbalanced (~23% of tickets are a single action). The
base model emits a correct action only by accident.

## Limitations — read before deploying

- **`suggested_reply` fabricates specifics ~59% of the time** (amounts, order
  numbers, names). The field is useful as a *draft for a human to edit*, not as
  sendable copy. Validate or template the parts that quote figures.
- **`next_action` is 56% accurate.** Use it to prioritise a queue, not to trigger
  an automated workflow without a review step.
- **Trained on synthetic CC0 data.** It has never seen real support tickets, so
  it is unvalidated on production phrasing, and topic drift on live data will be
  visible.
- **English only**, and one fictional retail brand (ShopNova) plus healthcare and
  finance-adjacent intents.
- **No safety or privacy evaluation** was performed. Do not expose it to customers
  directly.
- **Adversarial prompt robustness is untested.** Treat any ticket text as
  untrusted input.
- The `summary` field was produced largely by a grounded extractive summariser
  rather than by the model itself; see the source repo for detail.

## Training details

- Base: `Qwen/Qwen2.5-1.5B-Instruct`, LoRA fine-tuned on Apple Silicon via `mlx_lm`.
- 4,000 train / 600 validation / 600 test, split by `conversation_id` (no leakage).
- LoRA rank 16, `lora_alpha` 512, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj` for layers 12–27.
- Selected checkpoint: step 1200 of 2000 (best validation loss).
- 14 summaries were written by Gemini; the remaining ~5,186 by a grounded
  extractive fallback, because free-tier API quota was exhausted mid-run.

**Note on `lora_alpha = 512`.** This is intentional and matches the MLX
training-time multiplier of 32 (`512 / 16 = 32`). Do not "correct" it to 16 or 32
— that silently rescales the adapter by 32x or 2x.

**Note on quantization.** Training used a 4-bit MLX base, but this adapter loads
on the unquantised base. The weight update is bit-identical, yet predictions
differ slightly from the MLX run because the base weights differ. This is
expected.

## Provenance

- Code, training pipeline, evaluation scripts, and this card: MIT.
- This adapter (a derivative of Qwen2.5-1.5B-Instruct): Apache-2.0, matching the
  base model's license.
- Training data: synthetic, generated CC0-1.0.
- Base model: Qwen2.5-1.5B-Instruct, Apache-2.0.