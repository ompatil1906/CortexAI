# CortexAI — Support Ticket Triage on a MacBook Air

A support-ticket assistant that runs **entirely on Apple Silicon**. Given a
customer's message it returns five fields as structured JSON:

| field | meaning |
|---|---|
| `emotion` | tone of the customer — one of 7 labels |
| `urgency` | triage priority — `Low` / `Medium` / `High` / `Critical` |
| `summary` | one grounded sentence stating what the customer wants |
| `next_action` | the single next step, from a fixed 42-action list |
| `suggested_reply` | an agent-ready reply |

Built by LoRA fine-tuning a 4-bit **Qwen2.5-1.5B-Instruct** with MLX on a
16GB MacBook Air M4. The whole pipeline — data prep, labelling, training,
evaluation, demo — runs on the laptop; no GPU server and no paid API are needed.

---

## Published

The fine-tune is released as [SupportLM V1.0.2](https://huggingface.co/Ompatil19/SupportLM_V1.0.2),
a PEFT LoRA adapter, alongside the benchmark it was evaluated on,
[SupportLM triage v1.0.2](https://huggingface.co/datasets/Ompatil19/SupportLM-triage-dataset).

| | |
|---|---|
| Adapter | 4-bit base + LoRA, `r=16`, `alpha=512`, layers 12–27 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dataset | 4,000 / 600 / 600 split, CC0-1.0, synthetic, disjoint by `uid` |

Load it anywhere PEFT runs, not just MLX:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "Ompatil19/SupportLM_V1.0.2")
```

The exact weight mapping is documented under
[Converting MLX → PEFT](#converting-mlx--peft) and checked numerically by
`scripts/verify_conversion.py`.

> The adapter was trained with `CortexAI` as the assistant name in the system
> prompt; the Hub model card uses `SupportLM`. Nothing in the output depends on
> the name, but see the caveat on that card.

---

## Quick start

```bash
python3 -m venv venv
./venv/bin/python3 -m pip install -r requirements.txt

./venv/bin/python3 scripts/train_lora.py --config configs/final.json   # ~4h
./venv/bin/python3 scripts/evaluate.py --adapter adapters/final        # metrics
./venv/bin/python3 app/app.py                                          # demo
```

The demo at <http://127.0.0.1:7860> has a **Use fine-tuned model** checkbox.
Unchecking it swaps in the base model with the long prompt, which is the clearest
way to see what the fine-tune bought.

> Use `./venv/bin/python3` explicitly. A bare `python3` on this machine resolves to
> Homebrew's interpreter, which has no `mlx_lm`, and the failure surfaces as a
> confusing `No module named mlx_lm` inside a subprocess.

---

## Results

Measured on 600 held-out test rows that share no `conversation_id` with training
(`scripts/evaluate.py --adapter adapters/final`, JSON in `eval/final_tuned.json`).

All figures below are on the **same 600 held-out rows**, produced by
`scripts/evaluate.py` and saved to `eval/final_tuned.json` and `eval/base_full.json`.

| metric | base model (long prompt) | **fine-tuned** (short prompt) | majority class | noise floor |
|---|---|---|---|---|
| valid JSON emitted | 96.8% | **99.8%** | – | – |
| `emotion` accuracy | 55.7% | **99.0%** | 27.7% | – |
| `emotion` macro-F1 | 52.1% | **99.2%** | – | – |
| `urgency` accuracy | 19.8% | **94.8%** | 55.8% | – |
| `urgency` macro-F1 | 18.0% | **91.5%** | – | – |
| `next_action` accuracy | 1.2% | **56.3%** | 23.2% | – |
| `next_action` macro-F1 | 0.6% | **45.7%** | – | – |
| `emotion`+`urgency` exact | 10.0% | **94.5%** | – | – |
| `summary` ROUGE-L | 0.291 | **0.864** | – | 0.136 |
| `summary` invented details | 1.5% | **0.2%** | – | – |
| `reply` ROUGE-L | 0.141 | **0.735** | – | 0.136 |
| `reply` invented names | not measured | **8.7%** | – | – |
| `reply` length vs gold | 0.51× | **0.98×** | – | – |

Three things worth reading off this table:

- **The base model is at chance on `next_action`** (1.2% against a 23.2% majority).
  It cannot emit one of 42 actions from a name it has never been fine-tuned on. This
  is the field that needed the fine-tune most.
- **The base model's `reply` score is the noise floor.** 0.141 versus a floor of
  0.136 means its replies overlap the gold almost not at all — it writes replies at
  half the gold length (0.51×) and they are not about the same ticket. The fine-tuned
  model clears the floor by 5×.
- **`urgency` is 19.8% for the base — worse than predicting the majority class**
  (55.8%). Getting below the majority baseline means it is systematically wrong in
  both directions, not merely uninformative.

### Training scale

The 800-row pilot and the 4,000-row final run, same model and hyperparameters:

| | pilot (800 rows) | final (4,000 rows) |
|---|---|---|
| `emotion` accuracy | 95.3% | 99.0% |
| `urgency` accuracy | 66.0% | 95.3% |
| `next_action` accuracy | 42.0% | 54.7% |
| `emotion`+`urgency` exact | 63.3% | 94.8% |

The pilot trained on a 4-field target (no `summary`), so its reply/summary figures are
not directly comparable to the final run; the four rows above are, and they show
`urgency` gaining 29 points purely from data volume.

### Checkpoint selection

Validation loss bottomed at iteration 1000 (0.172) and rose to 0.198 by 1200, yet the
1200 checkpoint still won on the downstream metrics. Validation loss is per-token over
the whole target *including reply prose*, so it does not track field accuracy exactly.
Checkpoints at 400/800/1200 are kept as `adapters/final/0000400_adapters.safetensors`,
`0000800_`, and `0001200_`; the 1200 checkpoint was copied to
`adapters/final/adapters.safetensors` and the choice is recorded in
`adapters/final/PROVENANCE.md`.

**Read these against the baselines, not in isolation.** Every number below has a
matching reference point, because several headline figures look alarming until
you compare them to what the data itself supports. See
[Honest limitations](#honest-limitations) — this is the part worth reading.

---

## Pipeline

```
dataset/*.jsonl          500,700 synthetic tickets (CC0-1.0)
        │
        ▼  prepare_data.py     English only · exact [user, agent] turns · quality ≥ 75
data/labeled/*_pool.jsonl      dedup on normalised text · group-split by conversation_id
        │                      63,244 unique tickets
        ▼  build_labels.py     emotion/urgency/action from metadata + intent map
data/labeled/{train,valid,test}.jsonl   summary from summarize.py (or LLM when available)
        │                      4,000 / 600 / 600
        ▼  build_sft.py        MLX chat format · mask_prompt · length guard
data/sft/*.jsonl
        │
        ▼  train_lora.py       mlx_lm lora · rank 16 · 2 epochs · bf16 compute
adapters/final/                   + run_manifest.json recording the exact command
        │
        ▼  evaluate.py         accuracy · macro-F1 · confusion · grounding checks
eval/*.json
        │
        ▼  app/app.py          Gradio demo, same inference path as the metrics
```

### Why these choices

**Qwen2.5-1.5B, not a 3.8B model.** Phi-4-mini was tried first and measured
13–19 s/iteration at sequence length 512 — roughly 10 hours for one epoch. The
1.5B model runs at ~7 s/iteration, which makes a real train/eval loop possible on
a laptop. Quality per unit time wins here.

**`mask_prompt=True`.** Loss is computed only on the assistant's JSON, never on the
customer's message. Otherwise the model spends capacity learning to reproduce
support tickets.

**Group splitting, not row splitting.** The shipped `train`/`test` files overlap:
2,196 shared `conversation_id`s and 5,298 shared message hashes. Splitting rows
randomly would have leaked near-duplicate tickets across the boundary and
inflated every metric. `prepare_data.py` splits on `conversation_id` and asserts
zero overlap.

**Sequence length 640.** Token lengths are measured before training: p50 = 422,
p99 = 483, max = 531. 640 truncates nothing. (At the earlier 512 cap, 9 rows were
being silently cut.)

---

## The five fields, and how each is labelled

| field | source | notes |
|---|---|---|
| `emotion` | `sentiment` remapped to the 7-label enum | only 5 labels occur in the data |
| `urgency` | metadata score from resolution, difficulty, sentiment, category | see rubric in `taxonomy.py` |
| `summary` | `summarize.py`, or an LLM batch when one exists | grounded by construction |
| `next_action` | intent + category → action map | covers all 87 intents, 42 actions |
| `suggested_reply` | the corpus's own `chosen_response` | verbatim, never generated |

### `summary` without an API key

`summary` is free text, which normally means an LLM. The Gemini free tier allows
about 20 requests/minute and then returns `429 … Please retry in 12h38m`, so it
cannot be a build dependency for 5,200 rows.

`scripts/summarize.py` therefore derives the summary locally by **intent-aware
extractive selection**: split into sentences, score each by problem-bearing
language (failure verbs, explicit asks, urgency markers), penalise greetings,
praise and self-description, and return the best sentence. Because it quotes the
input verbatim, it cannot hallucinate — verified at **0% invented identifiers or
amounts**.

It is less fluent than an LLM summary ("I tried using my promo code GALAXY24 for
15% off my Galaxy S24 order [ORDER_ID], but it says it's invalid.") but it is
accurate and grounded. `scripts/label_llm.py` overrides it wherever LLM labels
exist, so the schema and the code never change.

```bash
# Optional: upgrade summaries to LLM-written ones (needs quota)
./venv/bin/python3 scripts/label_llm.py --split train --provider gemini
./venv/bin/python3 scripts/build_labels.py && ./venv/bin/python3 scripts/build_sft.py
```

---

## Honest limitations

These are the findings that changed how the numbers above should be read. They are
reproducible via `scripts/diagnose_data.py`.

**1. The labels are close to derivable from the input.** A TF-IDF probe recovers
`emotion` at ~100% and `urgency` at ~99.4% accuracy *without any language model*. The
fine-tuned model scores 99.0% and 95.3%. So the model's 99.0% on `emotion` is not
evidence of understanding — it is at the ceiling of a bag-of-words lookup, and a
logistic regression would nearly match it. The genuinely learned fields are
`summary` and `next_action`.

**2. The gold replies are weakly aligned with their tickets.** A permutation test
gives z = +2.30 — a real but small dependence on intent. Some pairs are simply
mismatched in the source corpus: an `account_deletion` ticket whose reply talks
about OTP setup, a `password_reset` ticket whose reply discusses profile-picture
uploads.

**3. ROUGE-L has a noise floor of 0.136.** Scoring each gold reply against a random
*other* gold reply gives 0.136; across domains, 0.123. The base model's 0.141 is *at*
that floor, i.e. indistinguishable from a content-blind generator. The fine-tuned
0.729 is a real signal; the summary's 0.865 is a strong one.

**4. The gold targets fabricate specifics more than the model does.** Gold replies
invent an amount in **63.7%** of rows and an identifier in **23.5%**. The fine-tuned
model invented amounts in 57.5% and identifiers in 5.7% — i.e. *less often than its own
training data*. An "invented amount" metric here mostly describes the corpus.
Reporting the 57.5% as a model defect would be wrong; the honest reading is that
~57% fabricated prices are **inherited from the labels** and the remedy is to clean
the targets, not to change the model. By contrast `summary` invents **0.0%**,
because it is extractive.

**5. `next_action` is the real weak point, and it is a 37-class problem.** 54.7%
accuracy / 44.3% macro-F1 against a 23.2% majority. The confusion is concentrated in
semantically adjacent actions (`UPDATE_PROFILE` vs `ESCALATE_HUMAN`,
`EXPLAIN_PRODUCT` vs `EXPLAIN_CHARGES`). Part of this is genuine label ambiguity: the
intent map is many-to-one, so two intents that share an action are indistinguishable by
construction.

**6. It invents customer names in 8.7% of replies.** The corpus splices real customer
names into tickets, so the model learned to open replies with "Hi Ms. Silva". When a
ticket names nobody — which is most of them — any name it produces is fabricated. This
is the most customer-visible failure in the system and none of the other grounding
checks catch it: the identifier and amount rates are fine while the model is still
addressing someone who does not exist. The demo shows this immediately
("Ms. Rodriguez" on a card-fraud ticket that named no one). A cheap mitigation would be
a post-generation check against `NAME_RE` in `evaluate.py`, but it needs fixing in the
model, not the metric.

**7. `Low` urgency is the one per-class weak spot.** 73.7% recall versus 94–99% for
every other urgency class, on only 38 test rows. `Low` is the default when nothing
indicates urgency, so the model tends to over-predict `Medium`/`High` on quiet
questions. Small support, so treat it as a signal rather than a measurement.

**8. Synthetic, single-domain, single-turn.** The corpus is LLM-generated
(`llm_generated_v6_augmented`), CC0-1.0, one message per ticket with no thread
history. Real tickets are messier: multi-turn, code-quoted, attachments, mixed
languages. Treat these numbers as an upper bound.

**9. `Polite` and `Angry` are unexercised.** The offline remap never produces them,
so the model has no training signal for 2 of the 7 emotion labels. Macro-F1 is
computed only over classes present in the gold labels — averaging over the full
enum would score those two absent classes as 0.0 and understate the model.

---

## Using the adapter outside MLX

`peft/final/` is a **standard PEFT adapter**. It runs on any platform with a
torch/transformers stack — Linux, Windows, or a Mac without MLX — and needs no
Apple-silicon-specific dependency:

```bash
pip install torch transformers peft
```

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"          # the unquantised base
ADAPTER = "peft/final"                       # this repo, or your HF repo id

tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16),
    ADAPTER,
).eval()

fields = json.load(open("data/labeled/schema.json"))["fields"]
prompt = tok.apply_chat_template(
    [
        {"role": "system", "content": (
            "You are CortexAI, a customer support triage assistant.\n\n"
            "Read the customer's support message and reply with a single JSON "
            "object and nothing else.\nKeys, in this order:\n"
            + "\n".join(f"- {name}" for name in fields)
        )},
        {"role": "user", "content": (
            "Customer: Refund for order [ORDER_ID] still has not arrived.")},
    ],
    tokenize=False, add_generation_prompt=True,
)

ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
out = model.generate(ids, max_new_tokens=320, do_sample=False)
print(json.loads(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
```

Use the same five-field prompt as `scripts/prompts.py:short_system` for the tuned
results reported above; the base model does much better with the longer prompt
built by `full_system`.

### Converting MLX → PEFT

The adapter was trained with `mlx_lm`, which stores the two low-rank matrices in
its own layout. `scripts/convert_to_peft.py` rewrites them into PEFT form:

```bash
./venv/bin/python3 scripts/convert_to_peft.py --adapter adapters/final --out peft/final
./venv/bin/python3 scripts/verify_conversion.py        # proof it is faithful
```

Two details in that conversion are easy to get wrong, and **neither raises an
error** — the adapter loads, runs, and emits perfectly well-formed JSON:

- **Orientation.** For a `Linear(in, out)`, MLX stores `lora_a` as `[in, r]` and
  `lora_b` as `[r, out]`; PEFT wants `lora_A` as `[r, in]` and `lora_B` as
  `[out, r]`. So each matrix is **transposed**, with no change of role. Qwen's
  `q_proj`/`o_proj` are square (1536×1536) and hide this mistake; only
  `k_proj`/`v_proj` (1536→256, GQA) expose it.
- **Scaling.** MLX applies `scale` directly; PEFT computes its multiplier as
  `alpha / r`. Here `scale = 32`, so `lora_alpha` is `32 × 16 = 512` — an unusual
  number that is nevertheless correct, since `512 / 16 = 32`.

`scripts/verify_conversion.py` checks all three of: tensor shapes against the real
model config, the effective weight update recomputed independently from MLX's own
formula, and a forward pass against a hand-merged fp32 copy.

Current status: shapes 0/224 wrong, effective updates identical (0.0e+00 relative
error), forward argmax and logits matching to 1.7e-06.

### One caveat: 4-bit vs fp32

Training ran against `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, but PEFT loads the
**unquantised** `Qwen/Qwen2.5-1.5B-Instruct`. The adapter transfers exactly — the
weight update is bit-identical — but the *base* weights differ, so predictions are
not identical to the MLX run. Even with no adapter at all, the 4-bit and fp32 bases
disagree on the argmax for the same token ids. The verified agreement above is
therefore between two fp32 models, deliberately.

In practice the labels agree: emotion and urgency matched on every spot-check, and
`next_action` matched on 2 of 3 (the one difference fell in the field that is only
~56% accurate even on the MLX side). If you need exact MLX numbers, use the MLX
path in this repo; if you need portability, use PEFT.

---

## Layout

```
app/app.py                  Gradio demo
scripts/
  taxonomy.py               enums, urgency rubric, intent → action map
  prepare_data.py           filtering, dedup, leak-safe splitting
  build_labels.py           merges offline + optional LLM labels
  summarize.py              offline extractive summaries
  label_llm.py              optional LLM labelling (Gemini / OpenAI)
  llm_client.py             provider abstraction over urllib
  build_sft.py              MLX chat format + token-length guard
  train_lora.py             mlx_lm wrapper, memory ladder, manifest
  evaluate.py               metrics, confusion matrices, grounding checks
  diagnose_data.py          corpus sanity checks and reference baselines
  prompts.py                short (tuned) and full (base) prompts
  inference.py              shared inference path for app + eval
configs/pilot.json          800 rows, 1 epoch  (~35 min)
configs/final.json          4,000 rows, 2 epochs (~4 h)
```

`train_lora.py` writes `adapters/<name>/run_manifest.json` with the fully resolved
command, so any run can be reproduced by hand.

## Notes

- **Memory.** On 16GB the training script walks a fallback ladder — halve the batch,
  shorten the sequence, adapt fewer layers, then accumulate — and retries on
  allocation failure rather than dying.
- **Checkpoint comparison.** MLX saves `{iter}_adapters.safetensors` at each
  `save_every`, so the 1-epoch and 2-epoch adapters can be compared directly.
- **Interpreter.** Always `./venv/bin/python3`; see Quick start.
- **Data/weights are not committed.** `.gitignore` excludes the corpus, the venv,
  the API key and the adapters.

## License

Code in this repository is provided as-is for evaluation. The source corpus is
`CC0-1.0 (Public Domain)`. The base model is
[`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit),
**Apache 2.0**, so this derivative can be redistributed and used commercially
under the same terms. Apache 2.0 §4(b)-(c) apply: mark the files as modified and
retain the upstream attribution.
