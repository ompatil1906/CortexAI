"""Prove the MLX -> PEFT conversion is numerically correct.

A LoRA adapter can be mis-converted in ways that still load, still run, and still
emit plausible JSON:

* swapping the two low-rank matrices applies the update across the wrong axis;
* getting ``lora_alpha`` wrong rescales the update by a constant factor.

Neither raises, and neither is obvious from output text. So this script checks the
thing that actually matters -- the effective weight update each format produces.

    MLX   dW = scale     * (lora_a @ lora_b)      lora_a [out, r], lora_b [r, in]
    PEFT  dW = alpha / r * (lora_B @ lora_A)      lora_B [out, r], lora_A [r, in]

If ``lora_a`` was mapped to ``lora_B`` and vice versa, and the multiplier matches,
the two products are the same matrix element for element.

One trap this deliberately avoids: comparing logits between the MLX build and
`transformers`. The MLX model is 4-bit quantized and the transformers model is
fp32, so their *base* predictions already differ on the argmax for identical
token ids. That looks like a conversion bug and is not one. The functional check
below therefore compares two **fp32** models that differ only in how the adapter
was applied.

    python scripts/verify_conversion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="numpy") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def compare_effective_updates(mlx_path: Path, peft_dir: Path, tolerance: float) -> bool:
    """Exact check: the weight update implied by each format must be identical.

    MLX applies ``y += scale * lora_b @ (lora_a @ x)`` with lora_a [in, r] and
    lora_b [r, out], so the equivalent weight delta is ``scale * (lora_b.T @ lora_a.T)``.
    PEFT applies ``(alpha/r) * (lora_B @ (lora_A @ x))``, giving
    ``(alpha/r) * (lora_B @ lora_A)``.

    Deriving the expected matrix from MLX's own formula -- rather than from the
    conversion's output -- is what makes this check independent of the conversion.
    """
    mlx = load_tensors(mlx_path)
    peft = load_tensors(peft_dir / "adapter_model.safetensors")
    config = json.loads((peft_dir / "adapter_config.json").read_text())
    multiplier = config["lora_alpha"] / config["r"]

    stems = sorted({key.rsplit(".lora_", 1)[0] for key in mlx})
    print(f"modules to compare : {len(stems)}")
    print(f"PEFT alpha / r     : {multiplier}")

    # Reference: rebuild the delta from MLX tensors using MLX's own formula, and
    # read the scale from the converter's parser rather than re-parsing the YAML.
    from convert_to_peft import read_lora_hyperparameters

    scale = read_lora_hyperparameters(REPO_ROOT / "adapters" / "final")["scale"]

    worst = 0.0
    for stem in stems:
        a = mlx[f"{stem}.lora_a"].astype(np.float32)   # [in,  r]
        b = mlx[f"{stem}.lora_b"].astype(np.float32)   # [r, out]
        expected = scale * (b.T @ a.T)                 # [out, in]

        peft_stem = f"base_model.model.{stem}"
        peft_a = peft[f"{peft_stem}.lora_A.weight"].astype(np.float32)  # [r, in]
        peft_b = peft[f"{peft_stem}.lora_B.weight"].astype(np.float32)  # [out, r]
        actual = multiplier * (peft_b @ peft_a)

        if expected.shape != actual.shape:
            print(f"  SHAPE MISMATCH {stem}: expected {expected.shape}, got {actual.shape}")
            return False
        denom = max(1e-9, float(np.abs(expected).max()))
        worst = max(worst, float(np.abs(expected - actual).max()) / denom)

    print(f"worst relative error: {worst:.3e}   (tolerance {tolerance:.0e})")
    ok = worst < tolerance
    print(f"effective updates   : {'IDENTICAL' if ok else 'DIFFER'}")
    return ok


def check_shapes_against_config(peft_dir: Path) -> bool:
    """Validate every tensor against the real model's in/out features.

    This is the check that catches a transposed (or unswapped) conversion: PEFT
    itself raises on shape mismatch only once its key prefix resolves, and Qwen's
    square q_proj/o_proj would hide a mistake that k_proj/v_proj expose.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    expected = {
        "q_proj": (cfg.hidden_size, cfg.num_attention_heads * head_dim),
        "k_proj": (cfg.hidden_size, cfg.num_key_value_heads * head_dim),
        "v_proj": (cfg.hidden_size, cfg.num_key_value_heads * head_dim),
        "o_proj": (cfg.num_attention_heads * head_dim, cfg.hidden_size),
        "gate_proj": (cfg.hidden_size, cfg.intermediate_size),
        "up_proj": (cfg.hidden_size, cfg.intermediate_size),
        "down_proj": (cfg.intermediate_size, cfg.hidden_size),
    }

    peft = load_tensors(peft_dir / "adapter_model.safetensors")
    rank = json.loads((peft_dir / "adapter_config.json").read_text())["r"]
    print(f"\nshape check against {BASE_MODEL}")

    bad = 0
    for key, tensor in sorted(peft.items()):
        module = key.split(".")[-3]
        if key.endswith(".lora_A.weight"):
            want = (rank, expected[module][0])
        else:
            want = (expected[module][1], rank)
        if tuple(tensor.shape) != want:
            if bad < 4:
                print(f"  BAD {key}: {tuple(tensor.shape)} != {want}")
            bad += 1
    print(f"tensors with wrong shape: {bad}/{len(peft)}")
    return bad == 0


def compare_forward(mlx_path: Path, peft_dir: Path, tolerance: float) -> bool:
    """Functional check in fp32 only: PEFT adapter vs a hand-merged equivalent."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nfunctional check (both fp32, so quantization is not a factor)...")
    prompt = (
        "You are CortexAI, a customer support triage assistant.\n\n"
        "Customer: My payment was declined for order [ORDER_ID] and I need to fix this today."
    )

    peft_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32),
        str(peft_dir),
    ).eval()

    # Build the same model by hand from the MLX tensors: W += scale * (a @ b).
    manual = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    mlx = load_tensors(mlx_path)
    lora_cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    scale = lora_cfg["lora_alpha"] / lora_cfg["r"]

    modules = dict(manual.named_modules())
    applied = 0
    for stem in sorted({key.rsplit(".lora_", 1)[0] for key in mlx}):
        # named_modules() on Qwen2ForCausalLM yields "model.layers.12...",
        # which is already the full path, so it is used unchanged.
        name = stem
        layer = modules.get(name)
        if layer is None:
            print(f"  could not locate module {name}")
            return False
        a = torch.from_numpy(mlx[f"{stem}.lora_a"].astype(np.float32))
        b = torch.from_numpy(mlx[f"{stem}.lora_b"].astype(np.float32))
        with torch.no_grad():
            # MLX lora_a is [in, r] and lora_b is [r, out], so the
            # equivalent delta is (lora_b.T @ lora_a.T) with shape [out, in].
            layer.weight += scale * (b.T @ a.T)
        applied += 1
    manual.eval()
    print(f"  hand-merged {applied} modules at scale {scale}")

    ids = AutoTokenizer.from_pretrained(BASE_MODEL)(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        peft_logits = peft_model(ids).logits[0, -1, :].float()
        manual_logits = manual(ids).logits[0, -1, :].float()

    same_argmax = int(peft_logits.argmax()) == int(manual_logits.argmax())
    max_abs = float((peft_logits - manual_logits).abs().max())
    spread = float(manual_logits.max() - manual_logits.min())
    print(f"  same argmax : {same_argmax}  (id {int(manual_logits.argmax())})")
    print(f"  max |delta| : {max_abs:.4f}   logit spread: {spread:.2f}")
    print(f"  relative    : {max_abs / max(1e-9, spread):.2e}")

    ok = same_argmax and max_abs / max(1e-9, spread) < tolerance
    print(f"forward outputs : {'MATCH' if ok else 'DIFFER'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlx-adapter", type=Path, default=REPO_ROOT / "adapters" / "final")
    parser.add_argument("--peft", type=Path, default=REPO_ROOT / "peft" / "final")
    parser.add_argument("--tensor-tolerance", type=float, default=1e-5)
    parser.add_argument("--forward-tolerance", type=float, default=1e-3)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    mlx_path = args.mlx_adapter / "adapters.safetensors"
    if not mlx_path.exists():
        raise SystemExit(f"missing {mlx_path}")
    if not (args.peft / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing {args.peft / 'adapter_model.safetensors'} - run convert_to_peft.py")

    print("=== 1. shapes vs the real model config ===")
    ok = check_shapes_against_config(args.peft)

    print("\n=== 2. effective weight updates (exact tensor math) ===")
    ok = compare_effective_updates(mlx_path, args.peft, args.tensor_tolerance) and ok

    if ok and not args.skip_forward:
        print("\n=== 3. forward pass (fp32 only) ===")
        ok = compare_forward(mlx_path, args.peft, args.forward_tolerance)

    print("\n" + ("PASS: the PEFT adapter faithfully represents the MLX adapter."
          if ok else "FAIL: the conversion is not faithful."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()