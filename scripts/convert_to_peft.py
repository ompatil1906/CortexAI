"""Convert an MLX LoRA adapter into PEFT format for use outside MLX.

Why this exists
---------------
`mlx_lm lora` writes adapters in MLX's own layout. For a ``Linear(in, out)``,
MLX computes ``y = W x + scale * (lora_b @ (lora_a @ x))`` and therefore stores:

    MLX   lora_a : [in,  r]        lora_b : [r, out]
    PEFT  lora_A : [r,  in]        lora_B : [out, r]

Both matrices are the **transpose** of the MLX original, with the same letter role.
Getting this wrong is silent: the shapes stay plausible, the model still loads, and
it still emits well-formed JSON, because ``q_proj`` and ``o_proj`` are square in
Qwen2.5-1.5B (1536x1536) and only ``k_proj``/``v_proj`` (1536->256, GQA) expose the
error. `verify_conversion.py` checks the shapes against the real model config and
compares the effective weight update, either of which catches it immediately.

Usage:
    python scripts/convert_to_peft.py --adapter adapters/final --out peft/final
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent

#: MLX and PEFT use the same letter for the same role; only the layout differs.
#: Each MLX matrix is stored transposed relative to PEFT's expectation.
PEFT_ROLE = {"lora_a": "lora_A", "lora_b": "lora_B"}

#: PEFT prefixes adapter weights under the wrapped model.
PEFT_PREFIX = "base_model.model."


def convert_tensors(source: Path) -> tuple[dict[str, np.ndarray], list[str], list[int]]:
    """Rewrite MLX LoRA tensors into PEFT keys.

    Returns (tensors, target_modules, layer_indices).
    """
    with safe_open(source, framework="numpy") as handle:
        keys = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in keys}

    out: dict[str, np.ndarray] = {}
    targets: set[str] = set()
    layers: set[int] = set()

    for key, tensor in tensors.items():
        if ".lora_a" not in key and ".lora_b" not in key:
            raise SystemExit(
                f"unexpected tensor {key!r}: not an MLX LoRA matrix. "
                "Is this a merged or full checkpoint rather than an adapter?"
            )

        stem, _, suffix = key.rpartition(".")
        role = PEFT_ROLE.get(suffix)
        if role is None:
            raise SystemExit(f"unrecognised LoRA suffix {suffix!r} in {key!r}")

        # MLX "model.layers.12.mlp.down_proj" -> PEFT
        # "base_model.model.model.layers.12.mlp.down_proj.lora_A.weight".
        # PEFT's saved keys carry TWO "model." segments: one from the PeftModel
        # wrapper and one from the Qwen2Model itself. Do not strip the inner one.
        module = stem
        targets.add(module.rsplit(".", 1)[-1])

        parts = module.split(".")
        if "layers" in parts:
            layers.add(int(parts[parts.index("layers") + 1]))

        out[f"{PEFT_PREFIX}{module}.{role}.weight"] = np.ascontiguousarray(tensor.T)

    return out, sorted(targets), sorted(layers)


def read_lora_hyperparameters(adapter_dir: Path) -> dict:
    """Recover rank/alpha/dropout from MLX's config.

    MLX applies ``scale`` directly as the multiplier on the low-rank update,
    while PEFT computes its multiplier as ``alpha / r``. To keep the two
    identical, ``lora_alpha`` must be ``scale * rank`` so that ``alpha / r == scale``. Getting this wrong rescales the adapter's
    contribution by a constant factor -- again, silent.
    """
    lora_config = adapter_dir / "lora_config.yaml"
    run_config = adapter_dir / "adapter_config.json"

    scale = rank = dropout = None
    if lora_config.exists():
        for line in lora_config.read_text().splitlines():
            stripped = line.strip()
            for field in ("rank", "scale", "dropout"):
                if stripped.startswith(f"{field}:"):
                    value = float(stripped.split(":", 1)[1])
                    if field == "rank":
                        rank = int(value)
                    elif field == "dropout":
                        dropout = value
                    else:
                        scale = value

    if rank is None and run_config.exists():
        parameters = json.loads(run_config.read_text()).get("lora_parameters", {})
        rank = parameters.get("rank")
        scale = parameters.get("scale", scale)
        dropout = parameters.get("dropout", dropout)

    if rank is None or scale is None:
        raise SystemExit(
            f"could not read rank/scale from {lora_config} or {run_config}; "
            "the adapter was not produced by scripts/train_lora.py"
        )

    return {"r": rank, "lora_alpha": scale * rank, "lora_dropout": dropout or 0.0, "scale": scale}


def build_peft_config(base_model: str, rank: int, alpha: float, dropout: float,
                      targets: list[str], layers: list[int]) -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "fan_in_fan_out": False,
        # False rather than True so the published adapter can also be used as the
        # starting point for further fine-tuning, not inference only.
        "inference_mode": False,
        "target_modules": sorted(targets),
        "modules_to_save": None,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "revision": None,
        "layers_to_transform": layers or None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "loftq_config": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "adapters" / "final")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "peft" / "final")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--keep-mlx-config", action="store_true", help="also copy lora_config.yaml alongside")
    args = parser.parse_args()

    source = args.adapter / "adapters.safetensors"
    if not source.exists():
        raise SystemExit(f"missing {source} - train an adapter first")

    tensors, targets, layers = convert_tensors(source)
    meta = read_lora_hyperparameters(args.adapter)
    config = build_peft_config(args.base_model, meta["r"], meta["lora_alpha"],
                               meta["lora_dropout"], targets, layers)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "adapter_model.safetensors"))
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.keep_mlx_config:
        shutil.copy(args.adapter / "lora_config.yaml", args.out / "lora_config.yaml")

    shapes = Counter(tuple(t.shape) for t in tensors.values())
    print(f"wrote {args.out}")
    print(f"  tensors            : {len(tensors)}")
    print(f"  target_modules     : {', '.join(targets)}")
    print(f"  layers_to_transform: {layers}")
    print(f"  r={meta['r']}  lora_alpha={meta['lora_alpha']}  dropout={meta['lora_dropout']}")
    print(f"  alpha/r            : {meta['lora_alpha'] / meta['r']:.4f} (MLX scale was {meta['scale']})")
    print(f"  distinct shapes    : {len(shapes)}")
    print(f"\nload it with:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base, \"{args.out}\")")


if __name__ == "__main__":
    main()