"""LoRA fine-tuning wrapper around ``mlx_lm.lora``, tuned for a 16GB MacBook Air.

Responsibilities beyond invoking the CLI:

*   Turn ``epochs`` into ``iters`` (mlx-lm only accepts an iteration count).
*   Write the ``lora_parameters`` YAML, which has no CLI flag -- rank, scale and
    dropout can only be set from a config file. Note mlx-lm's LoRA config uses
    ``scale``, not ``alpha``; passing ``alpha`` raises ``KeyError``.
*   **Memory ladder.** On a 16GB unified-memory machine a 3.8B 4-bit model plus
    activations can fail to allocate. Rather than making that a manual debugging
    session, each rung below is tried in order until one starts: halve the batch,
    shorten the sequence, adapt fewer layers, then accumulate gradients instead.
*   Record the exact resolved command so a run can be reproduced by hand.

Usage:
    python scripts/train_lora.py --config configs/pilot.json
    python scripts/train_lora.py --config configs/full.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = REPO_ROOT / "data" / "sft"
ADAPTER_ROOT = REPO_ROOT / "adapters"
RUNS_DIR = REPO_ROOT / "runs"

#: Tried in order when the previous rung fails to allocate. Later rungs trade speed
#: for headroom, so the first one that fits is the fastest option available.
MEMORY_LADDER: list[dict] = [
    {},
    {"batch_size": 2, "grad_accumulation_steps": 2},
    {"batch_size": 2, "grad_accumulation_steps": 2, "max_seq_length": 384},
    {"batch_size": 1, "grad_accumulation_steps": 4, "max_seq_length": 384, "num_layers": 8},
]

OOM_MARKERS = (
    "insufficient memory",
    "out of memory",
    "bad_alloc",
    "metal command buffer failed",
    "failed to allocate",
    "total bytes of encoded data",
)

VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python3"


def resolve_python() -> str:
    """Pick an interpreter that can actually import mlx_lm.

    ``python3`` on this machine resolves to Homebrew's interpreter while mlx lives in
    the repo venv, so trusting ``sys.executable`` (or PATH) launches a child that dies
    with "No module named mlx_lm". Prefer the venv explicitly, then verify by import
    rather than by name.
    """
    candidates = []
    if VENV_PYTHON.exists():
        candidates.append(str(VENV_PYTHON))
    candidates.append(sys.executable)

    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "import mlx_lm"], capture_output=True, timeout=120
            )
            if probe.returncode == 0:
                return candidate
        except Exception:
            continue

    raise SystemExit(
        "No interpreter with mlx_lm found. Tried:\n  "
        + "\n  ".join(candidates)
        + f"\nActivate the venv ({VENV_PYTHON}) or run:\n"
        f"  {VENV_PYTHON} -m pip install mlx mlx-lm"
    )


def count_rows(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"missing {path} - run scripts/build_sft.py first")
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def make_subset(source: Path, dest: Path, rows: int) -> int:
    """Materialise a training subset so pilots do not read all rows.

    ``rows <= 0`` means "use every row", but the destination must still be populated:
    the caller passes ``--data data/sft/<name>`` to mlx-lm, and mlx-lm requires
    train/valid/test to exist there. Returning early without copying produced
    "built subset ... with 4,000 train rows" followed by
    "missing data/sft/final/train.jsonl", because the caller's own verification
    looked in a directory that had never been created.
    """
    source = source / "train.jsonl" if source.is_dir() else source
    if not source.exists():
        raise SystemExit(f"missing {source} - run scripts/build_sft.py first")

    dest.mkdir(parents=True, exist_ok=True)
    if rows <= 0:
        shutil.copy(source, dest / "train.jsonl")
        kept = count_rows(dest / "train.jsonl")
    else:
        kept = 0
        with source.open() as src, (dest / "train.jsonl").open("w") as sink:
            for line in src:
                if not line.strip():
                    continue
                sink.write(line)
                kept += 1
                if kept >= rows:
                    break

    # Validation and test are copied whole so eval stays comparable across runs.
    for name in ("valid.jsonl", "test.jsonl"):
        shutil.copy(source.parent / name, dest / name)

    # Verify rather than trust: a row count that does not match what we just wrote
    # means the write did not land.
    written = count_rows(dest / "train.jsonl")
    if written != kept:
        raise SystemExit(
            f"subset verification failed: wrote {kept} rows to {dest / 'train.jsonl'} "
            f"but read back {written}"
        )
    return kept


def build_command(config: dict, data_dir: Path, adapter_path: Path, overrides: dict, python: str) -> list[str]:
    effective = {**config, **overrides}
    rows = count_rows(data_dir / "train.jsonl")
    effective_batch = effective["batch_size"] * effective["grad_accumulation_steps"]
    iters = max(1, math.ceil(rows / effective_batch) * effective["epochs"])

    lora_config = adapter_path / "lora_config.yaml"
    lora_config.parent.mkdir(parents=True, exist_ok=True)
    # Hand-written rather than json.dumps: the file must be YAML that mlx-lm's loader
    # reads as a nested mapping, and keeping it literal makes the three keys obvious.
    lora_config.write_text(
        "# Generated by scripts/train_lora.py -- do not edit by hand.\n"
        "# mlx-lm names this key `scale`, not `alpha`; `alpha` raises KeyError.\n"
        "lora_parameters:\n"
        f"  rank: {effective['lora']['rank']}\n"
        f"  scale: {effective['lora']['scale']}\n"
        f"  dropout: {effective['lora']['dropout']}\n"
    )

    command = [
        python, "-m", "mlx_lm", "lora",
        "--model", effective["model"],
        "--train",
        "--data", str(data_dir),
        "-c", str(lora_config),
        "--fine-tune-type", "lora",
        "--batch-size", str(effective["batch_size"]),
        "--grad-accumulation-steps", str(effective["grad_accumulation_steps"]),
        "--max-seq-length", str(effective["max_seq_length"]),
        "--num-layers", str(effective["num_layers"]),
        "--iters", str(iters),
        "--learning-rate", str(effective["learning_rate"]),
        "--val-batches", str(effective["val_batches"]),
        "--steps-per-report", str(effective["steps_per_report"]),
        "--steps-per-eval", str(effective["steps_per_eval"]),
        "--save-every", str(effective["save_every"]),
        "--adapter-path", str(adapter_path),
    ]
    if effective["mask_prompt"]:
        command.append("--mask-prompt")
    if effective["grad_checkpoint"]:
        command.append("--grad-checkpoint")
    return command


def looks_like_oom(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-ladder", action="store_true", help="fail instead of retrying on OOM")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    name = config.get("name", args.config.stem)

    data_dir = args.data_dir or (SFT_DIR / name)
    if data_dir == SFT_DIR:
        print("warning: training on the full SFT split", flush=True)
    elif not (data_dir / "train.jsonl").exists():
        rows = make_subset(SFT_DIR, data_dir, config.get("train_rows", 0))
        print(f"built subset {data_dir} with {rows:,} train rows", flush=True)

    adapter_path = args.adapter_path or (ADAPTER_ROOT / name)
    adapter_path.mkdir(parents=True, exist_ok=True)

    python = resolve_python()
    print(f"interpreter: {python}", flush=True)

    ladder = MEMORY_LADDER[:1] if args.no_ladder else MEMORY_LADDER
    last_output = ""

    for attempt, overrides in enumerate(ladder):
        command = build_command(config, data_dir, adapter_path, overrides, python)
        print(f"\n{'=' * 78}")
        print(f"attempt {attempt + 1}/{len(ladder)}  overrides={overrides or '{}'}")
        print(f"{'=' * 78}")
        print("  " + " ".join(command).replace(str(REPO_ROOT) + "/", ""), flush=True)

        if args.dry_run:
            return

        started = time.time()
        log_path = RUNS_DIR / name / f"train_attempt{attempt + 1}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream rather than capture: a full 25K run takes tens of minutes and
        # buffering it means no way to tell "slow" from "hung".
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            collected: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                collected.append(line)
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)
            returncode = process.wait()

        elapsed = time.time() - started
        output = "".join(collected)
        last_output = output

        if returncode == 0:
            print(f"\ntrained in {elapsed / 60:.1f} min -> {adapter_path}")
            manifest = {
                "config_name": name,
                "config": config,
                "overrides": overrides,
                "command": command,
                "train_rows": count_rows(data_dir / "train.jsonl"),
                "elapsed_minutes": round(elapsed / 60, 2),
            }
            (adapter_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"wrote {adapter_path / 'run_manifest.json'}")
            return

        if looks_like_oom(output) and attempt + 1 < len(ladder):
            print(
                f"\nallocation failure detected; stepping down the memory ladder "
                f"(see {log_path})",
                flush=True,
            )
            continue

        print(f"\ntraining failed with exit code {returncode}", file=sys.stderr)
        print(last_output[-3000:], file=sys.stderr)
        raise SystemExit(returncode)

    print("\nexhausted the memory ladder without a successful run", file=sys.stderr)
    print(last_output[-3000:], file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()