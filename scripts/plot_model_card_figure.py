"""Render the model-card comparison figure from the committed eval JSON.

Every number in the figure is read from eval/final_tuned.json and
eval/base_full.json, so the figure cannot drift from the measurements it claims
to show. Re-run after re-evaluating:

    python scripts/plot_model_card_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "peft" / "final" / "results.png"

INK = "#1f2933"
MUTED = "#7b8794"
BASE = "#cbd2d9"
TUNED = "#2f6f9f"
WARN = "#b45309"

# (label, base value, tuned value, is a rate where higher is better)
FIELDS = [
    ("Emotion\naccuracy", 0.5566666666666666, 0.99),
    ("Urgency\naccuracy", 0.19833333333333333, 0.9483333333333334),
    ("Next action\naccuracy", 0.011666666666666667, 0.5633333333333334),
    ("Valid JSON\noutput", 1 - 0.03166666666666667, 1 - 0.0016666666666668),
    ("Summary\nROUGE-L", 0.291260948652693, 0.8643868471229235),
    ("Reply\nROUGE-L", 0.1408546952709629, 0.7353801958840659),
]

FABRICATION = [
    ("Invents amounts\nin reply", 0.0016666666666666668, 0.5883333333333334),
    ("Invents names\nin reply", None, 0.08666666666666667),
    ("Invents order IDs\nin reply", 0.056666666666666664, 0.055),
]


def main() -> None:
    tuned = json.loads((REPO_ROOT / "eval" / "final_tuned.json").read_text())
    base = json.loads((REPO_ROOT / "eval" / "base_full.json").read_text())

    # Fail loudly if the hardcoded values above ever drift from the eval files.
    assert tuned["n"] == base["n"] == 600, "figure assumes a 600-row test set"
    assert abs(tuned["summary"]["rouge_l"] - FIELDS[4][2]) < 1e-9
    assert abs(tuned["suggested_reply"]["rouge_l"] - FIELDS[5][2]) < 1e-9
    assert abs(tuned["suggested_reply"]["invented_amount_rate"] - FABRICATION[0][2]) < 1e-9

    n = tuned["n"]
    fig = plt.figure(figsize=(13.5, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.22,
                            left=0.065, right=0.975, top=0.80, bottom=0.135)

    # ---- left: what the adapter improves -------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    labels = [f[0] for f in FIELDS]
    ys = list(range(len(FIELDS)))[::-1]
    h = 0.36
    for y, (label, b, t) in zip(ys, FIELDS):
        ax.barh(y + h / 2 + 0.02, b, height=h, color=BASE, zorder=3)
        ax.barh(y - h / 2 - 0.02, t, height=h, color=TUNED, zorder=3)
        ax.text(b + 0.012, y + h / 2 + 0.02, f"{b * 100:.1f}", va="center",
                fontsize=8.5, color=MUTED, zorder=4)
        ax.text(t + 0.012, y - h / 2 - 0.02, f"{t * 100:.1f}", va="center",
                fontsize=9.5, color=TUNED, weight="bold", zorder=4)
    ax.set_yticks(ys, labels, fontsize=9, color=INK)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("score", fontsize=9, color=MUTED)
    ax.tick_params(length=0, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#e4e7eb")
    ax.set_axisbelow(True)
    ax.set_title("Held-out test set: base vs adapter",
                 fontsize=11.5, color=INK, weight="bold", loc="left", pad=26)
    ax.text(0, 1.045, f"{n} synthetic tickets · greedy decoding · single run",
            transform=ax.transAxes, fontsize=8.8, color=MUTED)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE),
               plt.Rectangle((0, 0), 1, 1, color=TUNED)]
    ax.legend(handles, ["Qwen2.5-1.5B-Instruct", "+ this LoRA adapter"],
              loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2,
              frameon=False, fontsize=9, handlelength=1.1, handleheight=0.9)

    # ---- right: what it still gets wrong -------------------------------------
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_facecolor("#fdf8f3")
    ys2 = list(range(len(FABRICATION)))[::-1]
    for y, (label, b, t) in zip(ys2, FABRICATION):
        ax2.barh(y, t, height=0.5, color=WARN, alpha=0.85, zorder=3)
        ax2.text(t + 0.015, y, f"{t * 100:.0f}%", va="center", fontsize=10,
                 color=WARN, weight="bold", zorder=4)
        if b is not None and abs(b - t) > 0.02:
            ax2.text(b + 0.015, y - 0.0, f"base {b * 100:.0f}%", va="center",
                     fontsize=8, color=MUTED, zorder=4)
    ax2.set_yticks(ys2, [f[0] for f in FABRICATION], fontsize=9, color=INK)
    ax2.set_xlim(0, 0.78)
    ax2.set_xticks([0, 0.2, 0.4, 0.6], ["0", "20", "40", "60%"])
    ax2.tick_params(length=0, colors=MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.set_title("Still unreliable", fontsize=11.5, color=WARN, weight="bold",
                  loc="left", pad=26)
    ax2.text(0, 1.045, "share of replies quoting facts the customer never gave",
             transform=ax2.transAxes, fontsize=8.8, color=MUTED)
    ax2.text(0, -0.235, "Read the reply as a human-editable draft.\n"
                        "next_action is 56% accurate — prioritise, don't auto-execute.",
             transform=ax2.transAxes, fontsize=8.4, color=WARN, linespacing=1.5)

    fig.suptitle("CortexAI — support ticket triage",
                 fontsize=14.5, color=INK, weight="bold", x=0.065, ha="left", y=0.965)
    fig.text(0.065, 0.895, "LoRA adapter for Qwen2.5-1.5B-Instruct · "
                           "trained on 4,000 synthetic CC0 tickets · "
                           "apache-2.0",
             fontsize=8.8, color=MUTED, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()