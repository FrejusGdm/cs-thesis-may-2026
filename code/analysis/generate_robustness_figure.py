"""
generate_robustness_figure.py — Grouped bar chart for appendix robustness section.

Shows BLEU for 4 models × {Random-10K, Structured-4K, R10K+S4K} side by side.
Reads from experiments/results/summary/robustness_table.csv (output of compute_robustness_table.py).

Usage:
    python generate_robustness_figure.py
Output:
    experiments/results/figures/fig_robustness_models.{pdf,png}
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

# ── Paths ──
ROOT = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "experiments/results/summary/robustness_table.csv"
OUT_DIR  = ROOT / "experiments/results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Conditions to show (in display order) ──
# Using R6K+S4K as combined condition — all models including Gemini have this.
CONDITIONS = {
    "RANDOM-10K":              "Random-10K",
    "STRUCTURED-4K-ONLY":      "Structured-4K",
    "RANDOM-6K_STRUCTURED-4K": "R6K+S4K",
}

# ── Model display order and labels ──
MODEL_ORDER = [
    "NLLB-600M (main paper, 5-seed mean)",
    "NLLB-1.3B",
    "mBART-50 (French init)",
    "mBART-50 (Random init)",
    "Gemini-2.5 (fine-tuned, seed 42)",
]
MODEL_SHORT = {
    "NLLB-600M (main paper, 5-seed mean)": "NLLB\n600M",
    "NLLB-1.3B":                           "NLLB\n1.3B",
    "mBART-50 (French init)":              "mBART\n(Fr-init)",
    "mBART-50 (Random init)":              "mBART\n(Rand-init)",
    "Gemini-2.5 (fine-tuned, seed 42)":   "Gemini\n2.5",
}

COLORS = {
    "Random-10K":    "#d62728",   # red   — random only
    "Structured-4K": "#2ca02c",   # green — structured only
    "R6K+S4K":       "#1f77b4",   # blue  — combined
}


def load_csv(path: Path) -> dict:
    """Returns {(model, condition): bleu_mean}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            model = row["model"]
            cond  = row["condition"]
            bleu  = row.get("bleu_mean", "")
            if bleu:
                try:
                    data[(model, cond)] = float(bleu)
                except ValueError:
                    pass
    return data


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run compute_robustness_table.py first.")
        sys.exit(1)

    data = load_csv(CSV_PATH)

    # Filter to models that have at least one condition
    models_present = [m for m in MODEL_ORDER
                      if any((m, c) in data for c in CONDITIONS)]

    if not models_present:
        print("No data found in CSV.")
        sys.exit(1)

    n_models = len(models_present)
    n_conds  = len(CONDITIONS)
    cond_keys = list(CONDITIONS.keys())
    cond_labels = list(CONDITIONS.values())

    # ── Layout ──
    bar_width = 0.22
    group_gap = 0.1
    group_width = n_conds * bar_width + group_gap
    x = np.arange(n_models) * group_width

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.6), 4.5))
    fig.patch.set_facecolor("white")

    for ci, (cond_key, cond_label) in enumerate(CONDITIONS.items()):
        color = COLORS[cond_label]
        offsets = x + ci * bar_width - (n_conds - 1) * bar_width / 2

        bleus = []
        for m in models_present:
            v = data.get((m, cond_key))
            bleus.append(v if v is not None else 0.0)

        bars = ax.bar(offsets, bleus, width=bar_width * 0.9,
                      color=color, alpha=0.85, label=cond_label,
                      edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, bleus):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}",
                        ha="center", va="bottom",
                        fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models_present],
                       fontsize=9, family="serif")
    ax.set_ylabel("BLEU", fontsize=10, family="serif")
    ax.set_ylim(0, max(
        (data.get((m, c), 0) or 0)
        for m in models_present for c in CONDITIONS
    ) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(title="Training data", title_fontsize=8,
              fontsize=8, framealpha=0.7,
              loc="upper left", ncol=1)

    ax.set_title("Structured-data advantage across model architectures",
                 fontsize=10, family="serif", pad=8)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_robustness_models.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.close()


if __name__ == "__main__":
    main()
