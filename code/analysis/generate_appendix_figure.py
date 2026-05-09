"""
generate_appendix_figure.py — 3-panel robustness figure for appendix.

Panels:
  A (left):  BLEU heatmap  (model × condition)
  B (middle): chrF++ heatmap (model × condition)
  C (right): Seed-variance bar chart (BLEU std, mBART models only)

Input:  experiments/results/summary/robustness_table.csv
Output: experiments/results/figures/fig_appendix_robustness.{pdf,png}
"""

import sys
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT     = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / "experiments/results/summary/robustness_table.csv"
OUT_DIR  = ROOT / "experiments/results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Display order ──
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

COND_ORDER = [
    "RANDOM-10K",
    "STRUCTURED-4K-ONLY",
    "RANDOM-6K_STRUCTURED-4K",
    "RANDOM-10K_STRUCTURED-4K",
]
COND_SHORT = {
    "RANDOM-10K":               "Random\n10K",
    "STRUCTURED-4K-ONLY":       "Struct.\n4K",
    "RANDOM-6K_STRUCTURED-4K":  "R6K+\nS4K",
    "RANDOM-10K_STRUCTURED-4K": "R10K+\nS4K",
}

# Panel C: only models with multiple seeds
VARIANCE_MODELS = [
    "NLLB-1.3B",
    "mBART-50 (French init)",
    "mBART-50 (Random init)",
]
VARIANCE_MODEL_SHORT = {
    "NLLB-1.3B":              "NLLB 1.3B",
    "mBART-50 (French init)":  "mBART (Fr-init)",
    "mBART-50 (Random init)":  "mBART (Rand-init)",
}
VARIANCE_CONDS = [
    "RANDOM-10K",
    "STRUCTURED-4K-ONLY",
    "RANDOM-10K_STRUCTURED-4K",
]
VARIANCE_COND_LABEL = {
    "RANDOM-10K":               "Random-10K",
    "STRUCTURED-4K-ONLY":       "Struct.-4K",
    "RANDOM-10K_STRUCTURED-4K": "R10K+S4K",
}
VARIANCE_COLORS = {
    "RANDOM-10K":               "#d62728",
    "STRUCTURED-4K-ONLY":       "#2ca02c",
    "RANDOM-10K_STRUCTURED-4K": "#1f77b4",
}


def load_csv(path: Path) -> dict:
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            key = (row["model"], row["condition"])
            data[key] = {
                "bleu_mean":   float(row["bleu_mean"])   if row.get("bleu_mean")   else None,
                "bleu_std":    float(row["bleu_std"])    if row.get("bleu_std")    else None,
                "chrfpp_mean": float(row["chrfpp_mean"]) if row.get("chrfpp_mean") else None,
                "chrfpp_std":  float(row["chrfpp_std"])  if row.get("chrfpp_std")  else None,
                "n": int(row["n"]) if row.get("n") else 0,
            }
    return data


def build_matrix(data: dict, metric: str, models: list, conds: list) -> np.ndarray:
    mat = np.full((len(models), len(conds)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(conds):
            v = data.get((m, c), {}).get(metric)
            if v is not None:
                mat[i, j] = v
    return mat


def draw_heatmap(ax, mat, row_labels, col_labels, title, vmin, vmax, cmap="Blues"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Single masked imshow — NaN cells rendered as grey via set_bad
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d0d0d0")
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")

    # Hatch pattern over missing cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="////",
                    edgecolor="#aaaaaa", linewidth=0
                ))

    # Cell annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#888888", family="serif")
            else:
                val = mat[i, j]
                brightness = norm(val)
                text_color = "white" if brightness > 0.62 else "#1a1a1a"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9.5, color=text_color, fontweight="bold",
                        family="serif")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5, family="serif")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5, family="serif")
    ax.set_title(title, fontsize=10, family="serif", pad=7, fontweight="bold")

    # White grid lines between cells
    for x in np.arange(-0.5, len(col_labels), 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, len(row_labels), 1):
        ax.axhline(y, color="white", linewidth=1.5)

    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, fraction=0.046, pad=0.04)


def draw_variance(ax, data, models, conds, model_short, cond_labels, colors):
    bar_h = 0.22
    group_gap = 0.15
    y = np.arange(len(models)) * (len(conds) * bar_h + group_gap)

    for ci, cond in enumerate(conds):
        stds = [data.get((m, cond), {}).get("bleu_std") or 0.0 for m in models]
        offsets = y + ci * bar_h - (len(conds) - 1) * bar_h / 2
        bars = ax.barh(offsets, stds, height=bar_h * 0.85,
                       color=colors[cond], alpha=0.85,
                       label=cond_labels[cond],
                       edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, stds):
            if val > 0:
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([model_short[m] for m in models],
                       fontsize=8.5, family="serif")
    ax.set_xlabel("BLEU std dev (5 seeds)", fontsize=8, family="serif")
    ax.set_title("Seed Variance\n(BLEU std)", fontsize=10, family="serif",
                 pad=7, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.7, loc="lower right",
              title="Condition", title_fontsize=7)


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run compute_robustness_table.py first.")
        sys.exit(1)

    data = load_csv(CSV_PATH)

    row_labels = [MODEL_SHORT[m] for m in MODEL_ORDER]
    col_labels = [COND_SHORT[c] for c in COND_ORDER]

    bleu_mat   = build_matrix(data, "bleu_mean",   MODEL_ORDER, COND_ORDER)
    chrfpp_mat = build_matrix(data, "chrfpp_mean", MODEL_ORDER, COND_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    draw_heatmap(axes[0], bleu_mat, row_labels, col_labels,
                 "BLEU  (↑ higher is better)", vmin=0, vmax=25, cmap="Blues")
    draw_heatmap(axes[1], chrfpp_mat, row_labels, col_labels,
                 "chrF++  (↑ higher is better)", vmin=0, vmax=42, cmap="Blues")

    fig.suptitle(
        "Structured-data advantage confirmed by both metrics across all architectures",
        fontsize=10, family="serif", y=1.03
    )

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_appendix_robustness.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.close()


if __name__ == "__main__":
    main()
