"""
generate_paper_figures.py — Publication-quality figures for the paper.

Reads results/summary/all_results.csv and produces:
  - Fig 2: Data efficiency curves (structured vs random scaling)
  - Fig 3: Replacement curve (fixed 10K budget, varying composition)
  - Fig 4: Module leave-one-out ablation bars
  - Fig 5: Structure matters (PAIRS-INTACT vs BROKEN + verb diversity)
  - Fig 6: Additive curve (structured added to 6K random base)

Usage:
    python experiments/analysis/generate_paper_figures.py
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

COLORS = {
    "structured": "#2166AC",   # deep blue
    "random": "#D6604D",       # warm red
    "combined": "#4DAF4A",     # green
    "baseline": "#984EA3",     # purple
    "accent": "#FF7F00",       # orange
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
CSV_PATH = os.path.join(ROOT, "results", "summary", "all_results.csv")
OUT_DIR = os.path.join(ROOT, "results", "figures")


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


def agg(df, groupcols):
    """Group and compute mean/std for BLEU, chrF, chrF++."""
    g = df.groupby(groupcols).agg(
        bleu_mean=("test_bleu", "mean"),
        bleu_std=("test_bleu", "std"),
        chrf_mean=("test_chrf", "mean"),
        chrf_std=("test_chrf", "std"),
        chrfpp_mean=("test_chrfpp", "mean"),
        chrfpp_std=("test_chrfpp", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g["bleu_std"] = g["bleu_std"].fillna(0)
    g["chrf_std"] = g["chrf_std"].fillna(0)
    g["chrfpp_std"] = g["chrfpp_std"].fillna(0)
    return g


def extract_size(condition):
    """Extract numeric training size from condition name."""
    m = re.search(r"(\d+)", condition)
    return int(m.group(1)) if m else 0


# ── Figure 2: Data Efficiency Curves ─────────────────────────────────────────

def fig2_data_efficiency(df):
    """Structured vs Random scaling curves."""
    exp2 = df[df["experiment"] == "exp2"].copy()

    # Structured scaling
    struct_conds = [c for c in exp2["condition"].unique() if c.startswith("STRUCTURED-")]
    struct = exp2[exp2["condition"].isin(struct_conds)].copy()
    struct["size"] = struct["condition"].apply(extract_size)
    struct_agg = agg(struct, ["condition", "size"]).sort_values("size")

    # Random scaling
    rand_conds = [c for c in exp2["condition"].unique()
                  if c.startswith("RANDOM-") and "STRUCTURED" not in c]
    rand = exp2[exp2["condition"].isin(rand_conds)].copy()
    rand["size"] = rand["condition"].apply(extract_size)
    rand_agg = agg(rand, ["condition", "size"]).sort_values("size")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Random line
    ax.plot(rand_agg["size"], rand_agg["bleu_mean"], "o-",
            color=COLORS["random"], linewidth=2, markersize=6, label="Random (Tatoeba)")
    ax.fill_between(rand_agg["size"],
                    rand_agg["bleu_mean"] - rand_agg["bleu_std"],
                    rand_agg["bleu_mean"] + rand_agg["bleu_std"],
                    color=COLORS["random"], alpha=0.15)

    # Structured line
    ax.plot(struct_agg["size"], struct_agg["bleu_mean"], "s-",
            color=COLORS["structured"], linewidth=2, markersize=6, label="Structured (Curriculum)")
    ax.fill_between(struct_agg["size"],
                    struct_agg["bleu_mean"] - struct_agg["bleu_std"],
                    struct_agg["bleu_mean"] + struct_agg["bleu_std"],
                    color=COLORS["structured"], alpha=0.15)

    # Annotation
    ax.annotate("200 structured > 10K random",
                xy=(200, struct_agg[struct_agg["size"] == 200]["bleu_mean"].values[0]),
                xytext=(1500, 12),
                fontsize=9, fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                color="gray")

    # Horizontal reference line for random plateau
    ax.axhline(y=rand_agg["bleu_mean"].max(), color=COLORS["random"],
               linestyle="--", alpha=0.4, linewidth=1)
    ax.text(8500, rand_agg["bleu_mean"].max() + 0.5,
            f"Random plateau: {rand_agg['bleu_mean'].max():.1f}",
            fontsize=8, color=COLORS["random"], alpha=0.7)

    ax.set_xlabel("Training Sentences")
    ax.set_ylabel("BLEU")
    ax.set_title("Data Efficiency: Structured vs. Random")
    ax.legend(loc="center right")
    ax.set_xlim(-200, 10500)
    ax.set_ylim(-1, 25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig2_data_efficiency.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 2: Data efficiency curves")


# ── Figure 3: Replacement Curve ──────────────────────────────────────────────

def fig3_replacement_curve(df):
    """Fixed ~10K budget, varying % structured."""
    exp2 = df[df["experiment"] == "exp2"].copy()

    # Build the data points manually since conditions have different naming
    points = []

    # Pure random 10K
    r10k = exp2[exp2["condition"] == "RANDOM-10000"]
    if len(r10k) > 0:
        points.append({"pct_structured": 0, "label": "100% Random",
                        "bleu_mean": r10k["test_bleu"].mean(),
                        "bleu_std": r10k["test_bleu"].std()})

    # REPLACE conditions
    for cond, pct in [("REPLACE-R9500_S500", 5), ("REPLACE-R9000_S1000", 10),
                       ("REPLACE-R8000_S2000", 20), ("REPLACE-R6000_S4000", 40)]:
        sub = exp2[exp2["condition"] == cond]
        if len(sub) > 0:
            points.append({"pct_structured": pct, "label": cond,
                            "bleu_mean": sub["test_bleu"].mean(),
                            "bleu_std": sub["test_bleu"].std()})

    if not points:
        print("  [SKIP] Fig 3: No replacement curve data found")
        return

    pts = pd.DataFrame(points).sort_values("pct_structured")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pts["pct_structured"], pts["bleu_mean"], "D-",
            color=COLORS["combined"], linewidth=2, markersize=8)
    ax.fill_between(pts["pct_structured"],
                    pts["bleu_mean"] - pts["bleu_std"],
                    pts["bleu_mean"] + pts["bleu_std"],
                    color=COLORS["combined"], alpha=0.15)

    # Mark optimal point
    best_idx = pts["bleu_mean"].idxmax()
    best = pts.loc[best_idx]
    ax.scatter([best["pct_structured"]], [best["bleu_mean"]],
               s=120, color=COLORS["accent"], zorder=5, edgecolor="white", linewidth=2)
    ax.annotate(f"Optimal: {best['pct_structured']:.0f}% structured\nBLEU {best['bleu_mean']:.1f}",
                xy=(best["pct_structured"], best["bleu_mean"]),
                xytext=(best["pct_structured"] + 5, best["bleu_mean"] - 3),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    ax.set_xlabel("% Structured Data in 10K Budget")
    ax.set_ylabel("BLEU")
    ax.set_title("Budget Allocation: Replacing Random with Structured")
    ax.set_xticks(pts["pct_structured"].values)
    ax.set_xticklabels([f"{int(p)}%" for p in pts["pct_structured"]])

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig3_replacement_curve.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 3: Replacement curve")


# ── Figure 4: Module LOO Ablation ────────────────────────────────────────────

def fig4_module_ablation(df):
    """Horizontal bar chart of module leave-one-out results."""
    loo = df[df["experiment"] == "ablations/module_loo"].copy()
    if loo.empty:
        print("  [SKIP] Fig 4: No module_loo data")
        return

    loo_agg = agg(loo, ["condition"])

    # Order: FULL first, then sorted by BLEU descending
    full_bleu = loo_agg[loo_agg["condition"] == "FULL"]["bleu_mean"].values[0]

    # Compute delta
    loo_agg["delta"] = loo_agg["bleu_mean"] - full_bleu

    # Sort by BLEU for visual clarity
    loo_agg = loo_agg.sort_values("bleu_mean", ascending=True)

    # Friendly labels
    label_map = {
        "FULL": "Full (all modules)",
        "BASE-ONLY": "Base only (M1)",
        "NO-FUTURE": "No future tense",
        "NO-PAST": "No past tense",
        "NO-NEGATION": "No negation",
        "NO-QUESTIONS": "No questions",
    }
    loo_agg["label"] = loo_agg["condition"].map(label_map).fillna(loo_agg["condition"])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    colors = [COLORS["structured"] if c == "FULL" else "#7FAEDC" for c in loo_agg["condition"]]
    bars = ax.barh(loo_agg["label"], loo_agg["bleu_mean"],
                   xerr=loo_agg["bleu_std"], capsize=3,
                   color=colors, edgecolor="white", linewidth=0.5)

    # Delta labels
    for i, (_, row) in enumerate(loo_agg.iterrows()):
        if row["condition"] != "FULL":
            delta_str = f"{row['delta']:+.1f}"
            ax.text(row["bleu_mean"] + row["bleu_std"] + 0.5, i, delta_str,
                    va="center", fontsize=9, color="#666")

    ax.set_xlabel("BLEU")
    ax.set_title("Module Ablation (Leave-One-Out)")
    ax.set_xlim(0, full_bleu + 5)

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig4_module_ablation.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 4: Module LOO ablation")


# ── Figure 5: Structure Matters (two panels) ────────────────────────────────

def fig5_structure_matters(df):
    """PAIRS-INTACT vs BROKEN (left) + Verb diversity (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left panel: Minimal pairs ──
    mp = df[df["experiment"] == "ablations/minimal_pairs"]
    if not mp.empty:
        mp_agg = agg(mp, ["condition"])
        intact = mp_agg[mp_agg["condition"] == "PAIRS-INTACT"].iloc[0]
        broken = mp_agg[mp_agg["condition"] == "PAIRS-BROKEN"].iloc[0]

        bars = ax1.bar(["Pairs Intact", "Pairs Broken"],
                       [intact["bleu_mean"], broken["bleu_mean"]],
                       yerr=[intact["bleu_std"], broken["bleu_std"]],
                       capsize=5, width=0.5,
                       color=[COLORS["structured"], COLORS["random"]],
                       edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, [intact["bleu_mean"], broken["bleu_mean"]]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

        ax1.set_ylabel("BLEU")
        ax1.set_title("Minimal-Pair Structure")
        ax1.set_ylim(0, 30)
        ax1.text(0.5, -0.12, "Same sentences, different pairings",
                 transform=ax1.transAxes, ha="center", fontsize=9,
                 fontstyle="italic", color="gray")

    # ── Right panel: Verb diversity ──
    verb = df[df["experiment"] == "ablations/verb"]
    if not verb.empty:
        verb_agg = agg(verb, ["condition"])

        # Map condition to number of verbs
        verb_map = {"1-VERB": 1, "10-VERBS": 10}
        # For 3-VERBS and 5-VERBS, average across subsets a/b/c
        for n in [3, 5]:
            prefix = f"{n}-VERBS"
            subs = verb_agg[verb_agg["condition"].str.startswith(prefix)]
            if not subs.empty:
                verb_map[prefix] = n

        points = []
        # 1-VERB
        v1 = verb_agg[verb_agg["condition"] == "1-VERB"]
        if not v1.empty:
            points.append((1, v1["bleu_mean"].values[0], v1["bleu_std"].values[0]))

        # 3-VERBS (average across a/b/c)
        v3 = verb[verb["condition"].str.startswith("3-VERBS")]
        if not v3.empty:
            v3_by_seed = v3.groupby("condition")["test_bleu"].mean()
            points.append((3, v3_by_seed.mean(), v3_by_seed.std()))

        # 5-VERBS (average across a/b/c)
        v5 = verb[verb["condition"].str.startswith("5-VERBS")]
        if not v5.empty:
            v5_by_seed = v5.groupby("condition")["test_bleu"].mean()
            points.append((5, v5_by_seed.mean(), v5_by_seed.std()))

        # 10-VERBS
        v10 = verb_agg[verb_agg["condition"] == "10-VERBS"]
        if not v10.empty:
            points.append((10, v10["bleu_mean"].values[0], v10["bleu_std"].values[0]))

        if points:
            pts = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]

            ax2.plot(xs, ys, "o-", color=COLORS["structured"],
                     linewidth=2, markersize=8)
            ax2.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=COLORS["structured"],
                         capsize=4, alpha=0.6)

            # Annotate the jump
            if len(pts) >= 2:
                ax2.annotate(f"{pts[-1][1]:.1f}",
                             xy=(pts[-1][0], pts[-1][1]),
                             xytext=(pts[-1][0] - 1.5, pts[-1][1] + 2),
                             fontsize=10, fontweight="bold",
                             color=COLORS["structured"])

            ax2.set_xlabel("Number of Verbs")
            ax2.set_ylabel("BLEU")
            ax2.set_title("Verb Lexical Diversity")
            ax2.set_xticks(xs)
            ax2.set_ylim(0, 30)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig5_structure_matters.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 5: Structure matters (pairs + verbs)")


# ── Figure 6: Additive Curve ────────────────────────────────────────────────

def fig6_additive_curve(df):
    """Adding structured data to a fixed 6K random base."""
    exp2 = df[df["experiment"] == "exp2"].copy()

    points = []

    # Pure random 6K
    r6k = exp2[exp2["condition"] == "RANDOM-6000"]
    if not r6k.empty:
        points.append({"structured_added": 0,
                        "bleu_mean": r6k["test_bleu"].mean(),
                        "bleu_std": r6k["test_bleu"].std()})

    # Additive conditions
    for cond, n_struct in [("RANDOM-6K_STRUCTURED-500", 500),
                            ("RANDOM-6K_STRUCTURED-1000", 1000),
                            ("RANDOM-6K_STRUCTURED-2000", 2000),
                            ("RANDOM-6K_STRUCTURED-4000", 4000)]:
        sub = exp2[exp2["condition"] == cond]
        if not sub.empty:
            points.append({"structured_added": n_struct,
                            "bleu_mean": sub["test_bleu"].mean(),
                            "bleu_std": sub["test_bleu"].std()})

    if len(points) < 2:
        print("  [SKIP] Fig 6: Not enough additive data")
        return

    pts = pd.DataFrame(points).sort_values("structured_added")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pts["structured_added"], pts["bleu_mean"], "^-",
            color=COLORS["combined"], linewidth=2, markersize=8)
    ax.fill_between(pts["structured_added"],
                    pts["bleu_mean"] - pts["bleu_std"],
                    pts["bleu_mean"] + pts["bleu_std"],
                    color=COLORS["combined"], alpha=0.15)

    # Annotate the base point
    ax.annotate(f"6K random only:\nBLEU {pts.iloc[0]['bleu_mean']:.1f}",
                xy=(0, pts.iloc[0]["bleu_mean"]),
                xytext=(800, pts.iloc[0]["bleu_mean"] + 5),
                fontsize=9, fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                color="gray")

    ax.set_xlabel("Structured Sentences Added to 6K Random Base")
    ax.set_ylabel("BLEU")
    ax.set_title("Additive Effect of Structured Data")
    ax.set_xticks(pts["structured_added"].values)
    ax.set_xticklabels([f"+{int(x/1000)}K" if x >= 1000 else f"+{int(x)}"
                         for x in pts["structured_added"]])

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig6_additive_curve.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 6: Additive curve")


# ── Figure 7 (bonus): Exp1 Main Results Bar Chart ───────────────────────────

def fig7_exp1_overview(df):
    """Bar chart of all Exp1 conditions for a quick visual summary."""
    exp1 = df[df["experiment"] == "exp1"].copy()
    if exp1.empty:
        print("  [SKIP] Fig 7: No exp1 data")
        return

    exp1_agg = agg(exp1, ["condition"])

    # Order logically
    order = ["RANDOM-4K", "RANDOM-10K", "STRUCTURED-2K", "STRUCTURED-4K-ONLY",
             "RANDOM-6K_STRUCTURED-4K", "RANDOM-10K_STRUCTURED-4K"]
    exp1_agg["sort_key"] = exp1_agg["condition"].apply(lambda c: order.index(c) if c in order else 99)
    exp1_agg = exp1_agg.sort_values("sort_key")

    labels = {
        "RANDOM-4K": "Random\n4K",
        "RANDOM-10K": "Random\n10K",
        "STRUCTURED-2K": "Structured\n2K",
        "STRUCTURED-4K-ONLY": "Structured\n4K",
        "RANDOM-6K_STRUCTURED-4K": "6K Rand\n+4K Struct",
        "RANDOM-10K_STRUCTURED-4K": "10K Rand\n+4K Struct",
    }

    color_map = {
        "RANDOM-4K": COLORS["random"],
        "RANDOM-10K": COLORS["random"],
        "STRUCTURED-2K": COLORS["structured"],
        "STRUCTURED-4K-ONLY": COLORS["structured"],
        "RANDOM-6K_STRUCTURED-4K": COLORS["combined"],
        "RANDOM-10K_STRUCTURED-4K": COLORS["combined"],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(exp1_agg))
    bars = ax.bar(x, exp1_agg["bleu_mean"],
                  yerr=exp1_agg["bleu_std"], capsize=4,
                  color=[color_map.get(c, "#999") for c in exp1_agg["condition"]],
                  edgecolor="white", linewidth=0.5, width=0.6)

    # Value labels
    for bar, (_, row) in zip(bars, exp1_agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row["bleu_std"] + 0.5,
                f"{row['bleu_mean']:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(c, c) for c in exp1_agg["condition"]], fontsize=9)
    ax.set_ylabel("BLEU")
    ax.set_title("Experiment 1: Data Composition vs. Quantity")
    ax.set_ylim(0, 30)

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig7_exp1_overview.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 7: Exp1 overview bar chart")


# ── Heatmap: (Random × Structured → BLEU) ────────────────────────────────────

def fig_heatmap_composition(df):
    """Heatmap showing BLEU as a function of random and structured data amounts.

    Data points come from:
      - exp2/STRUCTURED-{200..4000}: structured-only (random=0)
      - exp2/RANDOM-{200..10000}: random-only (structured=0)
      - exp2/REPLACE-R*_S*: replacement diagonal (random+structured=10K)
      - exp2/RANDOM-6K_STRUCTURED-{500..4000}: additive (random=6K)
      - exp1/RANDOM-10K_STRUCTURED-4K: (10K, 4K)
      - exp1/STRUCTURED-4K-ONLY: (0, 4K)
      - exp1/STRUCTURED-2K: (0, 2K)
      - exp1/RANDOM-4K: (4K, 0)
      - exp1/RANDOM-10K: (10K, 0)
      - exp1/RANDOM-6K_STRUCTURED-4K: (6K, 4K)
    """
    points = []  # list of (random_amt, structured_amt, bleu_mean)

    # Helper: collect a single cell from any experiment/condition
    def _add(sub_df, r_amt, s_amt):
        if not sub_df.empty:
            points.append((r_amt, s_amt, sub_df["test_bleu"].mean()))

    exp1 = df[df["experiment"] == "exp1"]
    exp2 = df[df["experiment"] == "exp2"]

    # Structured-only from exp2
    for size in [200, 500, 1000, 2000, 3000, 4000]:
        _add(exp2[exp2["condition"] == f"STRUCTURED-{size}"], 0, size)

    # Random-only from exp2
    for size in [200, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
        _add(exp2[exp2["condition"] == f"RANDOM-{size}"], size, 0)

    # Replacement diagonal from exp2 (total = 10K)
    for s_size in [500, 1000, 2000, 4000]:
        r_size = 10000 - s_size
        _add(exp2[exp2["condition"] == f"REPLACE-R{r_size}_S{s_size}"], r_size, s_size)

    # Additive from exp2 (6K random base)
    for s_size in [500, 1000, 2000, 4000]:
        _add(exp2[exp2["condition"] == f"RANDOM-6K_STRUCTURED-{s_size}"], 6000, s_size)

    # From exp1
    _add(exp1[exp1["condition"] == "RANDOM-4K"], 4000, 0)
    _add(exp1[exp1["condition"] == "RANDOM-10K"], 10000, 0)
    _add(exp1[exp1["condition"] == "STRUCTURED-2K"], 0, 2000)
    _add(exp1[exp1["condition"] == "STRUCTURED-4K-ONLY"], 0, 4000)
    _add(exp1[exp1["condition"] == "RANDOM-6K_STRUCTURED-4K"], 6000, 4000)
    _add(exp1[exp1["condition"] == "RANDOM-10K_STRUCTURED-4K"], 10000, 4000)

    if not points:
        print("  [SKIP] Heatmap: No composition data found")
        return

    # De-duplicate by (random, structured) — take max BLEU if multiple sources
    pts_df = pd.DataFrame(points, columns=["random", "structured", "bleu"])
    pts_df = pts_df.groupby(["random", "structured"])["bleu"].max().reset_index()

    # ── Build grid ──
    r_vals = sorted(pts_df["random"].unique())
    s_vals = sorted(pts_df["structured"].unique())

    grid = np.full((len(s_vals), len(r_vals)), np.nan)
    r_idx = {v: i for i, v in enumerate(r_vals)}
    s_idx = {v: i for i, v in enumerate(s_vals)}

    for _, row in pts_df.iterrows():
        grid[s_idx[row["structured"]], r_idx[row["random"]]] = row["bleu"]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")  # gray for empty cells

    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower",
                   vmin=0, vmax=max(pts_df["bleu"].max(), 25))

    # Axis labels
    r_labels = [f"{int(v/1000)}K" if v >= 1000 else str(int(v)) for v in r_vals]
    s_labels = [f"{int(v/1000)}K" if v >= 1000 else str(int(v)) for v in s_vals]
    ax.set_xticks(range(len(r_vals)))
    ax.set_xticklabels(r_labels, fontsize=9)
    ax.set_yticks(range(len(s_vals)))
    ax.set_yticklabels(s_labels, fontsize=9)
    ax.set_xlabel("Random Sentences")
    ax.set_ylabel("Structured Sentences")
    ax.set_title("BLEU by Data Composition")

    # Annotate cells with values
    best_bleu = pts_df["bleu"].max()
    for _, row in pts_df.iterrows():
        ri = r_idx[row["random"]]
        si = s_idx[row["structured"]]
        bleu = row["bleu"]
        # White text on dark cells, black on light
        text_color = "white" if bleu > best_bleu * 0.65 else "black"
        fontweight = "bold" if bleu == best_bleu else "normal"
        ax.text(ri, si, f"{bleu:.1f}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight=fontweight)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("BLEU", fontsize=10)

    # Mark the optimal cell
    best_row = pts_df.loc[pts_df["bleu"].idxmax()]
    ri = r_idx[best_row["random"]]
    si = s_idx[best_row["structured"]]
    ax.plot(ri, si, "s", markersize=20, markeredgecolor=COLORS["accent"],
            markerfacecolor="none", markeredgewidth=2.5)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig_heatmap_composition.{fmt}"))
    plt.close(fig)
    print("  [OK] Heatmap: Data composition (Random × Structured → BLEU)")


# ── Figure 8: Baselines comparison ──────────────────────────────────────────

def fig8_baselines(df):
    """Compare baselines vs random 10K vs structured conditions, including combos."""
    baselines = df[df["experiment"] == "baselines"].copy()
    exp1 = df[df["experiment"] == "exp1"].copy()

    if baselines.empty:
        print("  [SKIP] Fig 8: No baselines data")
        return

    # Combine relevant conditions
    ref_conds = exp1[exp1["condition"].isin(["RANDOM-10K", "STRUCTURED-2K"])]
    combined = pd.concat([baselines, ref_conds])
    combined_agg = agg(combined, ["condition"])

    # Standalone + combo order
    order = ["RANDOM-10K", "TF-IDF-DIVERSE", "LENGTH-STRATIFIED",
             "VOCAB-MAXIMIZED", "STRUCTURED-2K",
             "STRUCT4K-TFIDF2K", "STRUCT4K-LENGTH2K",
             "STRUCT4K-VOCAB2K", "STRUCT4K-ALL-BASELINES"]
    combined_agg["sort_key"] = combined_agg["condition"].apply(
        lambda c: order.index(c) if c in order else 99)
    combined_agg = combined_agg.sort_values("sort_key")
    # Drop conditions not in our order
    combined_agg = combined_agg[combined_agg["sort_key"] < 99]

    color_map = {
        "RANDOM-10K": COLORS["random"],
        "TF-IDF-DIVERSE": COLORS["baseline"],
        "LENGTH-STRATIFIED": COLORS["baseline"],
        "VOCAB-MAXIMIZED": COLORS["baseline"],
        "STRUCTURED-2K": COLORS["structured"],
        "STRUCT4K-TFIDF2K": COLORS["combined"],
        "STRUCT4K-LENGTH2K": COLORS["combined"],
        "STRUCT4K-VOCAB2K": COLORS["combined"],
        "STRUCT4K-ALL-BASELINES": COLORS["combined"],
    }

    labels = {
        "RANDOM-10K": "Random\n10K",
        "TF-IDF-DIVERSE": "TF-IDF\n2K",
        "LENGTH-STRATIFIED": "Length\n2K",
        "VOCAB-MAXIMIZED": "Vocab\n2K",
        "STRUCTURED-2K": "Struct.\n2K",
        "STRUCT4K-TFIDF2K": "S4K +\nTF-IDF",
        "STRUCT4K-LENGTH2K": "S4K +\nLength",
        "STRUCT4K-VOCAB2K": "S4K +\nVocab",
        "STRUCT4K-ALL-BASELINES": "S4K +\nAll",
    }

    n_bars = len(combined_agg)
    fig, ax = plt.subplots(figsize=(max(8, n_bars * 1.1), 4.5))
    x = np.arange(n_bars)
    bars = ax.bar(x, combined_agg["bleu_mean"],
                  yerr=combined_agg["bleu_std"], capsize=3,
                  color=[color_map.get(c, "#999") for c in combined_agg["condition"]],
                  edgecolor="white", linewidth=0.5, width=0.6)

    for bar, (_, row) in zip(bars, combined_agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["bleu_std"] + 0.4,
                f"{row['bleu_mean']:.1f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(c, c) for c in combined_agg["condition"]],
                       fontsize=8, ha="center")
    ax.set_ylabel("BLEU")
    ax.set_title("Smart-Selection Baselines and Structured Combinations")

    # Dynamic ylim based on data
    max_val = (combined_agg["bleu_mean"] + combined_agg["bleu_std"]).max()
    ax.set_ylim(0, max_val * 1.18)

    # Vertical separator between standalone and combo groups
    combo_conds = {"STRUCT4K-TFIDF2K", "STRUCT4K-LENGTH2K",
                   "STRUCT4K-VOCAB2K", "STRUCT4K-ALL-BASELINES"}
    cond_list = list(combined_agg["condition"])
    for i, c in enumerate(cond_list):
        if c in combo_conds and i > 0 and cond_list[i - 1] not in combo_conds:
            ax.axvline(x=i - 0.5, color="#cccccc", linestyle="--", linewidth=1)
            ax.text(i - 0.5, max_val * 1.12, "← standalone | combos →",
                    ha="center", fontsize=7, color="#888888")
            break

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["random"], label="Random"),
        Patch(facecolor=COLORS["baseline"], label="Smart Selection (2K)"),
        Patch(facecolor=COLORS["structured"], label="Structured (2K)"),
        Patch(facecolor=COLORS["combined"], label="Struct 4K + Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig8_baselines.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 8: Baselines comparison")


# ── Figure 9: Additive Module Ablation ────────────────────────────────────────

def fig9_additive_modules(df):
    """Bar chart showing BLEU as modules are added incrementally (M1 → M1-M5)."""
    add = df[df["experiment"] == "ablations/additive"].copy()
    loo = df[df["experiment"] == "ablations/module_loo"].copy()

    if add.empty:
        print("  [SKIP] Fig 9: No additive ablation data")
        return

    # Get BASE-ONLY and FULL from module_loo
    base = loo[loo["condition"] == "BASE-ONLY"]
    full = loo[loo["condition"] == "FULL"]

    # Build combined df
    rows = []
    if not base.empty:
        rows.append(base.assign(condition="BASE-ONLY"))
    rows.append(add[add["condition"] == "ADD-M1M2"])
    rows.append(add[add["condition"] == "ADD-M1M2M3"])
    rows.append(add[add["condition"] == "ADD-M1M2M3M4"])
    if not full.empty:
        rows.append(full.assign(condition="FULL"))

    combined = pd.concat(rows)
    combined_agg = agg(combined, ["condition"])

    order = ["BASE-ONLY", "ADD-M1M2", "ADD-M1M2M3", "ADD-M1M2M3M4", "FULL"]
    combined_agg["sort_key"] = combined_agg["condition"].apply(
        lambda c: order.index(c) if c in order else 99)
    combined_agg = combined_agg.sort_values("sort_key")

    labels = {
        "BASE-ONLY": "M1\n(Base)",
        "ADD-M1M2": "M1+M2\n(+Negation)",
        "ADD-M1M2M3": "M1–M3\n(+Past)",
        "ADD-M1M2M3M4": "M1–M4\n(+Future)",
        "FULL": "M1–M5\n(+Questions)",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(combined_agg))

    # Gradient colors from light to dark blue
    n = len(combined_agg)
    cmap = plt.cm.Blues
    bar_colors = [cmap(0.3 + 0.6 * i / (n - 1)) for i in range(n)]

    bars = ax.bar(x, combined_agg["bleu_mean"],
                  yerr=combined_agg["bleu_std"], capsize=4,
                  color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.6)

    # Value labels and delta annotations
    prev_bleu = None
    for i, (bar, (_, row)) in enumerate(zip(bars, combined_agg.iterrows())):
        bleu = row["bleu_mean"]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["bleu_std"] + 0.4,
                f"{bleu:.1f}", ha="center", fontsize=10, fontweight="bold")
        if prev_bleu is not None:
            delta = bleu - prev_bleu
            ax.annotate(f"+{delta:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bleu / 2),
                        ha="center", fontsize=8, color="#666666", fontstyle="italic")
        prev_bleu = bleu

    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(c, c) for c in combined_agg["condition"]], fontsize=9)
    ax.set_ylabel("BLEU")
    ax.set_title("Additive Module Ablation: Incremental Contribution of Each Module")
    ax.set_ylim(0, combined_agg["bleu_mean"].max() * 1.25)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig9_additive_modules.{fmt}"))
    plt.close(fig)
    print("  [OK] Fig 9: Additive module ablation")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run collect_hf_results.py first.")
        sys.exit(1)

    print(f"Loading results from {CSV_PATH}...")
    df = load_data()
    print(f"  {len(df)} rows, {df['experiment'].nunique()} experiments, "
          f"{df['condition'].nunique()} conditions\n")

    print("Generating figures...")
    fig2_data_efficiency(df)
    fig3_replacement_curve(df)
    fig4_module_ablation(df)
    fig5_structure_matters(df)
    fig6_additive_curve(df)
    fig7_exp1_overview(df)
    fig8_baselines(df)
    fig9_additive_modules(df)
    fig_heatmap_composition(df)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Formats: .pdf (for paper) + .png (for preview)")


if __name__ == "__main__":
    main()
