#!/usr/bin/env python3
"""
compute_chrfpp_for_tables.py

Reads all_results.csv and prints LaTeX-ready table rows with chrF++ values
for the four tables currently missing this column:
  - tab:scaling
  - tab:module_loo
  - tab:structure_matters
  - tab:pronoun

Also prints updated values for existing tables to verify consistency.

Usage:
    python experiments/analysis/compute_chrfpp_for_tables.py
"""
import math
import pandas as pd
from pathlib import Path

RESULTS_CSV = Path(__file__).parent.parent / "results" / "summary" / "all_results.csv"


def mean_std(values):
    """Return (mean, std) for a list of floats, rounded to 1 decimal."""
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    if n == 1:
        return round(m, 1), None
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    s = math.sqrt(var)
    return round(m, 1), round(s, 1)


def fmt(mean, std, bold=False):
    """Format mean ± std for LaTeX."""
    if std is None:
        v = f"${mean}$"
    else:
        v = f"${mean} \\pm {std}$"
    if bold:
        v = f"$\\mathbf{{{mean} \\pm {std}}}$" if std else f"$\\mathbf{{{mean}}}$"
    return v


def fmt_delta(base_mean, cond_mean):
    """Format delta from base condition."""
    delta = round(cond_mean - base_mean, 1)
    sign = "+" if delta >= 0 else ""
    return f"${sign}{delta}$"


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    return df


def group_stats(df, experiment, condition, metric="test_chrfpp"):
    """Get mean and std for a specific experiment/condition/metric."""
    rows = df[(df["experiment"] == experiment) & (df["condition"] == condition)]
    values = rows[metric].dropna().tolist()
    return mean_std(values)


# ---------------------------------------------------------------------------
# TABLE: tab:scaling
# (structured vs. random data, by size)
# Current columns: Size | BLEU (struct) | chrF (struct) | BLEU (rand) | chrF (rand)
# New columns:     Size | BLEU | chrF | chrF++ | BLEU | chrF | chrF++
# ---------------------------------------------------------------------------

def print_scaling_table(df):
    print("\n" + "=" * 70)
    print("TAB:SCALING — Scaling comparison (add chrF++ columns)")
    print("=" * 70)
    print()
    print(r"\begin{tabular}{rccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{Structured}} & \multicolumn{3}{c}{\textbf{Random}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"\textbf{Size} & \textbf{BLEU} & \textbf{chrF} & \textbf{chrF++} & \textbf{BLEU} & \textbf{chrF} & \textbf{chrF++} \\")
    print(r"\midrule")

    structured_sizes = [
        ("200", "exp2", "STRUCTURED-200"),
        ("500", "exp2", "STRUCTURED-500"),
        ("1K", "exp2", "STRUCTURED-1000"),
        ("2K", "exp2", "STRUCTURED-2000"),
        ("4K", "exp2", "STRUCTURED-4000"),
    ]
    random_sizes = [
        ("200", "exp2", "RANDOM-200"),
        ("500", "exp2", "RANDOM-500"),
        ("1K", "exp2", "RANDOM-1000"),
        ("2K", "exp2", "RANDOM-2000"),
        ("4K", "exp2", "RANDOM-4000"),
        ("10K", "exp2", "RANDOM-10000"),
    ]

    # Build lookup for each size
    struct_data = {}
    for size_label, exp, cond in structured_sizes:
        bleu_m, bleu_s = group_stats(df, exp, cond, "test_bleu")
        chrf_m, chrf_s = group_stats(df, exp, cond, "test_chrf")
        chrfpp_m, chrfpp_s = group_stats(df, exp, cond, "test_chrfpp")
        struct_data[size_label] = (bleu_m, bleu_s, chrf_m, chrf_s, chrfpp_m, chrfpp_s)

    rand_data = {}
    for size_label, exp, cond in random_sizes:
        bleu_m, bleu_s = group_stats(df, exp, cond, "test_bleu")
        chrf_m, chrf_s = group_stats(df, exp, cond, "test_chrf")
        chrfpp_m, chrfpp_s = group_stats(df, exp, cond, "test_chrfpp")
        rand_data[size_label] = (bleu_m, bleu_s, chrf_m, chrf_s, chrfpp_m, chrfpp_s)

    all_sizes = ["200", "500", "1K", "2K", "4K", "10K"]
    for size in all_sizes:
        if size in struct_data:
            sb, sbs, sc, scs, scpp, scpps = struct_data[size]
            s_part = f"{fmt(sb, sbs)} & {fmt(sc, scs)} & {fmt(scpp, scpps)}"
        else:
            s_part = "--- & --- & ---"

        if size in rand_data:
            rb, rbs, rc, rcs, rcpp, rcpps = rand_data[size]
            r_part = f"{fmt(rb, rbs)} & {fmt(rc, rcs)} & {fmt(rcpp, rcpps)}"
        else:
            r_part = "--- & --- & ---"

        print(f"{size:6s}  & {s_part} & {r_part} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# TABLE: tab:module_loo
# Module leave-one-out ablation
# Current columns: Condition | Train | BLEU | Δ | chrF
# New columns:     Condition | Train | BLEU | Δ | chrF | chrF++
# ---------------------------------------------------------------------------

def print_module_loo_table(df):
    print("\n" + "=" * 70)
    print("TAB:MODULE_LOO — Leave-one-out ablation (add chrF++ column)")
    print("=" * 70)

    exp = "ablations/module_loo"
    conditions = [
        (r"\textsc{Full}", "FULL", 3823),
        (r"\textsc{No-Past}", "NO-PAST", 3010),
        (r"\textsc{No-Negation}", "NO-NEGATION", 3024),
        (r"\textsc{No-Questions}", "NO-QUESTIONS", 3240),
        (r"\textsc{No-Future}", "NO-FUTURE", 3009),
        (r"\textsc{Base-Only}", "BASE-ONLY", 815),
    ]

    full_bleu_mean, _ = group_stats(df, exp, "FULL", "test_bleu")

    print()
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Condition} & \textbf{Train} & \textbf{BLEU}$\uparrow$ & $\Delta$ & \textbf{chrF}$\uparrow$ & \textbf{chrF++}$\uparrow$ \\")
    print(r"\midrule")

    for label, cond, train in conditions:
        bleu_m, bleu_s = group_stats(df, exp, cond, "test_bleu")
        chrf_m, chrf_s = group_stats(df, exp, cond, "test_chrf")
        chrfpp_m, chrfpp_s = group_stats(df, exp, cond, "test_chrfpp")

        if bleu_m is None:
            print(f"  [WARNING] No data for {cond}")
            continue

        is_full = (cond == "FULL")
        bleu_str = fmt(bleu_m, bleu_s, bold=is_full)
        chrf_str = fmt(chrf_m, chrf_s, bold=is_full)
        chrfpp_str = fmt(chrfpp_m, chrfpp_s, bold=is_full)

        if is_full:
            delta_str = "---"
        else:
            delta_str = fmt_delta(full_bleu_mean, bleu_m)

        n_seeds = len(df[(df["experiment"] == exp) & (df["condition"] == cond)])
        seed_note = f"  % n={n_seeds}"
        print(f"{label:25s} & {train:>5} & {bleu_str} & {delta_str} & {chrf_str} & {chrfpp_str} \\\\{seed_note}")

    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# TABLE: tab:structure_matters
# Minimal-pair + verb diversity ablation
# Current columns: Condition | Train | BLEU | chrF
# New columns:     Condition | Train | BLEU | chrF | chrF++
# ---------------------------------------------------------------------------

def print_structure_matters_table(df):
    print("\n" + "=" * 70)
    print("TAB:STRUCTURE_MATTERS — Structure ablations (add chrF++ column)")
    print("=" * 70)

    print()
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"\textbf{Condition} & \textbf{Train} & \textbf{BLEU}$\uparrow$ & \textbf{chrF}$\uparrow$ & \textbf{chrF++}$\uparrow$ \\")
    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{Minimal-pair structure (same sentences)}} \\")

    # Minimal pairs section
    mp_exp = "ablations/minimal_pairs"
    for label, cond in [
        (r"\textsc{Pairs-Intact}", "PAIRS-INTACT"),
        (r"\textsc{Pairs-Broken}", "PAIRS-BROKEN"),
    ]:
        bleu_m, bleu_s = group_stats(df, mp_exp, cond, "test_bleu")
        chrf_m, chrf_s = group_stats(df, mp_exp, cond, "test_chrf")
        chrfpp_m, chrfpp_s = group_stats(df, mp_exp, cond, "test_chrfpp")
        train = 3823
        bold = (cond == "PAIRS-INTACT")
        print(f"{label:25s} & {train} & {fmt(bleu_m, bleu_s, bold)} & {fmt(chrf_m, chrf_s, bold)} & {fmt(chrfpp_m, chrfpp_s, bold)} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{Verb lexical diversity}} \\")

    # Verb diversity section — aggregate across multiple runs for same n-verbs
    verb_exp = "ablations/verb"
    verb_conditions = [
        ("1 verb", ["1-VERB"], 216),
        ("3 verbs", ["3-VERBS-a", "3-VERBS-b", "3-VERBS-c"], None),  # ~600
        ("5 verbs", ["5-VERBS-a", "5-VERBS-b", "5-VERBS-c"], None),  # ~950
        ("10 verbs", ["10-VERBS"], 3823),
    ]

    for label, conds, train_size in verb_conditions:
        # Aggregate across multiple conditions for same n-verbs
        all_rows = df[(df["experiment"] == verb_exp) & (df["condition"].isin(conds))]
        bleu_vals = all_rows["test_bleu"].dropna().tolist()
        chrf_vals = all_rows["test_chrf"].dropna().tolist()
        chrfpp_vals = all_rows["test_chrfpp"].dropna().tolist()

        bleu_m, bleu_s = mean_std(bleu_vals)
        chrf_m, chrf_s = mean_std(chrf_vals)
        chrfpp_m, chrfpp_s = mean_std(chrfpp_vals)

        if train_size is None:
            # Compute mean train size
            train_str = r"${\sim}$" + str(round(all_rows["train_size"].mean()))
        else:
            train_str = str(train_size)

        bold = (label == "10 verbs")
        print(f"{label:10s} & {train_str} & {fmt(bleu_m, bleu_s, bold)} & {fmt(chrf_m, chrf_s, bold)} & {fmt(chrfpp_m, chrfpp_s, bold)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# TABLE: tab:pronoun
# Pronoun coverage ablation
# Current columns: Condition | Pronouns | BLEU | chrF
# New columns:     Condition | Pronouns | BLEU | chrF | chrF++
# ---------------------------------------------------------------------------

def print_pronoun_table(df):
    print("\n" + "=" * 70)
    print("TAB:PRONOUN — Pronoun coverage ablation (add chrF++ column)")
    print("=" * 70)

    exp = "ablations/pronoun"
    conditions = [
        (r"\textsc{All-8}", "ALL-8", r"je, tu, il, elle, nous, vous, ils, elles"),
        (r"\textsc{Reduced-4}", "REDUCED-4", r"je, tu, il, nous"),
        (r"\textsc{Singular-3}", "SINGULAR-3", r"je, tu, il"),
        (r"\textsc{Minimal-1}", "MINIMAL-1", r"je"),
    ]

    print()
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"\textbf{Condition} & \textbf{Pronouns} & \textbf{BLEU}$\uparrow$ & \textbf{chrF}$\uparrow$ & \textbf{chrF++}$\uparrow$ \\")
    print(r"\midrule")

    for label, cond, pronouns in conditions:
        bleu_m, bleu_s = group_stats(df, exp, cond, "test_bleu")
        chrf_m, chrf_s = group_stats(df, exp, cond, "test_chrf")
        chrfpp_m, chrfpp_s = group_stats(df, exp, cond, "test_chrfpp")
        bold = (cond == "ALL-8")
        print(f"{label:20s} & {pronouns:45s} & {fmt(bleu_m, bleu_s, bold)} & {fmt(chrf_m, chrf_s, bold)} & {fmt(chrfpp_m, chrfpp_s, bold)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# BONUS: tab:module_size_ctrl — verify NO-PAST-1K single seed situation
# ---------------------------------------------------------------------------

def check_module_size_ctrl(df):
    print("\n" + "=" * 70)
    print("TAB:MODULE_SIZE_CTRL — Size-controlled ablation (check NO-PAST-1K)")
    print("=" * 70)

    exp = "ablations/module_size_ctrl"
    conditions = ["FULL-1K", "NO-PAST-1K", "NO-QUEST-1K", "NO-NEG-1K", "NO-FUT-1K", "BASE-1K"]

    print()
    for cond in conditions:
        rows = df[(df["experiment"] == exp) & (df["condition"] == cond)]
        seeds = rows["seed"].tolist()
        bleu_vals = rows["test_bleu"].dropna().tolist()
        chrfpp_vals = rows["test_chrfpp"].dropna().tolist()
        bleu_m, bleu_s = mean_std(bleu_vals)
        chrfpp_m, chrfpp_s = mean_std(chrfpp_vals)
        print(f"  {cond:15s}: n={len(seeds)} seeds={seeds} | BLEU={fmt(bleu_m, bleu_s)} | chrF++={fmt(chrfpp_m, chrfpp_s)}")

    print()
    print("  >> NO-PAST-1K has only 1 seed (456). Need to run seeds 42 and 123.")
    print("  >> See HPC job submission instructions in latex_experiments/2026-03-06_professor-feedback.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading: {RESULTS_CSV}")
    df = load_data()
    print(f"Loaded {len(df)} rows from all_results.csv")

    check_module_size_ctrl(df)
    print_scaling_table(df)
    print_module_loo_table(df)
    print_structure_matters_table(df)
    print_pronoun_table(df)

    print("\n" + "=" * 70)
    print("DONE. Copy the LaTeX rows above into acl_latex.tex.")
    print("See latex_experiments/2026-03-06_professor-feedback.md for context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
