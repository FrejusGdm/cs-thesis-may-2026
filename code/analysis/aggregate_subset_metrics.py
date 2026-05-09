"""
aggregate_subset_metrics.py — roll up per-file subset BLEU/chrF into a summary table.

Reads the per-file CSV produced by recompute_subset_metrics.py
(experiments/results/summary/rebuttal_subset_bleu.csv by default) and emits:
  1. A summary CSV with one row per (model, condition, subset), mean ± std over seeds.
  2. A pretty stdout table grouped by condition (the format the user wants for the rebuttal).
  3. A LaTeX appendix snippet ready to paste into acl_latex.tex (tab:subset_bleu).

Usage:
    python aggregate_subset_metrics.py                                 # all defaults
    python aggregate_subset_metrics.py --input <csv>  --output <csv>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = "experiments/results/summary/rebuttal_subset_bleu.csv"
DEFAULT_OUT_CSV = "experiments/results/summary/rebuttal_subset_bleu_aggregated.csv"
DEFAULT_OUT_TEX = "experiments/results/summary/tab_subset_bleu.tex"

# Display order for conditions and subsets
CONDITION_ORDER = [
    "RANDOM-10K",
    "STRUCTURED-2K",
    "STRUCTURED-4K-ONLY",
    "RANDOM-4K",
    "RANDOM-6K_STRUCTURED-4K",
    "RANDOM-10K_STRUCTURED-4K",
]
SUBSET_ORDER = ["combined", "structured", "tatoeba"]
MODEL_ORDER = ["nllb-600m", "mbart-fr", "nllb-1.3b"]
MODEL_LATEX = {"nllb-600m": r"\textsc{NLLB-600M}", "mbart-fr": r"\textsc{mBART-fr}", "nllb-1.3b": r"\textsc{NLLB-1.3B}"}
CONDITION_LATEX = {
    "RANDOM-10K":               r"\textsc{Rand-10K}",
    "STRUCTURED-2K":            r"\textsc{Struct-2K}",
    "STRUCTURED-4K-ONLY":       r"\textsc{Struct-4K}",
    "RANDOM-4K":                r"\textsc{Rand-4K}",
    "RANDOM-6K_STRUCTURED-4K":  r"\textsc{R6K+S4K}",
    "RANDOM-10K_STRUCTURED-4K": r"\textsc{R10K+S4K}",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUT_CSV)
    ap.add_argument("--latex",  default=DEFAULT_OUT_TEX)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input CSV: {in_path}\nRun recompute_subset_metrics.py with --output first.")

    # Group by (model, condition, subset) -> [(seed, bleu, chrf, chrfpp), ...]
    groups: dict[tuple[str, str, str], list[tuple[str, float, float, float]]] = defaultdict(list)
    n_per_group: dict[tuple[str, str, str], int] = {}
    with open(in_path) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["condition"], row["subset"])
            try:
                groups[key].append((
                    row["seed"],
                    float(row["bleu"]),
                    float(row["chrf"]),
                    float(row["chrfpp"]),
                ))
                n_per_group[key] = int(row["n_samples"])
            except (ValueError, KeyError):
                pass

    # Build aggregated rows
    rows: list[dict] = []
    for (model, cond, subset), entries in groups.items():
        bleus  = [e[1] for e in entries]
        chrfs  = [e[2] for e in entries]
        chrfpps = [e[3] for e in entries]
        m_b, s_b = mean_std(bleus)
        m_c, s_c = mean_std(chrfs)
        m_p, s_p = mean_std(chrfpps)
        rows.append({
            "model":      model,
            "condition":  cond,
            "subset":     subset,
            "n_samples":  n_per_group.get((model, cond, subset), 0),
            "n_seeds":    len(entries),
            "bleu_mean":  round(m_b, 2),
            "bleu_std":   round(s_b, 2),
            "chrf_mean":  round(m_c, 2),
            "chrf_std":   round(s_c, 2),
            "chrfpp_mean": round(m_p, 2),
            "chrfpp_std": round(s_p, 2),
        })

    def sort_key(r):
        return (
            MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
            CONDITION_ORDER.index(r["condition"]) if r["condition"] in CONDITION_ORDER else 99,
            SUBSET_ORDER.index(r["subset"]) if r["subset"] in SUBSET_ORDER else 99,
        )
    rows.sort(key=sort_key)

    # Write summary CSV
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "condition", "subset", "n_samples", "n_seeds",
                  "bleu_mean", "bleu_std", "chrf_mean", "chrf_std", "chrfpp_mean", "chrfpp_std"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    # ---- Pretty stdout: block-per-condition format, mean ± std over seeds ----
    # Reorganize rows so all subsets of one (model, condition) print together.
    by_cond: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[(r["model"], r["condition"])].append(r)
    # Stable ordering: model first, then condition, then subset.
    keys_sorted = sorted(by_cond.keys(), key=lambda mc: (
        MODEL_ORDER.index(mc[0]) if mc[0] in MODEL_ORDER else 99,
        CONDITION_ORDER.index(mc[1]) if mc[1] in CONDITION_ORDER else 99,
    ))

    print()
    print("=" * 86)
    print(f"  REBUTTAL SUBSET BLEU — paper-era data only (no new6k), 5 seeds, mean ± std")
    print("=" * 86)
    for model, cond in keys_sorted:
        block = by_cond[(model, cond)]
        n_seeds = max(r["n_seeds"] for r in block)
        seed_note = f"  (n_seeds={n_seeds})" if n_seeds > 1 else "  (n_seeds=1, smoke)"
        print()
        print(f"{model} / exp1 / {cond}{seed_note}")
        # Order subsets: combined / structured / tatoeba
        for subset_name in SUBSET_ORDER:
            sub = next((r for r in block if r["subset"] == subset_name), None)
            if sub is None:
                continue
            bleu = f"{sub['bleu_mean']:>5.2f}±{sub['bleu_std']:<4.2f}"
            chrf = f"{sub['chrf_mean']:>5.2f}±{sub['chrf_std']:<4.2f}"
            chrp = f"{sub['chrfpp_mean']:>5.2f}±{sub['chrfpp_std']:<4.2f}"
            print(f"  {subset_name:<11} n={sub['n_samples']:>4}   "
                  f"BLEU {bleu}   chrF {chrf}   chrF++ {chrp}")
    print()
    print("=" * 86)

    # ---- LaTeX snippet (NLLB-600M only — primary table; mBART-fr secondary) ----
    out_tex = Path(args.latex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"% Auto-generated by experiments/analysis/aggregate_subset_metrics.py")
    lines.append(r"% Drop into acl_latex.tex appendix as tab:subset_bleu")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Subset-level BLEU / chrF / chrF++ on the paper test set "
                 r"(455 structured + 1{,}000 Tatoeba sentences). All numbers are "
                 r"mean$\pm$std over 5 seeds. Trained on paper-era data only "
                 r"(no new6k contamination); per-sentence predictions saved as JSONL "
                 r"and re-scored offline by line index.}")
    lines.append(r"\label{tab:subset_bleu}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Subset & $n$ & BLEU $\uparrow$ & chrF $\uparrow$ & chrF++ $\uparrow$ \\")

    for model in MODEL_ORDER:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{MODEL_LATEX.get(model, model)}}}}} \\")
        lines.append(r"\midrule")
        last_cond = None
        for r in model_rows:
            cond_label = CONDITION_LATEX.get(r["condition"], r["condition"]).replace("_", r"\_")
            cond_cell = cond_label if r["condition"] != last_cond else ""
            last_cond = r["condition"]
            bleu = f"{r['bleu_mean']:.1f}$\\pm${r['bleu_std']:.1f}"
            chrf = f"{r['chrf_mean']:.1f}$\\pm${r['chrf_std']:.1f}"
            chrp = f"{r['chrfpp_mean']:.1f}$\\pm${r['chrfpp_std']:.1f}"
            lines.append(f"{cond_cell} & {r['subset']} & {r['n_samples']} & {bleu} & {chrf} & {chrp} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"\nWrote LaTeX snippet → {out_tex}")


if __name__ == "__main__":
    main()
