"""
compute_robustness_table.py — Aggregate HPC + Gemini results for appendix robustness table.

Scans:
  - experiments/results/hpc_new/{results_subdir}/{experiment}/{condition}/seed{seed}/test_metrics.json
  - experiments/results/gemini/exp1/{condition}/seed{seed}/test_metrics.json

Groups by (model_label, condition) and computes mean ± std over seeds for BLEU and chrF++.
Prints a table ready for copy-paste into the LaTeX appendix.

Usage:
    python compute_robustness_table.py [--hpc-dir PATH] [--output PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics


# ── Model key → display label ──
MODEL_LABELS = {
    "nllb-1.3b":  "NLLB-1.3B",
    "nllb-600m":  "NLLB-200-distilled-600M",
    "mbart-fr":   "mBART-50 (French init)",
    "mbart-rand": "mBART-50 (Random init)",
    "gemini-2.5-flash": "Gemini-2.5 (fine-tuned, seed 42)",
}

# ── Conditions to show in the appendix table (in display order) ──
FOCUS_CONDITIONS = [
    "RANDOM-10K",
    "STRUCTURED-4K-ONLY",
    "RANDOM-6K_STRUCTURED-4K",
    "RANDOM-10K_STRUCTURED-4K",
]

# ── Reference NLLB-600M numbers from main paper (mean across 5 seeds) ──
NLLB_600M_REF = {
    "RANDOM-10K":                 {"bleu": 4.13, "chrfpp": 27.56},
    "STRUCTURED-4K-ONLY":         {"bleu": 19.51, "chrfpp": 28.52},
    "RANDOM-6K_STRUCTURED-4K":    {"bleu": 21.42, "chrfpp": 38.58},
    "RANDOM-10K_STRUCTURED-4K":   {"bleu": 22.45, "chrfpp": 39.25},
}


def load_hpc_results(hpc_dir: Path) -> dict:
    """
    Load all test_metrics.json from hpc_dir.
    Expected path: {hpc_dir}/{results_subdir}/{experiment}/{condition}/seed{seed}/test_metrics.json
    Returns: {(results_subdir, experiment, condition, seed): metrics_dict}
    """
    results = {}
    for metrics_file in sorted(hpc_dir.rglob("test_metrics.json")):
        parts = metrics_file.relative_to(hpc_dir).parts
        # parts: (results_subdir, experiment, condition, seed_dir, test_metrics.json)
        # experiment can be multi-level: e.g. ablations/module_loo
        if len(parts) < 4:
            print(f"  SKIP (unexpected path depth): {metrics_file}", file=sys.stderr)
            continue
        seed_dir = parts[-2]          # e.g. "seed42"
        condition = parts[-3]         # e.g. "RANDOM-10K"
        results_subdir = parts[0]     # e.g. "nllb-1.3b"
        # experiment is everything between results_subdir and condition
        experiment = "/".join(parts[1:-3])
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            print(f"  SKIP (bad seed dir): {metrics_file}", file=sys.stderr)
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        results[(results_subdir, experiment, condition, seed)] = metrics
    return results


def load_gemini_results(gemini_dir: Path) -> dict:
    """
    Load Gemini exp1 results.
    Expected path: {gemini_dir}/exp1/{condition}/seed{seed}/test_metrics.json
    Returns: {(condition, seed): metrics_dict}
    """
    results = {}
    exp1_dir = gemini_dir / "exp1"
    if not exp1_dir.exists():
        return results
    for metrics_file in sorted(exp1_dir.rglob("test_metrics.json")):
        parts = metrics_file.relative_to(exp1_dir).parts
        if len(parts) < 3:
            continue
        seed_dir = parts[-2]
        condition = parts[-3]
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        results[(condition, seed)] = metrics
    return results


def aggregate(rows: list[dict], key_bleu="test_bleu", key_chrfpp="test_chrfpp") -> dict:
    """Compute mean and std for a list of metric dicts."""
    bleus  = [r[key_bleu]  for r in rows if key_bleu  in r]
    chrfpps = [r[key_chrfpp] for r in rows if key_chrfpp in r]
    result = {"n": len(bleus)}
    if bleus:
        result["bleu_mean"] = statistics.mean(bleus)
        result["bleu_std"]  = statistics.stdev(bleus) if len(bleus) > 1 else 0.0
    if chrfpps:
        result["chrfpp_mean"] = statistics.mean(chrfpps)
        result["chrfpp_std"]  = statistics.stdev(chrfpps) if len(chrfpps) > 1 else 0.0
    return result


def fmt(mean: float, std: float, n: int) -> str:
    """Format mean±std, appending seed count if < 5."""
    s = f"{mean:.1f}"
    if std > 0:
        s += f"±{std:.1f}"
    if n < 5:
        s += f" (n={n})"
    return s


def main():
    parser = argparse.ArgumentParser(description="Build robustness appendix table")
    parser.add_argument(
        "--hpc-dir",
        default="experiments/results/hpc_new",
        help="Local directory with rsync'd HPC results"
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Alias for --hpc-dir (clearer name for non-paper sweeps such as "
             "experiments/results/april2026/). Overrides --hpc-dir if given."
    )
    parser.add_argument(
        "--raw-dir",
        default="experiments/results/raw",
        help="Local directory with HF Jobs results (nllb-1.3b/exp1/...)"
    )
    parser.add_argument(
        "--gemini-dir",
        default="experiments/results/gemini",
        help="Local directory with Gemini results"
    )
    parser.add_argument(
        "--output",
        default="experiments/results/summary/robustness_table.csv",
        help="CSV output path"
    )
    parser.add_argument(
        "--experiment",
        default="exp1",
        help="Which experiment to focus on (default: exp1)"
    )
    parser.add_argument(
        "--include-conditions",
        default=None,
        help="Comma-separated condition names to include. Defaults to the "
             "paper's FOCUS_CONDITIONS. Pass 'auto' to use every condition "
             "discovered under --hpc-dir/--results-root."
    )
    parser.add_argument(
        "--include-models",
        default=None,
        help="Comma-separated results_subdir values to include "
             "(default: nllb-1.3b,mbart-fr,mbart-rand). Pass 'auto' to use "
             "every model directory found."
    )
    parser.add_argument(
        "--skip-paper-reference",
        action="store_true",
        help="Don't prepend the hard-coded NLLB-600M paper reference rows. "
             "Use this when aggregating non-exp1 sweeps such as april2026."
    )
    args = parser.parse_args()

    hpc_dir    = Path(args.results_root or args.hpc_dir)
    raw_dir    = Path(args.raw_dir)
    gemini_dir = Path(args.gemini_dir)
    output     = Path(args.output)
    exp        = args.experiment

    # ── Load HPC results ──
    if hpc_dir.exists():
        hpc_results = load_hpc_results(hpc_dir)
        print(f"Loaded {len(hpc_results)} HPC result entries from {hpc_dir}")
    else:
        hpc_results = {}
        print(f"WARNING: HPC dir not found: {hpc_dir} — run rsync first", file=sys.stderr)

    # ── Load raw results (HF Jobs, e.g. nllb-1.3b/exp1) ──
    if raw_dir.exists():
        raw_results = load_hpc_results(raw_dir)
        print(f"Loaded {len(raw_results)} raw result entries from {raw_dir}")
        for key, val in raw_results.items():
            if key not in hpc_results:
                hpc_results[key] = val
    else:
        print(f"INFO: Raw dir not found: {raw_dir}", file=sys.stderr)

    # ── Load Gemini results ──
    gemini_results = load_gemini_results(gemini_dir)
    print(f"Loaded {len(gemini_results)} Gemini result entries from {gemini_dir}")

    # ── Group HPC by (results_subdir, condition) for the target experiment ──
    grouped: dict[tuple, list] = defaultdict(list)
    for (results_subdir, experiment, condition, seed), metrics in hpc_results.items():
        if experiment == exp:
            grouped[(results_subdir, condition)].append(metrics)

    # ── Resolve which conditions / models to display ──
    if args.include_conditions == "auto":
        focus_conditions = sorted({cond for (_, cond) in grouped.keys()})
    elif args.include_conditions:
        focus_conditions = [c.strip() for c in args.include_conditions.split(",") if c.strip()]
    else:
        focus_conditions = list(FOCUS_CONDITIONS)

    if args.include_models == "auto":
        focus_models = sorted({sub for (sub, _) in grouped.keys()})
    elif args.include_models:
        focus_models = [m.strip() for m in args.include_models.split(",") if m.strip()]
    else:
        focus_models = ["nllb-1.3b", "mbart-fr", "mbart-rand"]

    # ── Build rows for the table ──
    # Each row: model_label, condition, bleu_mean, bleu_std, chrfpp_mean, chrfpp_std, n
    rows = []

    # Reference: NLLB-600M from main paper (skipped for non-paper sweeps).
    if not args.skip_paper_reference:
        for cond in focus_conditions:
            if cond in NLLB_600M_REF:
                ref = NLLB_600M_REF[cond]
                rows.append({
                    "model": "NLLB-600M (main paper, 5-seed mean)",
                    "condition": cond,
                    "bleu_mean": ref["bleu"], "bleu_std": None,
                    "chrfpp_mean": ref["chrfpp"], "chrfpp_std": None,
                    "n": 5
                })

    # HPC models
    for results_subdir in focus_models:
        label = MODEL_LABELS.get(results_subdir, results_subdir)
        for cond in focus_conditions:
            entries = grouped.get((results_subdir, cond), [])
            if not entries:
                continue
            agg = aggregate(entries)
            rows.append({
                "model": label,
                "condition": cond,
                "bleu_mean":   agg.get("bleu_mean"),
                "bleu_std":    agg.get("bleu_std"),
                "chrfpp_mean": agg.get("chrfpp_mean"),
                "chrfpp_std":  agg.get("chrfpp_std"),
                "n": agg["n"]
            })

    # Gemini (single seed per condition)
    for cond in FOCUS_CONDITIONS:
        if (cond, 42) in gemini_results:
            m = gemini_results[(cond, 42)]
            rows.append({
                "model": MODEL_LABELS["gemini-2.5-flash"],
                "condition": cond,
                "bleu_mean":   m.get("test_bleu"),
                "bleu_std":    None,
                "chrfpp_mean": m.get("test_chrfpp"),
                "chrfpp_std":  None,
                "n": 1
            })

    # ── Print table ──
    print()
    print("=" * 80)
    print(f"  ROBUSTNESS TABLE — experiment: {exp}")
    print("=" * 80)
    col_w = 38
    print(f"{'Model':<{col_w}} {'Condition':<35} {'BLEU':>12} {'chrF++':>12}")
    print("-" * 100)
    current_model = None
    for row in rows:
        if row["model"] != current_model:
            if current_model is not None:
                print()
            current_model = row["model"]
        bleu_str   = (fmt(row["bleu_mean"], row["bleu_std"] or 0.0, row["n"])
                      if row["bleu_mean"] is not None else "—")
        chrfpp_str = (fmt(row["chrfpp_mean"], row["chrfpp_std"] or 0.0, row["n"])
                      if row["chrfpp_mean"] is not None else "—")
        print(f"{row['model']:<{col_w}} {row['condition']:<35} {bleu_str:>12} {chrfpp_str:>12}")
    print("=" * 80)

    # ── Print LaTeX snippet ──
    print()
    print("── LaTeX table rows (copy into appendix) ──")
    print()
    current_model = None
    for row in rows:
        if row["model"] != current_model:
            if current_model is not None:
                print("\\midrule")
            current_model = row["model"]
            short = row["model"].split("(")[0].strip()
            print(f"% {row['model']}")
        bleu_str = (f"{row['bleu_mean']:.1f}" if row["bleu_mean"] is not None else "—")
        chrfpp_str = (f"{row['chrfpp_mean']:.1f}" if row["chrfpp_mean"] is not None else "—")
        cond_display = row["condition"].replace("_", "+").replace("-ONLY", "")
        print(f"{short} & {cond_display} & {bleu_str} & {chrfpp_str} \\\\")
    print()

    # ── Save CSV ──
    import csv as _csv
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = _csv.writer(f, quoting=_csv.QUOTE_ALL)
        writer.writerow(["model", "condition", "bleu_mean", "bleu_std",
                         "chrfpp_mean", "chrfpp_std", "n"])
        for row in rows:
            writer.writerow([
                row["model"],
                row["condition"],
                f"{row['bleu_mean']:.4f}"   if row["bleu_mean"]   is not None else "",
                f"{row['bleu_std']:.4f}"    if row["bleu_std"]    is not None else "",
                f"{row['chrfpp_mean']:.4f}" if row["chrfpp_mean"] is not None else "",
                f"{row['chrfpp_std']:.4f}"  if row["chrfpp_std"]  is not None else "",
                str(row["n"])
            ])
    print(f"Saved CSV → {output}")


if __name__ == "__main__":
    main()
