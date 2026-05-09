"""
compare_april2026_vs_acl.py — side-by-side: ACL paper baselines vs april-2026 sweep.

For every (model, baseline_condition → new_condition) mapping we know about,
print the paper's mean BLEU / chrF++ alongside the april-2026 mean and the
delta. Lets us see at a glance whether stacking the new ~9.4K rows on top of
the paper recipes actually helps each architecture.

Result file layout assumed:
  Paper / pre-april HPC results:
    experiments/results/{exp1|baselines}/{condition}/seed{seed}/test_metrics.json
    experiments/results/hpc_new/{model}/{exp1|baselines}/{condition}/seed{seed}/test_metrics.json
  April-2026 results (rsync'd from HPC):
    experiments/results/april2026/{model}/{april2026_runX}/{condition}/seed{seed}/test_metrics.json

Usage:
    python experiments/analysis/compare_april2026_vs_acl.py \
        --paper-root experiments/results \
        --paper-hpc-root experiments/results/hpc_new \
        --april-root experiments/results/april2026
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# (paper_experiment, paper_condition)  ⇄  (april_experiment, april_condition)
COMPARISONS = [
    (("baselines", "STRUCT4K-ALL-BASELINES"), ("april2026_runA", "STRUCT4K-ALL-BASELINES-PLUS-NEW")),
    (("exp1", "RANDOM-10K_STRUCTURED-4K"),    ("april2026_runA", "FULL")),
]

MODEL_LABELS = {
    "nllb-1.3b": "NLLB-1.3B",
    "nllb-600m": "NLLB-600M",
    "mbart-fr":  "mBART-50 (fr-init)",
    "mbart-rand": "mBART-50 (rand-init)",
}

METRIC_KEYS = ["test_bleu", "test_chrf", "test_chrfpp"]


def harvest(root: Path) -> dict:
    """Walk root collecting test_metrics.json keyed by (model, exp, cond, seed).

    Two layouts are supported:
      - {root}/{model}/{exp_path...}/{cond}/seed{N}/test_metrics.json   (HPC)
      - {root}/{exp_path...}/{cond}/seed{N}/test_metrics.json           (paper-local)
    """
    out: dict = {}
    if not root.exists():
        return out
    for f in root.rglob("test_metrics.json"):
        parts = f.relative_to(root).parts
        if len(parts) < 3:
            continue
        seed_dir = parts[-2]
        if not seed_dir.startswith("seed"):
            continue
        try:
            seed = int(seed_dir[4:])
        except ValueError:
            continue
        cond = parts[-3]
        upper = parts[:-3]  # everything before condition

        # Heuristic: if the first segment looks like a model_subdir, peel it off.
        if upper and upper[0] in MODEL_LABELS:
            model, exp_parts = upper[0], upper[1:]
        else:
            # Paper-local results don't include model in the path; tag as paper-default.
            model, exp_parts = "paper-local", upper
        exp = "/".join(exp_parts) if exp_parts else "(root)"

        with f.open() as fh:
            metrics = json.load(fh)
        out[(model, exp, cond, seed)] = metrics
    return out


def aggregate(entries: list[dict]) -> dict:
    summary = {"n": len(entries)}
    for k in METRIC_KEYS:
        vals = [e[k] for e in entries if k in e]
        if not vals:
            continue
        summary[f"{k}_mean"] = statistics.mean(vals)
        summary[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return summary


def fmt_pair(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None or std == 0:
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def collect_for_pair(
    results: dict,
    model: str,
    exp: str,
    cond: str,
) -> list[dict]:
    entries = [
        m for (mdl, e, c, _seed), m in results.items()
        if mdl == model and e == exp and c == cond
    ]
    # Paper-local results are model-agnostic in the path: include them iff the
    # caller is asking for the paper-default model bucket.
    if model == "paper-local":
        return entries
    # When asking for a specific model, also accept paper-local entries that
    # match the experiment+condition (the paper recorded only nllb-600m there).
    paper_local_matches = [
        m for (mdl, e, c, _seed), m in results.items()
        if mdl == "paper-local" and e == exp and c == cond
    ]
    if model == "nllb-600m" and not entries:
        return paper_local_matches
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--paper-root", type=Path, default=Path("experiments/results"))
    ap.add_argument("--paper-hpc-root", type=Path, default=Path("experiments/results/hpc_new"))
    ap.add_argument("--april-root", type=Path, default=Path("experiments/results/april2026"))
    ap.add_argument("--csv", type=Path, default=Path("experiments/results/summary/april2026_vs_acl.csv"))
    args = ap.parse_args()

    paper = harvest(args.paper_root)
    paper_hpc = harvest(args.paper_hpc_root)
    april = harvest(args.april_root)

    paper.update(paper_hpc)  # HPC entries shadow same-key paper entries
    print(f"Paper entries:       {len(paper):,}")
    print(f"April-2026 entries:  {len(april):,}")
    if not april:
        print("\nNo april-2026 results found yet. Run rsync from HPC first:")
        print("  rsync -av --include='*/test_metrics.json' --include='*/' --exclude='*' \\")
        print("    f006g5b@discovery.dartmouth.edu:"
              "<HPC_WORKDIR> \\")
        print("    experiments/results/april2026/")

    # Build the comparison table.
    rows: list[dict] = []
    models = ["nllb-1.3b", "nllb-600m", "mbart-fr", "mbart-rand"]
    for (paper_exp, paper_cond), (april_exp, april_cond) in COMPARISONS:
        for model in models:
            paper_entries = collect_for_pair(paper, model, paper_exp, paper_cond)
            april_entries = collect_for_pair(april, model, april_exp, april_cond)
            paper_agg = aggregate(paper_entries) if paper_entries else None
            april_agg = aggregate(april_entries) if april_entries else None
            row = {
                "model": MODEL_LABELS[model],
                "paper_recipe": f"{paper_exp}/{paper_cond}",
                "april_recipe": f"{april_exp}/{april_cond}",
                "paper_n": (paper_agg or {}).get("n", 0),
                "april_n": (april_agg or {}).get("n", 0),
            }
            for k in METRIC_KEYS:
                p_mean = (paper_agg or {}).get(f"{k}_mean")
                a_mean = (april_agg or {}).get(f"{k}_mean")
                row[f"{k}_paper"] = p_mean
                row[f"{k}_paper_std"] = (paper_agg or {}).get(f"{k}_std")
                row[f"{k}_april"] = a_mean
                row[f"{k}_april_std"] = (april_agg or {}).get(f"{k}_std")
                row[f"{k}_delta"] = (a_mean - p_mean) if (p_mean is not None and a_mean is not None) else None
            rows.append(row)

    # Console table.
    print()
    print("=" * 110)
    print(f"  {'Model':<22}{'Paper recipe':<42}{'BLEU paper→april (Δ)':<28}{'chrF++ Δ':<10}")
    print("=" * 110)
    for r in rows:
        bleu = (f"{fmt_pair(r['test_bleu_paper'], r['test_bleu_paper_std'])} → "
                f"{fmt_pair(r['test_bleu_april'], r['test_bleu_april_std'])}")
        delta = r["test_bleu_delta"]
        delta_str = "—" if delta is None else f"({delta:+.2f})"
        chrf_delta = r["test_chrfpp_delta"]
        chrf_str = "—" if chrf_delta is None else f"{chrf_delta:+.2f}"
        marker = "" if r["april_n"] else "  [no april data]"
        print(f"  {r['model']:<22}{r['paper_recipe']:<42}{bleu+' '+delta_str:<28}{chrf_str:<10}{marker}")
    print("=" * 110)

    # CSV.
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow([
            "model", "paper_recipe", "april_recipe", "paper_n", "april_n",
            *[f"{k}_{side}" for k in METRIC_KEYS for side in ("paper", "paper_std", "april", "april_std", "delta")],
        ])
        for r in rows:
            w.writerow([
                r["model"], r["paper_recipe"], r["april_recipe"], r["paper_n"], r["april_n"],
                *[
                    "" if r[f"{k}_{side}"] is None else f"{r[f'{k}_{side}']:.4f}"
                    for k in METRIC_KEYS
                    for side in ("paper", "paper_std", "april", "april_std", "delta")
                ],
            ])
    print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
