"""
length_stats_train.py — token-length distribution of TRAINING data per condition.

Companion to length_stats.py (which covers the test set). Addresses the train-set
half of Reviewer FtYD W3: "report sentence length statistics (in tokens) for both
training and test sets across corpora to ensure a fair comparison."

For each train.tsv under experiments/data/splits/exp1/{condition}/ plus the two
shared pools (random_train.tsv, structured_train.tsv), tokenizes the French source
and Adja target with the NLLB-200 tokenizer and reports:
  n, mean, median, stdev, min, max, p95, p99, n_over_128, pct_over_128

Usage:
    python length_stats_train.py        # auto-detect repo root, write CSV
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
EXP1_ROOT_REL = "experiments/data/splits/exp1"
SHARED_ROOT_REL = "experiments/data/splits/shared"
DEFAULT_OUTPUT = "experiments/results/summary/length_stats_train.csv"

REPO_ROOT_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent,
    Path.cwd(),
]


def find_repo_root() -> Path:
    for root in REPO_ROOT_CANDIDATES:
        if (root / EXP1_ROOT_REL).exists():
            return root
    raise FileNotFoundError(f"Could not locate {EXP1_ROOT_REL} from {REPO_ROOT_CANDIDATES}")


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def summarize(corpus: str, side: str, lens: list[int], max_length: int) -> dict:
    if not lens:
        return {"corpus": corpus, "side": side, "n": 0}
    sorted_lens = sorted(lens)
    over = sum(1 for n in lens if n > max_length)
    return {
        "corpus": corpus,
        "side": side,
        "n": len(lens),
        "mean": round(statistics.mean(lens), 2),
        "median": int(statistics.median(lens)),
        "stdev": round(statistics.stdev(lens), 2) if len(lens) > 1 else 0.0,
        "min": min(lens),
        "max": max(lens),
        "p95": sorted_lens[int(len(lens) * 0.95) - 1],
        "p99": sorted_lens[int(len(lens) * 0.99) - 1],
        f"n_over_{max_length}": over,
        f"pct_over_{max_length}": round(100 * over / len(lens), 2),
    }


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print()
    print("  " + "  ".join(f"{k:>{widths[k]}}" for k in keys))
    print("  " + "  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  " + "  ".join(f"{str(r.get(k, '')):>{widths[k]}}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    root = find_repo_root()
    exp1_root = root / EXP1_ROOT_REL
    shared_root = root / SHARED_ROOT_REL

    # Collect (corpus_label, file_path) pairs, in a stable order
    targets: list[tuple[str, Path]] = []
    for cond_dir in sorted(exp1_root.iterdir()):
        train_path = cond_dir / "train.tsv"
        if train_path.is_file():
            targets.append((f"exp1/{cond_dir.name}/train", train_path))
    for shared_name in ["random_train.tsv", "structured_train.tsv"]:
        p = shared_root / shared_name
        if p.is_file():
            targets.append((f"shared/{p.stem}", p))

    if not targets:
        print("No train.tsv files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}", file=sys.stderr)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: install transformers. `pip install transformers`", file=sys.stderr)
        sys.exit(1)
    tok = AutoTokenizer.from_pretrained(args.model)

    rows: list[dict] = []
    for corpus, path in targets:
        pairs = load_pairs(path)
        if not pairs:
            print(f"  WARN: empty {path}", file=sys.stderr)
            continue
        print(f"Tokenizing {corpus}: {len(pairs)} rows from {path}", file=sys.stderr)
        src_lens = [len(tok.encode(s, add_special_tokens=True)) for s, _ in pairs]
        tgt_lens = [len(tok.encode(t, add_special_tokens=True)) for _, t in pairs]
        rows.append(summarize(corpus, "src", src_lens, args.max_length))
        rows.append(summarize(corpus, "tgt", tgt_lens, args.max_length))

    print(f"\n=== Train-set token-length stats (tokenizer: {args.model}, MAX_LENGTH={args.max_length}) ===")
    print_summary(rows)

    out = Path(args.output_csv)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}", file=sys.stderr)

    # Quick rebuttal-ready summary
    print("\n=== Headline for FtYD W3 ===")
    total_n = sum(r["n"] for r in rows if r["side"] == "src")
    total_over = sum(r.get(f"n_over_{args.max_length}", 0) for r in rows)
    print(f"  Total train rows examined: {total_n} (across {len(targets)} train files, src+tgt sides)")
    print(f"  Total sentences exceeding MAX_LENGTH={args.max_length}: {total_over}")
    print(f"  Worst-case max token length across all train files: {max(r['max'] for r in rows)}")


if __name__ == "__main__":
    main()
