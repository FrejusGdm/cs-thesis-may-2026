"""
length_stats.py — token-length distribution of test.tsv per subset (structured / Tatoeba).

Addresses Reviewer FtYD W3: "the authors use a maximum sequence length of 128 tokens.
While sentences in the generated corpus are relatively short, this constraint may
disadvantage the Tatoeba corpus if longer sentences are truncated."

Loads the NLLB-200 tokenizer (the model used for the strongest paper conditions),
tokenizes the French source and Adja target for all 1,455 test rows, and reports
length distribution + truncation rate (>128 tokens) by subset.

Test set layout: idx 0..454 = structured (455 rows), idx 455..1454 = Tatoeba (1000 rows).

Usage:
    python length_stats.py                 # prints summary, writes CSV + PDF histogram
    python length_stats.py --no-figure     # skip the PDF (matplotlib not required)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

# Default paths — robust to running from repo root or from this directory.
REPO_ROOT_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent,
    Path.cwd(),
]
TEST_TSV = "experiments/data/splits/shared/test.tsv"
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
STRUCTURED_END_IDX_EXCLUSIVE = 455


def find_test_tsv() -> Path:
    for root in REPO_ROOT_CANDIDATES:
        p = root / TEST_TSV
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate {TEST_TSV} from {REPO_ROOT_CANDIDATES}")


def load_test(path: Path) -> list[tuple[str, str]]:
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


def summarize(label: str, lens: list[int], max_length: int) -> dict:
    if not lens:
        return {"label": label, "n": 0}
    over = [n for n in lens if n > max_length]
    return {
        "label": label,
        "n": len(lens),
        "mean": round(statistics.mean(lens), 2),
        "median": statistics.median(lens),
        "stdev": round(statistics.stdev(lens), 2) if len(lens) > 1 else 0.0,
        "min": min(lens),
        "max": max(lens),
        "p95": sorted(lens)[int(len(lens) * 0.95) - 1],
        "p99": sorted(lens)[int(len(lens) * 0.99) - 1],
        f"n_over_{max_length}": len(over),
        f"pct_over_{max_length}": round(100 * len(over) / len(lens), 2),
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
    ap.add_argument("--test", default=None, help="Path to test.tsv (auto-detected by default)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model for tokenization (default: {DEFAULT_MODEL})")
    ap.add_argument("--max-length", type=int, default=128, help="Training MAX_LENGTH to compare against (default: 128)")
    ap.add_argument("--output-csv", default="experiments/results/summary/length_stats.csv")
    ap.add_argument("--output-figure", default="experiments/results/figures/length_distribution.pdf")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    test_path = Path(args.test) if args.test else find_test_tsv()
    print(f"Loading test set: {test_path}", file=sys.stderr)
    pairs = load_test(test_path)
    print(f"  rows: {len(pairs)}", file=sys.stderr)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: install transformers to tokenize. `pip install transformers`", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)

    src_lens, tgt_lens = [], []
    src_lens_struct, tgt_lens_struct = [], []
    src_lens_tato, tgt_lens_tato = [], []

    for i, (src, tgt) in enumerate(pairs):
        src_n = len(tok.encode(src, add_special_tokens=True))
        tgt_n = len(tok.encode(tgt, add_special_tokens=True))
        src_lens.append(src_n)
        tgt_lens.append(tgt_n)
        if i < STRUCTURED_END_IDX_EXCLUSIVE:
            src_lens_struct.append(src_n)
            tgt_lens_struct.append(tgt_n)
        else:
            src_lens_tato.append(src_n)
            tgt_lens_tato.append(tgt_n)

    rows = [
        summarize("src / structured", src_lens_struct, args.max_length),
        summarize("src / tatoeba",    src_lens_tato,   args.max_length),
        summarize("src / combined",   src_lens,        args.max_length),
        summarize("tgt / structured", tgt_lens_struct, args.max_length),
        summarize("tgt / tatoeba",    tgt_lens_tato,   args.max_length),
        summarize("tgt / combined",   tgt_lens,        args.max_length),
    ]

    print(f"\n=== Token-length stats (tokenizer: {args.model}, MAX_LENGTH={args.max_length}) ===")
    print_summary(rows)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, (label, struct, tato) in zip(
                axes,
                [
                    ("Source (French) tokens", src_lens_struct, src_lens_tato),
                    ("Target (Adja) tokens",   tgt_lens_struct, tgt_lens_tato),
                ],
            ):
                bins = list(range(0, max(src_lens + tgt_lens) + 5, 5))
                ax.hist([struct, tato], bins=bins, label=["Structured (n=455)", "Tatoeba (n=1000)"], stacked=False)
                ax.axvline(args.max_length, color="red", linestyle="--", linewidth=1, label=f"MAX_LENGTH={args.max_length}")
                ax.set_xlabel(label)
                ax.set_ylabel("Number of test sentences")
                ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            out_pdf = Path(args.output_figure)
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Wrote {out_pdf}")
        except ImportError:
            print("matplotlib not available — skipping figure.", file=sys.stderr)


if __name__ == "__main__":
    main()
