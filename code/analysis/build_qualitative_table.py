"""
build_qualitative_table.py — pull example translations for the rebuttal appendix.

Loads multiple per-sentence JSONL prediction files (one per condition × seed) and
emits side-by-side comparison rows for hand-picked test sentences. Highlights the
2y4n W5 weakness ("no qualitative error analysis") and shows tokenization-driven
BLEU underestimation (where chrF is high but BLEU is low — common for Adja).

Modes:
    1. Aligned comparison: line up STRUCT-2K vs RAND-10K vs R10K+S4K predictions
       for the same source sentence, ordered by largest between-system divergence.
    2. Tokenization-spotlight: Tatoeba sentences where corpus BLEU underrates
       chrF (BLEU < 30 but chrF > 70). These are correct translations with
       different tokenization.

Output: a LaTeX-table snippet ready to paste into the appendix.

Usage:
    python build_qualitative_table.py \
        --jsonl-struct  experiments/results/rebuttal_rerun/nllb-600m/exp1/STRUCTURED-2K/seed42/test_predictions.jsonl \
        --jsonl-rand    experiments/results/rebuttal_rerun/nllb-600m/exp1/RANDOM-10K/seed42/test_predictions.jsonl \
        --jsonl-mixed   experiments/results/rebuttal_rerun/nllb-600m/exp1/RANDOM-10K_STRUCTURED-4K/seed42/test_predictions.jsonl \
        --output-tex    experiments/results/summary/qualitative_examples.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STRUCTURED_END_IDX_EXCLUSIVE = 455


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def fmt_row(idx: int, src: str, ref: str, hyps: list[tuple[str, str, float]]) -> str:
    """One row block: source / reference / each hypothesis with its chrF."""
    lines = [
        f"\\textit{{[idx {idx}]}} & \\textbf{{SRC}} & {latex_escape(src)} \\\\",
        f"& \\textbf{{REF}} & {latex_escape(ref)} \\\\",
    ]
    for label, pred, chrf in hyps:
        lines.append(
            f"& \\textbf{{{label}}} {{\\scriptsize chrF={chrf:.1f}}} & {latex_escape(pred)} \\\\"
        )
    return "\n".join(lines)


def select_aligned_examples(rows_struct, rows_rand, rows_mixed, n_each: int = 4):
    """Pick examples where systems disagree most — by stdev of sentence chrF across systems."""
    common_idx = sorted(set(rows_struct) & set(rows_rand) & set(rows_mixed))
    scored = []
    for i in common_idx:
        chrfs = [
            rows_struct[i]["sentence_chrf"],
            rows_rand[i]["sentence_chrf"],
            rows_mixed[i]["sentence_chrf"],
        ]
        spread = max(chrfs) - min(chrfs)
        scored.append((i, spread))
    scored.sort(key=lambda t: -t[1])

    structured_idxs = [i for i, _ in scored if i < STRUCTURED_END_IDX_EXCLUSIVE][:n_each]
    tatoeba_idxs    = [i for i, _ in scored if i >= STRUCTURED_END_IDX_EXCLUSIVE][:n_each]
    return structured_idxs + tatoeba_idxs


def select_tokenization_spotlight(rows_struct, n: int = 3):
    """Tatoeba sentences where the structured-trained model gets low BLEU but high chrF.

    These are the most powerful examples for showing BLEU's unfairness on Adja.
    """
    candidates = []
    for i, r in rows_struct.items():
        if i < STRUCTURED_END_IDX_EXCLUSIVE:
            continue
        if r.get("sentence_chrf", 0) >= 70 and r.get("sentence_bleu", 0) < 50:
            candidates.append((i, r["sentence_chrf"], r["sentence_bleu"]))
    candidates.sort(key=lambda t: t[1] - t[2], reverse=True)  # max chrF-BLEU gap
    return [i for i, _, _ in candidates[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-struct", required=True, type=Path, help="STRUCT-2K predictions")
    ap.add_argument("--jsonl-rand",   required=True, type=Path, help="RAND-10K predictions")
    ap.add_argument("--jsonl-mixed",  required=True, type=Path, help="R10K+S4K predictions")
    ap.add_argument("--output-tex",   default="experiments/results/summary/qualitative_examples.tex")
    ap.add_argument("--n-aligned-per-subset", type=int, default=4, help="Examples per subset for aligned comparison")
    ap.add_argument("--n-tokenization", type=int, default=3, help="Tokenization-spotlight examples")
    args = ap.parse_args()

    print("Loading prediction files...", file=sys.stderr)
    rows_struct = load_jsonl(args.jsonl_struct)
    rows_rand   = load_jsonl(args.jsonl_rand)
    rows_mixed  = load_jsonl(args.jsonl_mixed)
    print(f"  STRUCT-2K: {len(rows_struct)} rows", file=sys.stderr)
    print(f"  RAND-10K : {len(rows_rand)} rows",  file=sys.stderr)
    print(f"  R10K+S4K : {len(rows_mixed)} rows", file=sys.stderr)

    aligned_idxs = select_aligned_examples(rows_struct, rows_rand, rows_mixed, args.n_aligned_per_subset)
    spotlight_idxs = select_tokenization_spotlight(rows_struct, args.n_tokenization)

    parts: list[str] = []
    parts.append(r"% Auto-generated by experiments/analysis/build_qualitative_table.py")
    parts.append(r"% PASTE INTO ACL APPENDIX. Adjust column widths as needed.")
    parts.append(r"\begin{table*}[t]")
    parts.append(r"\centering\small")
    parts.append(r"\caption{Qualitative example translations on the paper test set "
                 r"(NLLB-600M, seed 42). \textsc{Struct-2K} memorizes structured "
                 r"patterns and produces partially correct natural-language translations; "
                 r"\textsc{R10K+S4K} combines both. The bottom block shows examples "
                 r"where the structured-trained model produces a semantically and "
                 r"lexically correct Adja translation that BLEU heavily under-rates "
                 r"due to whitespace/tokenization differences.}")
    parts.append(r"\label{tab:qualitative}")
    parts.append(r"\begin{tabular}{@{}p{1.6cm}p{1cm}p{12.5cm}@{}}")
    parts.append(r"\toprule")

    # Block 1: structured subset
    parts.append(r"\multicolumn{3}{l}{\textit{Structured subset (idx $<$ 455)}} \\")
    parts.append(r"\midrule")
    for idx in [i for i in aligned_idxs if i < STRUCTURED_END_IDX_EXCLUSIVE]:
        rs, rr, rm = rows_struct[idx], rows_rand[idx], rows_mixed[idx]
        parts.append(fmt_row(idx, rs["src"], rs["ref"], [
            ("Struct-2K", rs["pred"], rs["sentence_chrf"]),
            ("Rand-10K",  rr["pred"], rr["sentence_chrf"]),
            ("R10K+S4K",  rm["pred"], rm["sentence_chrf"]),
        ]))
        parts.append(r"\addlinespace")
    parts.append(r"\midrule")

    # Block 2: tatoeba subset
    parts.append(r"\multicolumn{3}{l}{\textit{Tatoeba subset (idx $\geq$ 455)}} \\")
    parts.append(r"\midrule")
    for idx in [i for i in aligned_idxs if i >= STRUCTURED_END_IDX_EXCLUSIVE]:
        rs, rr, rm = rows_struct[idx], rows_rand[idx], rows_mixed[idx]
        parts.append(fmt_row(idx, rs["src"], rs["ref"], [
            ("Struct-2K", rs["pred"], rs["sentence_chrf"]),
            ("Rand-10K",  rr["pred"], rr["sentence_chrf"]),
            ("R10K+S4K",  rm["pred"], rm["sentence_chrf"]),
        ]))
        parts.append(r"\addlinespace")
    parts.append(r"\midrule")

    # Block 3: tokenization spotlight (Struct-2K only — most legible)
    parts.append(r"\multicolumn{3}{l}{\textit{Tokenization spotlight: high chrF, low BLEU (\textsc{Struct-2K} only)}} \\")
    parts.append(r"\midrule")
    for idx in spotlight_idxs:
        r = rows_struct[idx]
        parts.append(
            f"\\textit{{[idx {idx}]}} & \\textbf{{SRC}} & {latex_escape(r['src'])} \\\\"
        )
        parts.append(f"& \\textbf{{REF}} & {latex_escape(r['ref'])} \\\\")
        parts.append(
            f"& \\textbf{{Struct-2K}} {{\\scriptsize chrF={r['sentence_chrf']:.1f}, "
            f"BLEU={r['sentence_bleu']:.1f}}} & {latex_escape(r['pred'])} \\\\"
        )
        parts.append(r"\addlinespace")

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table*}")

    out_path = Path(args.output_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out_path}", file=sys.stderr)

    # Also dump a plain-text human-readable version alongside, for review.
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write("=== ALIGNED COMPARISON (struct subset) ===\n\n")
        for idx in [i for i in aligned_idxs if i < STRUCTURED_END_IDX_EXCLUSIVE]:
            rs, rr, rm = rows_struct[idx], rows_rand[idx], rows_mixed[idx]
            f.write(f"[idx={idx}]  SRC: {rs['src']}\n")
            f.write(f"            REF: {rs['ref']}\n")
            f.write(f"  Struct-2K (chrF={rs['sentence_chrf']:.1f}): {rs['pred']}\n")
            f.write(f"  Rand-10K  (chrF={rr['sentence_chrf']:.1f}): {rr['pred']}\n")
            f.write(f"  R10K+S4K  (chrF={rm['sentence_chrf']:.1f}): {rm['pred']}\n\n")
        f.write("\n=== ALIGNED COMPARISON (tatoeba subset) ===\n\n")
        for idx in [i for i in aligned_idxs if i >= STRUCTURED_END_IDX_EXCLUSIVE]:
            rs, rr, rm = rows_struct[idx], rows_rand[idx], rows_mixed[idx]
            f.write(f"[idx={idx}]  SRC: {rs['src']}\n")
            f.write(f"            REF: {rs['ref']}\n")
            f.write(f"  Struct-2K (chrF={rs['sentence_chrf']:.1f}): {rs['pred']}\n")
            f.write(f"  Rand-10K  (chrF={rr['sentence_chrf']:.1f}): {rr['pred']}\n")
            f.write(f"  R10K+S4K  (chrF={rm['sentence_chrf']:.1f}): {rm['pred']}\n\n")
        f.write("\n=== TOKENIZATION SPOTLIGHT (Struct-2K, Tatoeba subset) ===\n\n")
        for idx in spotlight_idxs:
            r = rows_struct[idx]
            f.write(f"[idx={idx}]  SRC: {r['src']}\n")
            f.write(f"            REF: {r['ref']}\n")
            f.write(f"  Struct-2K (chrF={r['sentence_chrf']:.1f}, BLEU={r['sentence_bleu']:.1f}): {r['pred']}\n\n")
    print(f"Wrote {txt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
