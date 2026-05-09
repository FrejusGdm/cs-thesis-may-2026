"""
recompute_subset_metrics.py — compute subset BLEU/chrF/chrF++ from per-sentence JSONL.

The test set is laid out as: idx 0..454 = structured (455 rows), idx 455..1454 = Tatoeba
(1000 rows). Boundary verified against test.tsv: line 455 is the last Module-5 question
("Que savent-elles parler?"), line 456 is the first Tatoeba ("Je sue tous les jours.").

The training-time subset eval in hf_job_train_hpc.py uses string-matching against
structured_train.tsv, which silently breaks for held-out test sentences (none of them
appear in the train set by definition). Recomputing offline by index is robust.

Usage:
    python recompute_subset_metrics.py <jsonl_file> [<jsonl_file> ...]
    python recompute_subset_metrics.py --glob "experiments/results/april2026/**/test_predictions.jsonl"
    python recompute_subset_metrics.py --glob "..." --output subset_metrics.csv
    python recompute_subset_metrics.py --glob "..." \
      --test-csv experiments/data/splits/shared/test.csv \
      --alternative-refs experiments/data/cleaned/source_target_repair.alternative_references.tsv \
      --ref-mode multi

Output: a CSV with columns model, run, condition, seed, subset, n_samples,
bleu, chrf, chrfpp, ref_mode, alt_ref_rows_used.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import sacrebleu

STRUCTURED_END_IDX_EXCLUSIVE = 455  # 0..454 structured, 455..1454 tatoeba
RESULT_ROOTS = {
    "april2026",
    "hpc_new",
    "rebuttal_rerun",
    "rebuttal_repaired_20260502",
    "pipeline_repaired_20260502",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_path_metadata(path: Path) -> dict:
    """Extract model/run/condition/seed from a results path like
    .../april2026/nllb-600m/april2026_runA/FULL/seed42/test_predictions.jsonl.
    Falls back to the literal path if the layout doesn't match.
    """
    parts = path.parts
    md = {"model": "?", "run": "?", "condition": "?", "seed": "?"}
    for i, p in enumerate(parts):
        if p in RESULT_ROOTS:
            try:
                if parts[i + 1] in {"forward", "reverse"}:
                    direction = parts[i + 1]
                    md["model"] = f"{direction}/{parts[i + 2]}"
                    md["run"] = parts[i + 3]
                    md["condition"] = parts[i + 4]
                    md["seed"] = parts[i + 5]
                else:
                    md["model"] = parts[i + 1]
                    md["run"] = parts[i + 2]
                    md["condition"] = parts[i + 3]
                    md["seed"] = parts[i + 4]
            except IndexError:
                pass
            break
    md["seed"] = re.sub(r"^seed", "", md["seed"])
    return md


def load_test_sentence_ids(path: Path | None) -> dict[int, str]:
    """Map JSONL idx to sentence_id using the shared test CSV row order."""
    if path is None:
        return {}
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "sentence_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no sentence_id column")
        for idx, row in enumerate(reader):
            out[idx] = row["sentence_id"]
    return out


def load_alternative_refs(path: Path | None) -> dict[str, dict[str, object]]:
    """Load sidecar refs keyed by sentence_id.

    The canonical field is used as a direction guard: alternatives are only
    applied when a JSONL row's ref exactly equals the canonical Adja target.
    Reverse jobs have French refs and therefore remain canonical-only.
    """
    if path is None:
        return {}
    out: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence_id", "canonical_adja", "alternative_adja"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            sid = row["sentence_id"]
            canonical = row["canonical_adja"].strip()
            alternative = row["alternative_adja"].strip()
            if not sid or not canonical or not alternative or canonical == alternative:
                continue
            bucket = out.setdefault(sid, {"canonical": canonical, "alternatives": []})
            if bucket["canonical"] != canonical:
                raise ValueError(f"Conflicting canonical alternatives for sentence_id={sid}")
            alternatives = bucket["alternatives"]
            assert isinstance(alternatives, list)
            if alternative not in alternatives:
                alternatives.append(alternative)
    return out


def reference_variants(
    row: dict,
    *,
    ref_mode: str,
    idx_to_sentence_id: dict[int, str],
    alternative_refs: dict[str, dict[str, object]],
) -> tuple[list[str], int]:
    canonical = row["ref"]
    if ref_mode != "multi":
        return [canonical], 0

    sid = idx_to_sentence_id.get(int(row["idx"]))
    if not sid or sid not in alternative_refs:
        return [canonical], 0

    entry = alternative_refs[sid]
    if entry["canonical"] != canonical:
        return [canonical], 0

    alternatives = [a for a in entry["alternatives"] if a != canonical]
    return [canonical, *alternatives], 1 if alternatives else 0


def compute_metrics(ref_variants: list[list[str]], hyps: list[str]) -> dict:
    if not ref_variants:
        return {"bleu": float("nan"), "chrf": float("nan"), "chrfpp": float("nan")}
    max_refs = max(len(refs) for refs in ref_variants)
    ref_streams: list[list[str]] = []
    for ref_idx in range(max_refs):
        stream = []
        for refs in ref_variants:
            stream.append(refs[ref_idx] if ref_idx < len(refs) else refs[0])
        ref_streams.append(stream)
    bleu = sacrebleu.corpus_bleu(hyps, ref_streams).score
    chrf = sacrebleu.corpus_chrf(hyps, ref_streams, word_order=0).score
    chrfpp = sacrebleu.corpus_chrf(hyps, ref_streams, word_order=2).score
    return {"bleu": bleu, "chrf": chrf, "chrfpp": chrfpp}


def process_file(
    path: Path,
    *,
    ref_mode: str,
    idx_to_sentence_id: dict[int, str],
    alternative_refs: dict[str, dict[str, object]],
) -> list[dict]:
    rows = load_jsonl(path)
    md = parse_path_metadata(path)

    scored_rows = []
    for r in rows:
        refs, used_alt = reference_variants(
            r,
            ref_mode=ref_mode,
            idx_to_sentence_id=idx_to_sentence_id,
            alternative_refs=alternative_refs,
        )
        scored_rows.append({"idx": int(r["idx"]), "refs": refs, "pred": r["pred"], "used_alt": used_alt})

    structured = [r for r in scored_rows if r["idx"] < STRUCTURED_END_IDX_EXCLUSIVE]
    tatoeba = [r for r in scored_rows if r["idx"] >= STRUCTURED_END_IDX_EXCLUSIVE]
    combined = scored_rows

    out = []
    for subset_name, pairs in [("combined", combined), ("structured", structured), ("tatoeba", tatoeba)]:
        refs = [r["refs"] for r in pairs]
        hyps = [r["pred"] for r in pairs]
        m = compute_metrics(refs, hyps)
        out.append({
            **md,
            "subset": subset_name,
            "n_samples": len(pairs),
            "bleu": round(m["bleu"], 2),
            "chrf": round(m["chrf"], 2),
            "chrfpp": round(m["chrfpp"], 2),
            "ref_mode": ref_mode,
            "alt_ref_rows_used": sum(r["used_alt"] for r in pairs),
            "source_file": str(path),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="JSONL prediction files")
    ap.add_argument("--glob", help="Glob pattern (e.g. 'experiments/results/april2026/**/test_predictions.jsonl')")
    ap.add_argument("--test-csv", help="Shared test CSV used to map JSONL idx to sentence_id")
    ap.add_argument("--alternative-refs", help="TSV sidecar with canonical and alternative Adja references")
    ap.add_argument("--ref-mode", choices=["canonical", "multi"], default="canonical")
    ap.add_argument("--output", help="CSV output path; if omitted, prints a TSV summary to stdout")
    args = ap.parse_args()

    if args.ref_mode == "multi" and (not args.test_csv or not args.alternative_refs):
        ap.error("--ref-mode multi requires --test-csv and --alternative-refs")

    paths: list[Path] = []
    if args.glob:
        paths.extend(Path(p) for p in glob.glob(args.glob, recursive=True))
    paths.extend(Path(p) for p in args.paths)
    paths = sorted(set(paths))

    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    idx_to_sentence_id = load_test_sentence_ids(Path(args.test_csv)) if args.test_csv else {}
    alternative_refs = load_alternative_refs(Path(args.alternative_refs)) if args.alternative_refs else {}

    rows: list[dict] = []
    for p in paths:
        try:
            rows.extend(process_file(
                p,
                ref_mode=args.ref_mode,
                idx_to_sentence_id=idx_to_sentence_id,
                alternative_refs=alternative_refs,
            ))
        except Exception as e:  # noqa: BLE001
            print(f"WARN: {p} failed: {e}", file=sys.stderr)

    if args.output:
        fieldnames = [
            "model", "run", "condition", "seed", "subset", "n_samples",
            "bleu", "chrf", "chrfpp", "ref_mode", "alt_ref_rows_used", "source_file",
        ]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
    else:
        # Pretty stdout summary, grouped by file
        last_file = None
        for r in rows:
            if r["source_file"] != last_file:
                print()
                print(f"{r['model']} / {r['run']} / {r['condition']} / seed{r['seed']}")
                print(f"  ({r['source_file']})")
                last_file = r["source_file"]
            print(f"  {r['subset']:<11} n={r['n_samples']:>4}   "
                  f"BLEU {r['bleu']:>5.2f}   chrF {r['chrf']:>5.2f}   chrF++ {r['chrfpp']:>5.2f}")


if __name__ == "__main__":
    main()
