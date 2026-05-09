"""
compute_neural_metrics.py — Add BERTScore + COMET to saved test predictions.

Reads every test_predictions.jsonl under --results-root, computes:

  - **BERTScore** using an African-language-aware encoder
    (default: Davlan/afro-xlmr-large)
  - **COMET** using McGill-NLP/ssa-comet-mtl (an SSA-fine-tuned COMET)

…and writes the per-system aggregates to test_neural_metrics.json next
to the existing test_metrics.json. Per-sentence scores are appended back
into the predictions JSONL (so future analysis can slice errors).

WHY THIS LIVES OUTSIDE THE TRAINING JOB
We don't want HPC training jobs entangled with eval-model downloads
(no Internet on compute nodes). This script runs locally (or on any node
that has Internet for the first model fetch — both models are then HF-cached).

Adja-specific note: Adja is NOT directly in either model's training data,
but Ewe and Fon ARE (same Gbe language family, similar lexicon and syntax).
Treat the resulting scores as a *discriminative* signal — they will
under-rate absolute quality but should preserve system rankings.

USAGE
    # Compute on every saved prediction file under the april-2026 sweep:
    python experiments/analysis/compute_neural_metrics.py

    # Subset to a single condition / model:
    python experiments/analysis/compute_neural_metrics.py \\
        --results-root experiments/results/april2026 \\
        --filter nllb-600m/april2026_runA/FULL

    # Skip COMET (faster — useful for quick BERTScore-only passes):
    python experiments/analysis/compute_neural_metrics.py --no-comet

INSTALL FIRST
    pip install bert-score unbabel-comet
    # On first run each model is downloaded (~2-4 GB combined) into HF cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_BERT_MODEL = "Davlan/afro-xlmr-large"
# Per the model card, ssa-comet-mtl is fine-tuned on Ewe + Fon (among others)
# — both Gbe languages closely related to Adja.
DEFAULT_COMET_MODEL = "McGill-NLP/ssa-comet-mtl"


def find_prediction_files(root: Path, name_filter: str | None) -> list[Path]:
    files = sorted(root.rglob("test_predictions.jsonl"))
    if name_filter:
        files = [f for f in files if name_filter in str(f.relative_to(root))]
    return files


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_bertscore(rows: list[dict], model_type: str, device: str | None) -> dict | None:
    try:
        from bert_score import BERTScorer
    except ImportError:
        print(f"  [bertscore] skipped (`pip install bert-score` to enable)")
        return None
    refs = [r["ref"] for r in rows]
    preds = [r["pred"] for r in rows]
    print(f"  [bertscore] scoring {len(refs)} pairs with {model_type}…")
    t0 = time.time()
    # Pass model_type explicitly so it doesn't try to look up by language code
    # (the standard `--lang` lookup table doesn't know about Gbe languages).
    scorer = BERTScorer(model_type=model_type, num_layers=None, device=device, lang="en")
    P, R, F1 = scorer.score(preds, refs, verbose=False)
    print(f"  [bertscore] done in {time.time() - t0:.1f}s")

    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()
    for row, p, r, f1 in zip(rows, p_list, r_list, f1_list):
        row["bertscore_p"] = float(p)
        row["bertscore_r"] = float(r)
        row["bertscore_f1"] = float(f1)
    return {
        "bertscore_model": model_type,
        "bertscore_p_mean": float(P.mean()),
        "bertscore_r_mean": float(R.mean()),
        "bertscore_f1_mean": float(F1.mean()),
        "bertscore_f1_std": float(F1.std()),
    }


def compute_comet(rows: list[dict], model_id: str, gpus: int) -> dict | None:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(f"  [comet] skipped (`pip install unbabel-comet` to enable)")
        return None
    print(f"  [comet] loading {model_id}…")
    t0 = time.time()
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)
    data = [{"src": r["src"], "mt": r["pred"], "ref": r["ref"]} for r in rows]
    print(f"  [comet] scoring {len(data)} pairs (gpus={gpus})…")
    out = model.predict(data, batch_size=8, gpus=gpus)
    print(f"  [comet] done in {time.time() - t0:.1f}s")

    # COMET 2.x: out["scores"] is the per-sentence list, out["system_score"]
    # is the corpus mean. Older versions: out is a (scores, system_score)
    # tuple — handle both.
    if isinstance(out, tuple):
        scores, sys_score = out
    else:
        scores = out["scores"]
        sys_score = out.get("system_score", sum(scores) / len(scores))
    for row, s in zip(rows, scores):
        row["comet"] = float(s)
    return {
        "comet_model": model_id,
        "comet_system_score": float(sys_score),
        "comet_score_mean": float(sum(scores) / len(scores)),
    }


def process(path: Path, args) -> dict:
    rows = load_predictions(path)
    if not rows:
        print(f"  empty file, skipping: {path}")
        return {"path": str(path), "skipped": "empty"}

    aggregates: dict = {"path": str(path), "n": len(rows)}

    if not args.no_bertscore:
        bs = compute_bertscore(rows, args.bertscore_model, args.device)
        if bs:
            aggregates.update(bs)

    if not args.no_comet:
        cm = compute_comet(rows, args.comet_model, args.comet_gpus)
        if cm:
            aggregates.update(cm)

    if not args.dry_run:
        # 1. Write the aggregate sidecar.
        sidecar = path.with_name("test_neural_metrics.json")
        with sidecar.open("w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)
        # 2. Re-write the JSONL with the per-sentence scores attached.
        write_predictions(path, rows)
        # 3. Optionally merge aggregates into test_metrics.json so the
        #    existing aggregator picks them up without code changes.
        if args.merge:
            metrics_path = path.with_name("test_metrics.json")
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                for k, v in aggregates.items():
                    if k not in {"path", "n"}:
                        metrics[k] = v
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
    return aggregates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results-root", type=Path,
                    default=Path("experiments/results/april2026"),
                    help="Search this tree for test_predictions.jsonl files.")
    ap.add_argument("--filter", default=None,
                    help="Only process files whose path contains this substring.")
    ap.add_argument("--bertscore-model", default=DEFAULT_BERT_MODEL,
                    help=f"BERTScore encoder (default: {DEFAULT_BERT_MODEL}).")
    ap.add_argument("--comet-model", default=DEFAULT_COMET_MODEL,
                    help=f"COMET checkpoint (default: {DEFAULT_COMET_MODEL}).")
    ap.add_argument("--device", default=None,
                    help="BERTScore device, e.g. 'cuda', 'mps', 'cpu'. "
                         "Default lets bert-score pick.")
    ap.add_argument("--comet-gpus", type=int, default=0,
                    help="GPUs for COMET (default 0 = CPU; set 1 if you have CUDA).")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--no-merge", dest="merge", action="store_false", default=True,
                    help="Don't merge aggregates back into test_metrics.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute scores but don't write any files.")
    args = ap.parse_args()

    if not args.results_root.exists():
        print(f"ERROR: results root not found: {args.results_root}")
        return 1

    files = find_prediction_files(args.results_root, args.filter)
    if not files:
        print(f"No test_predictions.jsonl under {args.results_root}"
              + (f" matching '{args.filter}'" if args.filter else ""))
        return 0

    print(f"Found {len(files)} prediction file(s) to process.")
    summaries = []
    for i, f in enumerate(files, 1):
        rel = f.relative_to(args.results_root)
        print(f"\n[{i}/{len(files)}] {rel}")
        try:
            summaries.append(process(f, args))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summaries.append({"path": str(f), "error": str(e)})

    # Top-level summary index for quick scanning.
    if not args.dry_run:
        summary_path = args.results_root / "neural_metrics_index.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
