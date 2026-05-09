"""
evaluate.py — Compute all automatic metrics for a predictions file.

Metrics computed:
  - BLEU (SacreBLEU)
  - chrF, chrF++
  - TER
  - COMET (wmt22-comet-da)
  - BERTScore

Usage:
    python evaluate.py \
        --predictions ./results/exp1/RANDOM-10K/seed42/predictions.tsv \
        --output ./results/exp1/RANDOM-10K/seed42/full_metrics.json

    # Or evaluate a directory of predictions across seeds:
    python evaluate.py \
        --predictions-dir ./results/exp1/RANDOM-10K/ \
        --output ./results/exp1/RANDOM-10K/aggregated_metrics.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_predictions(pred_file):
    """Load predictions TSV: src\\tref\\tpred per line."""
    sources, references, predictions = [], [], []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                sources.append(parts[0])
                references.append(parts[1])
                predictions.append(parts[2])
    return sources, references, predictions


def compute_sacrebleu_metrics(references, predictions):
    """Compute BLEU, chrF, chrF++, TER using SacreBLEU."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])

    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "chrf": chrf.score,
        "chrf_signature": str(chrf),
        "chrfpp": chrfpp.score,
        "ter": ter.score,
    }


def compute_comet(sources, references, predictions, model_name="Unbabel/wmt22-comet-da"):
    """Compute COMET score."""
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(sources, predictions, references)
        ]
        output = model.predict(data, batch_size=32, gpus=1 if _has_gpu() else 0)
        return {
            "comet": float(output.system_score),
            "comet_model": model_name,
        }
    except ImportError:
        print("WARNING: COMET not installed (pip install unbabel-comet). Skipping.")
        return {"comet": None, "comet_model": model_name}
    except Exception as e:
        print(f"WARNING: COMET failed: {e}")
        return {"comet": None, "comet_model": model_name}


def compute_bertscore(references, predictions, model_name="bert-base-multilingual-cased"):
    """Compute BERTScore."""
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(
            predictions, references,
            model_type=model_name,
            verbose=False,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
            "bertscore_model": model_name,
        }
    except ImportError:
        print("WARNING: bert-score not installed (pip install bert-score). Skipping.")
        return {"bertscore_f1": None, "bertscore_model": model_name}
    except Exception as e:
        print(f"WARNING: BERTScore failed: {e}")
        return {"bertscore_f1": None, "bertscore_model": model_name}


def _has_gpu():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def evaluate_single(pred_file, output_file=None, skip_comet=False, skip_bertscore=False):
    """Evaluate a single predictions file."""
    sources, references, predictions = load_predictions(pred_file)
    print(f"Loaded {len(predictions)} predictions from {pred_file}")

    metrics = {"n_samples": len(predictions), "predictions_file": pred_file}

    # SacreBLEU metrics
    print("Computing BLEU, chrF, chrF++, TER...")
    sacrebleu_metrics = compute_sacrebleu_metrics(references, predictions)
    metrics.update(sacrebleu_metrics)
    print(f"  BLEU={sacrebleu_metrics['bleu']:.2f}  "
          f"chrF={sacrebleu_metrics['chrf']:.2f}  "
          f"chrF++={sacrebleu_metrics['chrfpp']:.2f}  "
          f"TER={sacrebleu_metrics['ter']:.2f}")

    # COMET
    if not skip_comet:
        print("Computing COMET...")
        comet_metrics = compute_comet(sources, references, predictions)
        metrics.update(comet_metrics)
        if comet_metrics["comet"] is not None:
            print(f"  COMET={comet_metrics['comet']:.4f}")

    # BERTScore
    if not skip_bertscore:
        print("Computing BERTScore...")
        bert_metrics = compute_bertscore(references, predictions)
        metrics.update(bert_metrics)
        if bert_metrics["bertscore_f1"] is not None:
            print(f"  BERTScore F1={bert_metrics['bertscore_f1']:.4f}")

    # Save
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")

    return metrics


def evaluate_directory(pred_dir, output_file=None, skip_comet=False, skip_bertscore=False):
    """Evaluate all seed predictions in a directory and aggregate."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "seed*/predictions.tsv")))
    if not pred_files:
        print(f"No prediction files found in {pred_dir}/seed*/predictions.tsv")
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction files")
    all_metrics = []

    for pf in pred_files:
        seed_dir = os.path.dirname(pf)
        seed_name = os.path.basename(seed_dir)
        print(f"\n--- {seed_name} ---")
        out = os.path.join(seed_dir, "full_metrics.json")
        metrics = evaluate_single(pf, out, skip_comet, skip_bertscore)
        metrics["seed"] = seed_name
        all_metrics.append(metrics)

    # Aggregate
    aggregated = {"condition": os.path.basename(pred_dir), "n_seeds": len(all_metrics)}
    metric_keys = ["bleu", "chrf", "chrfpp", "ter", "comet", "bertscore_f1"]
    for key in metric_keys:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(min(values))
            aggregated[f"{key}_max"] = float(max(values))
            aggregated[f"{key}_values"] = values

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS ({len(all_metrics)} seeds)")
    print(f"{'='*60}")
    for key in metric_keys:
        if f"{key}_mean" in aggregated:
            print(f"  {key}: {aggregated[f'{key}_mean']:.2f} +/- {aggregated[f'{key}_std']:.2f}")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated metrics saved to {output_file}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Compute all automatic metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", help="Single predictions TSV file")
    group.add_argument("--predictions-dir", help="Directory with seed*/predictions.tsv")
    parser.add_argument("--output", help="Output JSON file for metrics")
    parser.add_argument("--skip-comet", action="store_true", help="Skip COMET (requires GPU)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore")
    args = parser.parse_args()

    if args.predictions:
        evaluate_single(args.predictions, args.output, args.skip_comet, args.skip_bertscore)
    else:
        evaluate_directory(args.predictions_dir, args.output, args.skip_comet, args.skip_bertscore)


if __name__ == "__main__":
    main()
