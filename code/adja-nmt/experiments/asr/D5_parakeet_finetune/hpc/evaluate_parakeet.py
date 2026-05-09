#!/usr/bin/env python3
"""
Evaluate a finetuned parakeet-tdt-0.6b-v3 model on a test set.

Loads from a PyTorch Lightning .ckpt checkpoint (not .nemo),
runs inference, computes per-sentence WER/CER, and saves results.

Usage:
    python evaluate_parakeet.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --test_manifest /path/to/test_manifest.json \
        --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import jiwer

import nemo.collections.asr as nemo_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_from_ckpt(ckpt_path: str):
    """
    Load a finetuned ASR model from a .ckpt file.

    Strategy:
      1. Load the pretrained model architecture from HuggingFace
      2. Load the finetuned weights from the .ckpt file
    """
    logger.info("Loading pretrained model architecture …")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )

    logger.info(f"Loading finetuned weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # PL checkpoints store state_dict under "state_dict" key
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load weights (strict=False in case of minor mismatches)
    missing, unexpected = asr_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}…")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}…")

    logger.info("Finetuned weights loaded successfully.")
    return asr_model


def evaluate(
    asr_model,
    test_manifest: str,
    output_dir: str,
    batch_size: int = 16,
) -> None:
    """
    Run inference on the test set, compute per-sentence WER & CER,
    and save results to CSV + a summary text file.
    """
    logger.info("=" * 70)
    logger.info("  EVALUATION on test set")
    logger.info("=" * 70)

    asr_model.eval()
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()

    # ── Read test manifest ─────────────────────────────────────────────
    audio_paths = []
    references = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_paths.append(entry["audio_filepath"])
            references.append(entry["text"])

    logger.info(f"Test set size: {len(audio_paths)} utterances")

    # ── Run inference ──────────────────────────────────────────────────
    logger.info("Running inference …")
    with torch.no_grad():
        hypotheses = asr_model.transcribe(
            audio_paths,
            batch_size=batch_size,
            return_hypotheses=False,
        )

    # Handle different return types from transcribe()
    if isinstance(hypotheses, (list, tuple)):
        if isinstance(hypotheses[0], (list, tuple)):
            predictions = [str(h) for h in hypotheses[0]]
        elif hasattr(hypotheses[0], "text"):
            predictions = [h.text for h in hypotheses]
        else:
            predictions = [str(h) for h in hypotheses]
    else:
        predictions = [str(hypotheses)]

    # ── Compute per-sentence WER and CER ──────────────────────────────
    logger.info("Computing WER and CER for each sentence …")

    wer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    results = []
    all_wers = []
    all_cers = []

    for ref, hyp, audio_fp in zip(references, predictions, audio_paths):
        ref_clean = ref.strip() if ref else ""
        hyp_clean = hyp.strip() if hyp else ""

        if ref_clean == "" and hyp_clean == "":
            sent_wer = 0.0
            sent_cer = 0.0
        elif ref_clean == "":
            sent_wer = 1.0
            sent_cer = 1.0
        else:
            try:
                sent_wer = jiwer.wer(
                    ref_clean, hyp_clean,
                    truth_transform=wer_transform,
                    hypothesis_transform=wer_transform,
                )
            except Exception:
                sent_wer = 1.0
            try:
                sent_cer = jiwer.cer(
                    ref_clean, hyp_clean,
                    truth_transform=cer_transform,
                    hypothesis_transform=cer_transform,
                )
            except Exception:
                sent_cer = 1.0

        all_wers.append(sent_wer)
        all_cers.append(sent_cer)

        results.append({
            "audio_filepath": audio_fp,
            "reference": ref_clean,
            "prediction": hyp_clean,
            "wer": round(sent_wer, 6),
            "cer": round(sent_cer, 6),
        })

    # ── Save per-sentence results to CSV ──────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "test_results.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved per-sentence results to: {results_csv_path}")

    # ── Compute aggregate metrics ─────────────────────────────────────
    median_wer = float(np.median(all_wers))
    median_cer = float(np.median(all_cers))
    mean_wer = float(np.mean(all_wers))
    mean_cer = float(np.mean(all_cers))

    try:
        corpus_wer = jiwer.wer(
            references, predictions,
            truth_transform=wer_transform,
            hypothesis_transform=wer_transform,
        )
    except Exception:
        corpus_wer = float("nan")

    try:
        corpus_cer = jiwer.cer(
            references, predictions,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
    except Exception:
        corpus_cer = float("nan")

    # ── Save summary to text file ─────────────────────────────────────
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    summary_lines = [
        "=" * 60,
        "  EVALUATION SUMMARY",
        "=" * 60,
        f"Test manifest:       {test_manifest}",
        f"Number of sentences: {len(all_wers)}",
        "",
        "--- Per-Sentence Metrics ---",
        f"Median WER:  {median_wer:.6f}  ({median_wer * 100:.2f}%)",
        f"Median CER:  {median_cer:.6f}  ({median_cer * 100:.2f}%)",
        f"Mean WER:    {mean_wer:.6f}  ({mean_wer * 100:.2f}%)",
        f"Mean CER:    {mean_cer:.6f}  ({mean_cer * 100:.2f}%)",
        "",
        "--- Corpus-Level Metrics ---",
        f"Corpus WER:  {corpus_wer:.6f}  ({corpus_wer * 100:.2f}%)",
        f"Corpus CER:  {corpus_cer:.6f}  ({corpus_cer * 100:.2f}%)",
        "=" * 60,
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    for line in summary_lines:
        logger.info(line)

    logger.info(f"Saved evaluation summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned parakeet-tdt-0.6b-v3 on a test set."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .ckpt checkpoint file.",
    )
    parser.add_argument(
        "--test_manifest", type=str, required=True,
        help="Path to test_manifest.json (NeMo JSON-lines format).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_output",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference.",
    )
    args = parser.parse_args()

    # Load model
    asr_model = load_model_from_ckpt(args.checkpoint)

    # Evaluate
    evaluate(
        asr_model=asr_model,
        test_manifest=args.test_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    logger.info("All done!")


if __name__ == "__main__":
    main()
