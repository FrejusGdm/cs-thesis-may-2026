#!/usr/bin/env python3
"""
Full Pipeline Orchestrator
==========================
Runs all three stages in sequence, then creates train/val/test splits
in the exact format Qwen3-TTS or Qwen3-ASR expects.

This is the single entry point for converting your raw dataset
(CSV/TSV with text + audio URLs) into training-ready data.

Usage:
    # For TTS:
    python prepare_dataset.py \
        --input_file dataset.csv \
        --work_dir ./data_workspace \
        --mode tts \
        --language YOUR_LANG_CODE \
        --text_column text \
        --url_column audio_url \
        --ref_audio path/to/reference_speaker.wav \
        --val_split 0.05 \
        --test_split 0.05

    # For ASR:
    python prepare_dataset.py \
        --input_file dataset.csv \
        --work_dir ./data_workspace \
        --mode asr \
        --language YOUR_LANG_CODE \
        --text_column text \
        --url_column audio_url \
        --val_split 0.05 \
        --test_split 0.05
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_splits(
    input_jsonl: str,
    output_dir: str,
    val_split: float = 0.05,
    test_split: float = 0.05,
    seed: int = 42,
):
    """Split dataset into train/val/test with optional stratification.

    We shuffle deterministically (fixed seed) so splits are reproducible.
    This is important: you need the same split every time you re-run
    to avoid data leakage between train and eval sets.
    """
    # Load all records
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    if n < 10:
        logger.warning(
            f"Only {n} records — too few for meaningful splits. "
            "Using all data for training."
        )
        splits = {"train": records}
    else:
        # Deterministic shuffle
        random.seed(seed)
        indices = list(range(n))
        random.shuffle(indices)

        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_val - n_test

        splits = {
            "train": [records[i] for i in indices[:n_train]],
            "val": [records[i] for i in indices[n_train:n_train + n_val]],
            "test": [records[i] for i in indices[n_train + n_val:]],
        }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_records in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in split_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(split_records)} records → {output_path}")

    return splits


def add_ref_audio_field(input_jsonl: str, output_jsonl: str, ref_audio: str):
    """Add ref_audio field to all records (required for TTS fine-tuning).

    Qwen3-TTS fine-tuning expects a `ref_audio` field in every record.
    Using the same reference audio for all samples improves speaker
    consistency — the model learns to associate one speaker identity
    with the linguistic patterns in your data.
    """
    ref_audio = os.path.abspath(ref_audio)
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["ref_audio"] = ref_audio
                records.append(rec)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Added ref_audio to {len(records)} records")


def main():
    parser = argparse.ArgumentParser(
        description="Full data preparation pipeline for Qwen3 TTS/ASR"
    )
    parser.add_argument("--input_file", required=True, help="Raw dataset file (CSV/TSV/JSON/JSONL)")
    parser.add_argument("--work_dir", required=True, help="Working directory for all outputs")
    parser.add_argument("--mode", choices=["tts", "asr"], required=True, help="Target model type")
    parser.add_argument("--language", default="en", help="ISO 639-1 language code")
    parser.add_argument("--text_column", default="text", help="Text column name")
    parser.add_argument("--url_column", default="audio_url", help="Audio URL column name")
    parser.add_argument("--ref_audio", default=None, help="Reference speaker audio (TTS only)")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--max_workers", type=int, default=8, help="Download parallelism")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    target_sr = 24000 if args.mode == "tts" else 16000
    max_duration = 30.0 if args.mode == "tts" else 300.0

    # -----------------------------------------------------------------------
    # Stage 1: Download audio
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Downloading audio files")
    logger.info("=" * 60)

    from data_pipeline.download_audio import download_dataset

    raw_audio_dir = os.path.join(work_dir, "raw_audio")
    raw_jsonl = os.path.join(work_dir, "01_raw.jsonl")

    download_dataset(
        input_file=args.input_file,
        output_dir=raw_audio_dir,
        output_jsonl=raw_jsonl,
        text_column=args.text_column,
        url_column=args.url_column,
        max_workers=args.max_workers,
    )

    # -----------------------------------------------------------------------
    # Stage 2: Process audio
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Processing audio files")
    logger.info("=" * 60)

    from data_pipeline.process_audio import process_dataset

    processed_audio_dir = os.path.join(work_dir, "processed_audio")
    processed_jsonl = os.path.join(work_dir, "02_processed.jsonl")

    process_dataset(
        input_jsonl=raw_jsonl,
        output_dir=processed_audio_dir,
        output_jsonl=processed_jsonl,
        target_sr=target_sr,
        max_duration=max_duration,
    )

    # -----------------------------------------------------------------------
    # Stage 3: Process text
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Normalizing text")
    logger.info("=" * 60)

    from data_pipeline.process_text import process_text_dataset

    normalized_jsonl = os.path.join(work_dir, "03_normalized.jsonl")

    process_text_dataset(
        input_jsonl=processed_jsonl,
        output_jsonl=normalized_jsonl,
        language=args.language,
        mode=args.mode,
    )

    # -----------------------------------------------------------------------
    # Stage 4: Add ref_audio (TTS only) and create splits
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Creating train/val/test splits")
    logger.info("=" * 60)

    final_jsonl = normalized_jsonl

    if args.mode == "tts":
        if args.ref_audio:
            tts_jsonl = os.path.join(work_dir, "04_with_ref.jsonl")
            add_ref_audio_field(normalized_jsonl, tts_jsonl, args.ref_audio)
            final_jsonl = tts_jsonl
        else:
            logger.warning(
                "No --ref_audio provided. For TTS fine-tuning, you need a reference speaker audio file. "
                "You can add it later by editing the JSONL manually."
            )

    splits_dir = os.path.join(work_dir, "splits")
    splits = create_splits(
        final_jsonl, splits_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Final splits in:   {splits_dir}/")
    for name, recs in splits.items():
        logger.info(f"  {name}.jsonl: {len(recs)} samples")

    if args.mode == "tts":
        logger.info(
            "\nNEXT STEP: Run Qwen3-TTS tokenization:\n"
            f"  python -m qwen_tts.finetuning.prepare_data \\\n"
            f"    --input_jsonl {splits_dir}/train.jsonl \\\n"
            f"    --output_jsonl {splits_dir}/train_with_codes.jsonl \\\n"
            f"    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\\n"
            f"    --device cuda:0"
        )
    else:
        logger.info(
            "\nNEXT STEP: Run ASR fine-tuning directly:\n"
            f"  python qwen3_asr_sft.py \\\n"
            f"    --model_path Qwen/Qwen3-ASR-1.7B \\\n"
            f"    --train_file {splits_dir}/train.jsonl \\\n"
            f"    --output_dir ./asr_output"
        )


if __name__ == "__main__":
    main()
