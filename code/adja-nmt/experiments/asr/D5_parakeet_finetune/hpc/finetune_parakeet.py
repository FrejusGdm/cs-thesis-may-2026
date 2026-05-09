#!/usr/bin/env python3
"""
Finetune nvidia/parakeet-tdt-0.6b-v3 on custom data and evaluate on test set.

Expected CSV format (train/dev/test):
  - Columns: "audio_filepath", "text"
  - Optional columns: "duration" (float, in seconds)
  - audio_filepath: absolute or relative path to .wav files (16kHz mono recommended)
  - text: ground-truth transcription

Usage:
    python finetune_parakeet.py \
        --train_csv /path/to/train.csv \
        --dev_csv /path/to/dev.csv \
        --test_csv /path/to/test.csv \
        --output_dir /path/to/output \
        --num_epochs 10 \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_gpus 1
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# ─── NeMo imports ───────────────────────────────────────────────────────────
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.core.config import hydra_runner
from nemo.utils import logging as nemo_logging
from nemo.utils.exp_manager import exp_manager

import lightning.pytorch as pl
from omegaconf import OmegaConf, DictConfig

# ─── Evaluation metric imports ──────────────────────────────────────────────
# jiwer is used for per-sentence WER/CER
import jiwer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: CSV → NeMo JSON manifest
# ═══════════════════════════════════════════════════════════════════════════
def csv_to_nemo_manifest(csv_path: str, manifest_path: str) -> str:
    """
    Convert a CSV file with columns [audio_filepath, text, (duration)] to a
    NeMo-compatible JSON-lines manifest.

    If 'duration' is missing from the CSV, it will be computed from the audio.
    Returns the path to the generated manifest.
    """
    df = pd.read_csv(csv_path)

    required = {"audio_filepath", "text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {required}. Found: {list(df.columns)}"
        )

    has_duration = "duration" in df.columns
    missing_files = []

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            audio_fp = str(row["audio_filepath"])
            text = str(row["text"])

            if has_duration and pd.notna(row["duration"]):
                dur = float(row["duration"])
            else:
                # Compute duration from the audio file
                if not os.path.isfile(audio_fp):
                    missing_files.append(audio_fp)
                    continue
                try:
                    info = sf.info(audio_fp)
                    dur = info.duration
                except Exception as e:
                    logger.error(
                        f"Could not read audio file {audio_fp}: {e}"
                    )
                    missing_files.append(audio_fp)
                    continue

            entry = {
                "audio_filepath": audio_fp,
                "text": text,
                "duration": round(dur, 4),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if missing_files:
        logger.error(
            f"{len(missing_files)} audio file(s) could not be found or read. "
            f"First 5: {missing_files[:5]}"
        )
        logger.error(
            "Check that: (1) the paths in your CSV are correct, and "
            "(2) the audio directory is bind-mounted in Singularity "
            "(--bind /path/to/audio:/path/to/audio)."
        )
        if len(missing_files) == len(df):
            raise FileNotFoundError(
                "ALL audio files are missing — cannot proceed. "
                "Are your Singularity bind mounts correct?"
            )

    n_lines = sum(1 for _ in open(manifest_path))
    logger.info(f"Created manifest {manifest_path} with {n_lines} entries.")
    return manifest_path


# ═══════════════════════════════════════════════════════════════════════════
#  Finetuning
# ═══════════════════════════════════════════════════════════════════════════
def finetune(
    train_manifest: str,
    dev_manifest: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_gpus: int = 1,
    warmup_steps: int = 500,
    weight_decay: float = 1e-3,
    accumulate_grad_batches: int = 1,
) -> str:
    """
    Finetune parakeet-tdt-0.6b-v3 on the given training manifest.
    Returns the path to the best checkpoint (.nemo file).
    """
    logger.info("=" * 70)
    logger.info("  FINETUNING: nvidia/parakeet-tdt-0.6b-v3")
    logger.info("=" * 70)

    # ── Trainer (must be created BEFORE model for NeMo compatibility) ──
    exp_dir = os.path.join(output_dir, "nemo_experiments")
    os.makedirs(exp_dir, exist_ok=True)

    trainer_cfg = {
        "devices": num_gpus,
        "accelerator": "gpu",
        "max_epochs": num_epochs,
        "accumulate_grad_batches": accumulate_grad_batches,
        "precision": "bf16-mixed",
        "log_every_n_steps": 10,
        "enable_checkpointing": False,  # exp_manager handles checkpointing
        "logger": False,  # we use exp_manager for logging
        "num_sanity_val_steps": 2,
    }

    trainer = pl.Trainer(**trainer_cfg)

    # ── Experiment manager (handles checkpoints & logging) ─────────────
    exp_manager_cfg = {
        "exp_dir": exp_dir,
        "name": "parakeet_tdt_finetune",
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 3,
            "always_save_nemo": True,
        },
        "create_tensorboard_logger": True,
    }
    exp_manager(trainer, DictConfig(exp_manager_cfg))

    # ── Load pretrained model WITH trainer ─────────────────────────────
    logger.info("Loading pretrained model …")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        trainer=trainer,
    )

    # ── Update dataset configs ─────────────────────────────────────────
    OmegaConf.set_struct(asr_model.cfg, False)

    # Training data
    asr_model.cfg.train_ds.manifest_filepath = train_manifest
    asr_model.cfg.train_ds.batch_size = batch_size
    asr_model.cfg.train_ds.shuffle = True
    asr_model.cfg.train_ds.num_workers = 4
    asr_model.cfg.train_ds.pin_memory = True
    asr_model.cfg.train_ds.max_duration = 20.0
    asr_model.cfg.train_ds.min_duration = 0.1
    asr_model.cfg.train_ds.is_tarred = False
    asr_model.cfg.train_ds.use_lhotse = False
    asr_model.cfg.train_ds.pretokenize = False

    # Validation data
    asr_model.cfg.validation_ds.manifest_filepath = dev_manifest
    asr_model.cfg.validation_ds.batch_size = batch_size
    asr_model.cfg.validation_ds.num_workers = 4
    asr_model.cfg.validation_ds.pin_memory = True
    asr_model.cfg.validation_ds.max_duration = 20.0
    asr_model.cfg.validation_ds.use_lhotse = False
    asr_model.cfg.validation_ds.pretokenize = False

    OmegaConf.set_struct(asr_model.cfg, True)

    # ── Apply dataset config changes ───────────────────────────────────
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)

    # ── Freeze encoder for first pass (optional — improves stability) ──
    # Uncomment the next 2 lines to freeze the encoder and only train decoder/joint:
    # asr_model.encoder.freeze()
    # logger.info("Encoder frozen — only decoder & joint will be trained.")

    # ── Estimate max_steps for the scheduler ───────────────────────────
    n_train = sum(1 for _ in open(train_manifest))
    steps_per_epoch = max(1, n_train // (batch_size * accumulate_grad_batches))
    max_steps = steps_per_epoch * num_epochs
    logger.info(
        f"Estimated steps_per_epoch={steps_per_epoch}, "
        f"max_steps={max_steps}"
    )

    # ── Optimizer config ───────────────────────────────────────────────
    optim_cfg = OmegaConf.create(
        {
            "name": "adamw",
            "lr": learning_rate,
            "betas": [0.9, 0.98],
            "weight_decay": weight_decay,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "min_lr": 1e-6,
            },
        }
    )
    asr_model.setup_optimization(optim_cfg)

    # ── Train ──────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.fit(asr_model)
    logger.info("Training complete.")

    # ── Find the best checkpoint (.ckpt) ───────────────────────────────
    # exp_manager saves checkpoints in the experiment directory.
    # We find the best .ckpt file (lowest val_wer in filename).
    best_ckpt_path = None
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Best checkpoint (from callback): {best_ckpt_path}")
    else:
        # Fallback: search the experiment directory for .ckpt files
        import glob
        ckpt_pattern = os.path.join(exp_dir, "**", "*.ckpt")
        ckpt_files = glob.glob(ckpt_pattern, recursive=True)
        # Filter out "-last.ckpt" files
        ckpt_files = [f for f in ckpt_files if "-last" not in f]
        if ckpt_files:
            # Sort by val_wer in filename if possible
            best_ckpt_path = sorted(ckpt_files)[0]
            logger.info(f"Best checkpoint (from search): {best_ckpt_path}")

    if best_ckpt_path is None:
        # Last resort: save as .nemo
        best_ckpt_path = os.path.join(output_dir, "parakeet_tdt_finetuned.nemo")
        asr_model.save_to(best_ckpt_path)
        logger.info(f"No .ckpt found, saved .nemo to: {best_ckpt_path}")
    else:
        # Load the best checkpoint weights back into the model
        logger.info(f"Loading best checkpoint weights: {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        asr_model.load_state_dict(state_dict, strict=False)

    return best_ckpt_path, asr_model


# ═══════════════════════════════════════════════════════════════════════════
#  Inference & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(
    model_path: str,
    test_manifest: str,
    output_dir: str,
    batch_size: int = 16,
    asr_model=None,
) -> None:
    """
    Run inference on the test set, compute per-sentence WER & CER,
    and save results to CSV + a summary text file.
    """
    logger.info("=" * 70)
    logger.info("  EVALUATION on test set")
    logger.info("=" * 70)

    # ── Load finetuned model ──────────────────────────────────────────
    if asr_model is not None:
        logger.info("Using model already in memory (skipping reload).")
    elif model_path.endswith(".ckpt"):
        # Load pretrained architecture, then apply finetuned weights
        logger.info(f"Loading finetuned model from .ckpt: {model_path}")
        logger.info("Loading pretrained model architecture …")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        logger.info("Loading finetuned weights from .ckpt …")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = asr_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}…")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
        logger.info("Finetuned weights loaded successfully.")
    else:
        # Try loading as .nemo file
        logger.info(f"Loading finetuned model from: {model_path}")
        asr_model = nemo_asr.models.ASRModel.restore_from(model_path)

    asr_model.eval()
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()

    # ── Read test manifest to get audio paths and reference texts ──────
    audio_paths = []
    references = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
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

    # transcribe may return a list of strings or list of Hypothesis objects
    if isinstance(hypotheses, (list, tuple)):
        # For some NeMo versions, transcribe returns (texts, ...) tuple
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

    results = []
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

    all_wers = []
    all_cers = []

    for ref, hyp, audio_fp in zip(references, predictions, audio_paths):
        # Handle empty references or predictions
        ref_clean = ref.strip() if ref else ""
        hyp_clean = hyp.strip() if hyp else ""

        if ref_clean == "" and hyp_clean == "":
            sent_wer = 0.0
            sent_cer = 0.0
        elif ref_clean == "":
            # If reference is empty but hypothesis is not, WER/CER = infinity
            # We'll cap it at 1.0 for practical purposes
            sent_wer = 1.0
            sent_cer = 1.0
        else:
            try:
                sent_wer = jiwer.wer(
                    ref_clean,
                    hyp_clean,
                    truth_transform=wer_transform,
                    hypothesis_transform=wer_transform,
                )
            except Exception:
                sent_wer = 1.0

            try:
                sent_cer = jiwer.cer(
                    ref_clean,
                    hyp_clean,
                    truth_transform=cer_transform,
                    hypothesis_transform=cer_transform,
                )
            except Exception:
                sent_cer = 1.0

        all_wers.append(sent_wer)
        all_cers.append(sent_cer)

        results.append(
            {
                "audio_filepath": audio_fp,
                "reference": ref_clean,
                "prediction": hyp_clean,
                "wer": round(sent_wer, 6),
                "cer": round(sent_cer, 6),
            }
        )

    # ── Save per-sentence results to CSV ──────────────────────────────
    results_csv_path = os.path.join(output_dir, "test_results.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved per-sentence results to: {results_csv_path}")

    # ── Compute aggregate metrics ─────────────────────────────────────
    median_wer = float(np.median(all_wers))
    median_cer = float(np.median(all_cers))
    mean_wer = float(np.mean(all_wers))
    mean_cer = float(np.mean(all_cers))

    # Also compute corpus-level WER (not just median of per-sentence WERs)
    try:
        corpus_wer = jiwer.wer(
            references,
            predictions,
            truth_transform=wer_transform,
            hypothesis_transform=wer_transform,
        )
    except Exception:
        corpus_wer = float("nan")

    try:
        corpus_cer = jiwer.cer(
            references,
            predictions,
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
        f"Model:              {model_path}",
        f"Test manifest:      {test_manifest}",
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Finetune parakeet-tdt-0.6b-v3 and evaluate on test set."
    )

    # Data paths
    parser.add_argument(
        "--train_csv", type=str, required=True, help="Path to training CSV."
    )
    parser.add_argument(
        "--dev_csv", type=str, required=True, help="Path to dev/validation CSV."
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to test CSV."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for all outputs.",
    )

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)

    # Evaluation
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )

    # Flags
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only run evaluation (requires --pretrained_model).",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to a .nemo model file to use for evaluation (skips training).",
    )

    args = parser.parse_args()

    # ── Create output directory ────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Convert CSVs to NeMo manifests ─────────────────────────────────
    manifest_dir = os.path.join(args.output_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    train_manifest = csv_to_nemo_manifest(
        args.train_csv, os.path.join(manifest_dir, "train_manifest.json")
    )
    dev_manifest = csv_to_nemo_manifest(
        args.dev_csv, os.path.join(manifest_dir, "dev_manifest.json")
    )
    test_manifest = csv_to_nemo_manifest(
        args.test_csv, os.path.join(manifest_dir, "test_manifest.json")
    )

    # ── Finetune or use existing model ─────────────────────────────────
    asr_model = None
    if args.skip_training:
        if args.pretrained_model is None:
            logger.error("--skip_training requires --pretrained_model.")
            sys.exit(1)
        model_path = args.pretrained_model
        logger.info(f"Skipping training. Using model: {model_path}")
    else:
        model_path, asr_model = finetune(
            train_manifest=train_manifest,
            dev_manifest=dev_manifest,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_gpus=args.num_gpus,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )

    # ── Evaluate ───────────────────────────────────────────────────────
    evaluate(
        model_path=model_path,
        test_manifest=test_manifest,
        output_dir=args.output_dir,
        batch_size=args.eval_batch_size,
        asr_model=asr_model,
    )

    logger.info("All done!")


if __name__ == "__main__":
    main()
