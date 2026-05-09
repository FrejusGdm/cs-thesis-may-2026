#!/usr/bin/env python3
from __future__ import annotations
"""
C1: Whisper zero-shot inference on Adja speech.

Runs Whisper (no fine-tuning) on the Adja test/dev sets under multiple
language settings and reports WER/CER. This is a zero-shot baseline:
Whisper has almost certainly never seen Adja during training.

Usage:
    python experiments/asr/C1_whisper_zeroshot/infer.py \
        --data-dir data/adja_asr \
        --output-dir experiments/asr/C1_whisper_zeroshot/outputs

    python experiments/asr/C1_whisper_zeroshot/infer.py \
        --data-dir data/adja_asr \
        --output-dir experiments/asr/C1_whisper_zeroshot/outputs \
        --dry-run
"""

import sys

sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the shared metrics module to sys.path so we can import it regardless
# of where this script is invoked from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_DIR = _SCRIPT_DIR.parent / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from metrics import compute_wer, compute_cer  # noqa: E402

import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # noqa: E402


# ---------------------------------------------------------------------------
# Model size -> HuggingFace model ID mapping
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

# Language settings to try.  Each entry is (label, language_kwarg) where
# language_kwarg is passed to processor / forced_decoder_ids.
# None means "let Whisper auto-detect".
LANGUAGE_SETTINGS = [
    ("auto", None),
    ("fr", "fr"),
    ("transcribe_forced", "__transcribe_forced__"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(tsv_path: str) -> list[dict]:
    """Load a TSV manifest into a list of dicts.

    Expected columns: id, audio_path, text, duration_sec, sampling_rate
    """
    entries = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            entries.append(row)
    return entries


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    entries: list[dict],
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    language: str | None,
    device: torch.device,
) -> list[str]:
    """Run Whisper inference on a list of manifest entries.

    Args:
        entries: list of dicts with at least 'audio_path' and 'text' keys.
        processor: WhisperProcessor instance.
        model: WhisperForConditionalGeneration instance.
        language: language code (e.g. "fr"), None for auto-detect, or
                  "__transcribe_forced__" for forced transcription without
                  a language token.
        device: torch device.

    Returns:
        List of hypothesis strings, one per entry.
    """
    hypotheses = []

    for i, entry in enumerate(entries):
        audio_path = entry["audio_path"]

        # Load audio
        audio_array, sr = sf.read(audio_path, dtype="float32")

        # Resample to 16kHz if necessary (Whisper expects 16kHz)
        if sr != 16000:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            except ImportError:
                raise RuntimeError(
                    f"Audio at {audio_path} has sample rate {sr} but Whisper "
                    f"expects 16000. Install librosa for resampling: "
                    f"pip install librosa"
                )

        # Prepare input features
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)

        # Build generation kwargs based on language setting
        gen_kwargs = {}

        if language == "__transcribe_forced__":
            # Force transcription task, no language token
            gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                task="transcribe",
                no_timestamps=True,
            )
        elif language is not None:
            # Force a specific language
            gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
                no_timestamps=True,
            )
        else:
            # Auto-detect: let Whisper choose language
            gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                task="transcribe",
                no_timestamps=True,
            )

        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                **gen_kwargs,
            )

        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        hyp = transcription[0].strip()
        hypotheses.append(hyp)

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"  [{i + 1}/{len(entries)}] processed")

    return hypotheses


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------

def save_decode_samples(
    references: list[str],
    hypotheses: list[str],
    ids: list[str],
    output_path: str,
    n_samples: int = 20,
    seed: int = 42,
):
    """Save random ref/hyp pairs for qualitative inspection."""
    n = min(n_samples, len(references))
    rng = random.Random(seed)
    indices = rng.sample(range(len(references)), n)
    indices.sort()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Decode samples ({n} random pairs)\n")
        f.write(f"# Format: ID | REF | HYP\n\n")
        for idx in indices:
            f.write(f"[{ids[idx]}]\n")
            f.write(f"  REF: {references[idx]}\n")
            f.write(f"  HYP: {hypotheses[idx]}\n\n")

    print(f"  Saved {n} decode samples to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="C1: Whisper zero-shot ASR inference on Adja"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory (contains manifests/ subdirectory with test.tsv, dev.tsv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write results",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        choices=list(MODEL_MAP.keys()),
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--model-cache",
        type=str,
        default=None,
        help="Local cache directory for pre-downloaded model weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (currently processes one utterance at a time; reserved for future use)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 utterances per split",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve model
    # ------------------------------------------------------------------
    model_id = MODEL_MAP[args.model_size]
    print(f"Model: {model_id}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    # ------------------------------------------------------------------
    # Load model and processor
    # ------------------------------------------------------------------
    load_kwargs = {}
    if args.model_cache:
        cache_dir = Path(args.model_cache).resolve()
        print(f"Using model cache: {cache_dir}")
        load_kwargs["cache_dir"] = str(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_id, **load_kwargs)

    print("Loading model...")
    t0_load = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model = model.to(device)
    model.eval()
    load_time = time.time() - t0_load
    print(f"Model loaded in {load_time:.1f}s")

    # ------------------------------------------------------------------
    # Discover splits to run
    # ------------------------------------------------------------------
    manifest_dir = data_dir / "manifests"
    splits_to_run = []
    for split_name in ["test", "dev"]:
        tsv_path = manifest_dir / f"{split_name}.tsv"
        if tsv_path.exists():
            splits_to_run.append((split_name, str(tsv_path)))
            print(f"Found {split_name} manifest: {tsv_path}")
        else:
            print(f"WARNING: {tsv_path} not found, skipping {split_name}")

    if not splits_to_run:
        print("ERROR: No manifest files found. Run data_prep.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run inference for each split x language setting
    # ------------------------------------------------------------------
    all_metrics = {}
    timing_info = {"model_load_sec": round(load_time, 2), "runs": {}}

    for split_name, tsv_path in splits_to_run:
        print(f"\n{'=' * 60}")
        print(f"Split: {split_name}")
        print(f"{'=' * 60}")

        entries = load_manifest(tsv_path)
        print(f"Loaded {len(entries)} utterances from {tsv_path}")

        if args.dry_run:
            entries = entries[:5]
            print(f"DRY RUN: truncated to {len(entries)} utterances")

        references = [e["text"] for e in entries]
        utt_ids = [e["id"] for e in entries]

        for lang_label, lang_value in LANGUAGE_SETTINGS:
            run_key = f"{split_name}_{lang_label}"
            run_output_dir = output_dir / run_key
            run_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Language setting: {lang_label} (value={lang_value}) ---")

            t0 = time.time()
            hypotheses = run_inference(entries, processor, model, lang_value, device)
            inference_time = time.time() - t0

            total_audio_sec = sum(float(e.get("duration_sec", 0)) for e in entries)
            rtf = inference_time / total_audio_sec if total_audio_sec > 0 else 0.0

            print(f"  Inference time: {inference_time:.1f}s")
            print(f"  Audio duration: {total_audio_sec:.1f}s")
            print(f"  RTF: {rtf:.2f}")

            # Compute metrics
            wer_results = compute_wer(references, hypotheses)
            cer_results = compute_cer(references, hypotheses)

            metrics = {
                "split": split_name,
                "language_setting": lang_label,
                "model": model_id,
                "wer": wer_results["wer"],
                "cer": cer_results["cer"],
                "wer_details": wer_results,
                "cer_details": cer_results,
                "num_utterances": len(entries),
                "total_audio_sec": round(total_audio_sec, 2),
                "inference_time_sec": round(inference_time, 2),
                "rtf": round(rtf, 3),
                "dry_run": args.dry_run,
            }

            all_metrics[run_key] = metrics

            # Print summary
            print(f"  WER: {wer_results['wer']}%")
            print(f"  CER: {cer_results['cer']}%")
            print(
                f"  WER breakdown: S={wer_results['substitutions']} "
                f"I={wer_results['insertions']} D={wer_results['deletions']}"
            )

            # Save per-run metrics
            metrics_path = run_output_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"  Saved metrics to {metrics_path}")

            # Save decode samples
            samples_path = run_output_dir / "decode_samples.txt"
            save_decode_samples(
                references, hypotheses, utt_ids, str(samples_path), n_samples=20
            )

            # Save all hypotheses
            all_hyps_path = run_output_dir / "hypotheses.txt"
            with open(all_hyps_path, "w", encoding="utf-8") as f:
                for hyp in hypotheses:
                    f.write(hyp + "\n")

            # Save all references (for convenience)
            all_refs_path = run_output_dir / "references.txt"
            with open(all_refs_path, "w", encoding="utf-8") as f:
                for ref in references:
                    f.write(ref + "\n")

            timing_info["runs"][run_key] = {
                "inference_time_sec": round(inference_time, 2),
                "total_audio_sec": round(total_audio_sec, 2),
                "rtf": round(rtf, 3),
                "num_utterances": len(entries),
            }

    # ------------------------------------------------------------------
    # Save aggregated results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    summary_lines = []
    for run_key, m in all_metrics.items():
        line = f"{run_key:30s}  WER={m['wer']:7.2f}%  CER={m['cer']:7.2f}%"
        summary_lines.append(line)
        print(line)

    # Save combined metrics
    combined_path = output_dir / "metrics.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to {combined_path}")

    # Save timing
    timing_path = output_dir / "timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing info saved to {timing_path}")

    # Save a combined decode_samples.txt at the top level
    top_samples_path = output_dir / "decode_samples.txt"
    with open(top_samples_path, "w", encoding="utf-8") as f:
        for run_key in all_metrics:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Run: {run_key}\n")
            f.write(f"{'=' * 60}\n\n")
            run_samples = output_dir / run_key / "decode_samples.txt"
            if run_samples.exists():
                f.write(run_samples.read_text(encoding="utf-8"))
                f.write("\n")
    print(f"Combined decode samples saved to {top_samples_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
