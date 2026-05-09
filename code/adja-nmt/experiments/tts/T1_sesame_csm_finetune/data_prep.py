#!/usr/bin/env python3
from __future__ import annotations
"""
Data preparation for Sesame CSM fine-tuning with Unsloth.

Loads Adja audio from either:
  1. Local WAV files + TSV manifests (from ASR data_prep.py output)
  2. HuggingFace dataset directly (JosueG/adja-tts-orpheus)

Resamples audio from 48kHz -> 24kHz (CSM's expected sample rate),
applies NFC normalization, and saves as a HuggingFace Dataset
ready for Unsloth's TTS fine-tuning pipeline.

References:
    - Sesame CSM: https://github.com/SesameAILabs/csm
    - CSM uses Mimi codec at 24kHz: https://huggingface.co/kyutai/mimi
    - Unsloth TTS fine-tuning: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning

Usage:
    # From local manifests (preferred — avoids re-downloading):
    python data_prep.py --source local --data-dir ../../data --output-dir ./prepared_data

    # From HuggingFace directly:
    python data_prep.py --source hf --hf-token $HF_TOKEN --output-dir ./prepared_data

    # Dry run (5 samples per split):
    python data_prep.py --source local --data-dir ../../data --output-dir ./prepared_data --dry-run
"""

import argparse
import os
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import soundfile as sf

# CSM expects 24kHz audio (Mimi codec sample rate)
TARGET_SR = 24000


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation.

    For production quality, librosa.resample is better, but this avoids
    the heavy librosa dependency. The quality difference is minimal
    for speech at these sample rates.
    """
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def load_from_manifest(manifest_path: Path, dry_run: bool = False):
    """Load samples from a TSV manifest file."""
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f):
            if dry_run and i >= 5:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            utt_id, audio_path, text, duration, sr = parts[:5]
            samples.append({
                "id": utt_id,
                "audio_path": audio_path,
                "text": text,
                "duration": float(duration),
                "sampling_rate": int(sr),
            })
    return samples


def process_samples(samples: list[dict], output_dir: Path, split_name: str):
    """Process samples: resample to 24kHz, normalize text, save WAVs."""
    wav_dir = output_dir / "wavs_24k" / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    for sample in samples:
        # Load audio
        audio, sr = sf.read(sample["audio_path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        audio = audio.astype(np.float32)

        # Resample to 24kHz
        audio_24k = resample_audio(audio, sr, TARGET_SR)

        # NFC normalize text — critical for Adja tone marks (ɛ, ɔ, ŋ, ɖ, é, è)
        text = unicodedata.normalize("NFC", sample["text"].strip())
        text = " ".join(text.split())

        # Save resampled WAV
        out_path = wav_dir / f"{sample['id']}.wav"
        sf.write(str(out_path), audio_24k, TARGET_SR)

        processed.append({
            "id": sample["id"],
            "audio_path": str(out_path),
            "text": text,
            "duration_sec": len(audio_24k) / TARGET_SR,
            "sampling_rate": TARGET_SR,
        })

    return processed


def save_manifest(samples: list[dict], output_path: Path):
    """Save processed samples as a TSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "id\taudio_path\ttext\tduration_sec\tsampling_rate"
    rows = [
        f"{s['id']}\t{s['audio_path']}\t{s['text']}\t{s['duration_sec']:.3f}\t{s['sampling_rate']}"
        for s in samples
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def save_as_hf_dataset(all_splits: dict, output_dir: Path):
    """Save as HuggingFace Dataset for direct use with Unsloth."""
    try:
        from datasets import Audio, Dataset, DatasetDict
    except ImportError:
        print("WARNING: `datasets` not installed. Skipping HF dataset export.")
        print("  Install with: pip install datasets")
        return

    ds_dict = {}
    for split_name, samples in all_splits.items():
        ds_dict[split_name] = Dataset.from_dict({
            "audio": [s["audio_path"] for s in samples],
            "text": [s["text"] for s in samples],
        }).cast_column("audio", Audio(sampling_rate=TARGET_SR))

    hf_ds = DatasetDict(ds_dict)
    save_path = output_dir / "hf_dataset"
    hf_ds.save_to_disk(str(save_path))
    print(f"  HuggingFace Dataset saved to {save_path}")
    return hf_ds


def main():
    parser = argparse.ArgumentParser(description="Prepare Adja data for Sesame CSM fine-tuning")
    parser.add_argument("--source", choices=["local", "hf"], default="local",
                        help="Data source: 'local' (from ASR manifests) or 'hf' (download from HuggingFace)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data/ directory with manifests/ and wavs/ (for --source local)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (for --source hf)")
    parser.add_argument("--dataset", type=str, default="JosueG/adja-tts-orpheus",
                        help="HuggingFace dataset ID")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for resampled data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 5 samples per split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_splits = {}

    if args.source == "local":
        # Load from existing ASR manifests
        data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent.parent / "data"
        manifest_dir = data_dir / "manifests"

        if not manifest_dir.exists():
            print(f"ERROR: Manifest directory not found: {manifest_dir}")
            print("Run experiments/asr/shared/data_prep.py first, or use --source hf")
            sys.exit(1)

        for split_name in ["train", "dev", "test"]:
            manifest_path = manifest_dir / f"{split_name}.tsv"
            if not manifest_path.exists():
                print(f"WARNING: {manifest_path} not found, skipping {split_name}")
                continue

            print(f"\nProcessing {split_name} split...")
            samples = load_from_manifest(manifest_path, dry_run=args.dry_run)
            print(f"  Loaded {len(samples)} samples from manifest")

            processed = process_samples(samples, output_dir, split_name)
            all_splits[split_name] = processed

            # Save 24kHz manifest
            out_manifest = output_dir / "manifests" / f"{split_name}.tsv"
            save_manifest(processed, out_manifest)
            print(f"  Saved {len(processed)} resampled samples")

            total_dur = sum(s["duration_sec"] for s in processed)
            print(f"  Total duration: {total_dur / 60:.1f} min ({total_dur / 3600:.2f} h)")

    elif args.source == "hf":
        # Load directly from HuggingFace
        from datasets import load_dataset

        token = args.hf_token or os.environ.get("HF_TOKEN")
        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset(args.dataset, token=token, split="train")

        # Create splits (same logic as ASR data_prep.py)
        split1 = ds.train_test_split(test_size=0.1, seed=42)
        split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=42)
        splits = {"train": split2["train"], "dev": split2["test"], "test": split1["test"]}

        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
            n_samples = min(5, len(split_data)) if args.dry_run else len(split_data)

            wav_dir = output_dir / "wavs_24k" / split_name
            wav_dir.mkdir(parents=True, exist_ok=True)

            processed = []
            for i in range(n_samples):
                sample = split_data[i]
                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]

                audio_24k = resample_audio(audio, sr, TARGET_SR)
                text = unicodedata.normalize("NFC", sample["text"].strip())
                text = " ".join(text.split())

                utt_id = f"{split_name}_{i:05d}"
                out_path = wav_dir / f"{utt_id}.wav"
                sf.write(str(out_path), audio_24k, TARGET_SR)

                processed.append({
                    "id": utt_id,
                    "audio_path": str(out_path),
                    "text": text,
                    "duration_sec": len(audio_24k) / TARGET_SR,
                    "sampling_rate": TARGET_SR,
                })

            all_splits[split_name] = processed
            out_manifest = output_dir / "manifests" / f"{split_name}.tsv"
            save_manifest(processed, out_manifest)

            total_dur = sum(s["duration_sec"] for s in processed)
            print(f"  Saved {len(processed)} samples, {total_dur / 60:.1f} min")

    # Also save as HF Dataset for direct Unsloth consumption
    print("\nSaving as HuggingFace Dataset...")
    save_as_hf_dataset(all_splits, output_dir)

    print("\nData preparation complete!")
    if args.dry_run:
        print("(DRY RUN — only processed 5 samples per split)")


if __name__ == "__main__":
    main()
