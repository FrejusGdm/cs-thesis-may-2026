#!/usr/bin/env python3
from __future__ import annotations
"""
Data preparation for Adja ASR experiments.

Loads the HuggingFace dataset (JosueG/adja-tts-orpheus),
creates train/dev/test splits, exports audio as WAV files,
and generates manifest TSV files for ESPnet and HuggingFace training.

Usage:
    python data_prep.py --output-dir /path/to/data --hf-token YOUR_TOKEN
    python data_prep.py --output-dir /path/to/data --hf-token YOUR_TOKEN --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Flush prints immediately (critical for HPC log visibility)
sys.stdout.reconfigure(line_buffering=True)

import unicodedata

import numpy as np
import soundfile as sf
from datasets import load_dataset


def get_char_vocab(texts: list[str]) -> dict:
    """Extract character vocabulary from all transcriptions.

    Important for Adja: includes special characters like ɛ, ɔ, ŋ, ɖ, etc.
    These are critical for tone-aware ASR.
    """
    chars = set()
    for text in texts:
        chars.update(text)

    # Sort for reproducibility, add special tokens.
    # IMPORTANT: <blank> at index 0 for CTC compatibility (PyTorch CTCLoss default).
    vocab = sorted(chars)
    char2idx = {"<blank>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3, "<unk>": 4}
    for i, c in enumerate(vocab, start=len(char2idx)):
        char2idx[c] = i

    return char2idx


def create_splits(dataset, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Create train/dev/test splits from a single-split dataset.

    Uses stratified random split. Since the dataset is small (1.6k),
    we use ~10% dev, ~10% test to keep training data maximal.
    """
    # First split: separate test
    split1 = dataset.train_test_split(test_size=test_ratio, seed=seed)

    # Second split: separate dev from remaining train
    dev_ratio_adjusted = dev_ratio / (1 - test_ratio)
    split2 = split1["train"].train_test_split(test_size=dev_ratio_adjusted, seed=seed)

    return {
        "train": split2["train"],
        "dev": split2["test"],
        "test": split1["test"],
    }


def export_split(split_data, split_name: str, output_dir: Path, dry_run: bool = False):
    """Export a split to WAV files + manifest TSV.

    Manifest format (compatible with ESPnet and easy to convert for HF):
        id\taudio_path\ttext\tduration_sec\tsampling_rate
    """
    split_dir = output_dir / "wavs" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifests" / f"{split_name}.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_samples = min(5, len(split_data)) if dry_run else len(split_data)

    for i in range(n_samples):
        sample = split_data[i]
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        # Apply Unicode NFC normalization — critical for Adja tone marks
        sr = sample["audio"]["sampling_rate"]
        text = unicodedata.normalize("NFC", sample["text"].strip())
        text = " ".join(text.split())  # collapse whitespace

        # Generate unique ID
        utt_id = f"{split_name}_{i:05d}"
        wav_path = split_dir / f"{utt_id}.wav"

        # Write WAV file
        sf.write(str(wav_path), audio_array, sr)

        duration = len(audio_array) / sr
        rows.append(f"{utt_id}\t{wav_path}\t{text}\t{duration:.3f}\t{sr}")

    # Write manifest
    header = "id\taudio_path\ttext\tduration_sec\tsampling_rate"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    return manifest_path, len(rows)


def compute_stats(manifest_path: Path) -> dict:
    """Compute dataset statistics from a manifest."""
    durations = []
    text_lengths = []
    chars = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                durations.append(float(parts[3]))
                text = parts[2]
                text_lengths.append(len(text))
                chars.update(text)

    return {
        "num_utterances": len(durations),
        "total_duration_sec": sum(durations),
        "total_duration_hours": sum(durations) / 3600,
        "mean_duration_sec": np.mean(durations) if durations else 0,
        "min_duration_sec": min(durations) if durations else 0,
        "max_duration_sec": max(durations) if durations else 0,
        "mean_text_length": np.mean(text_lengths) if text_lengths else 0,
        "unique_chars": len(chars),
        "char_list": sorted(chars),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Adja ASR data from HuggingFace")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for WAVs and manifests")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (for private dataset)")
    parser.add_argument("--dataset", type=str, default="JosueG/adja-tts-orpheus", help="HuggingFace dataset ID")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 samples per split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: No HF token provided. Will fail for private datasets.")
        print("Set HF_TOKEN env var or pass --hf-token")

    ds = load_dataset(args.dataset, token=token, split="train")
    print(f"Loaded {len(ds)} samples")

    # Create splits
    print(f"Creating splits (dev={args.dev_ratio}, test={args.test_ratio}, seed={args.seed})")
    splits = create_splits(ds, dev_ratio=args.dev_ratio, test_ratio=args.test_ratio, seed=args.seed)

    for name, data in splits.items():
        print(f"  {name}: {len(data)} samples")

    # Export each split
    for name, data in splits.items():
        print(f"\nExporting {name} split...")
        manifest_path, n = export_split(data, name, output_dir, dry_run=args.dry_run)
        print(f"  Wrote {n} samples to {manifest_path}")

        # Compute and print stats
        stats = compute_stats(manifest_path)
        print(f"  Duration: {stats['total_duration_hours']:.2f}h ({stats['total_duration_sec']:.0f}s)")
        print(f"  Mean utterance: {stats['mean_duration_sec']:.1f}s")
        print(f"  Unique chars: {stats['unique_chars']}")

    # Build and save character vocabulary
    all_texts = [unicodedata.normalize("NFC", s["text"].strip()) for s in ds]
    char_vocab = get_char_vocab(all_texts)
    vocab_path = output_dir / "char_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)
    print(f"\nCharacter vocabulary ({len(char_vocab)} tokens) saved to {vocab_path}")
    print(f"Special chars found: {[c for c in char_vocab if len(c) == 1 and ord(c) > 127]}")

    # Save split info for reproducibility
    split_info = {
        "dataset": args.dataset,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "test_ratio": args.test_ratio,
        "splits": {name: len(data) for name, data in splits.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\nData preparation complete!")
    if args.dry_run:
        print("(DRY RUN — only processed 5 samples per split)")


if __name__ == "__main__":
    main()
