#!/usr/bin/env python3
from __future__ import annotations
"""
B1 BiLSTM-CTC Training Script for Adja ASR.

Self-contained PyTorch training loop (no Trainer abstraction).
Loads data from TSV manifests, trains BiLSTM-CTC model with SpecAugment
and speed perturbation, evaluates on dev set with CER, applies early
stopping, and saves checkpoints + metrics.

Usage:
    python train.py --config conf/train.yaml --data-dir ../../data --output-dir ./output
    python train.py --config conf/train.yaml --data-dir ../../data --output-dir ./output --dry-run

Manifest format (TSV):
    id  audio_path  text  duration_sec  sampling_rate
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml

# Import model from same directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from model import BiLSTMCTCModel

# Import shared metrics
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))
from metrics import compute_cer


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def load_and_remap_vocab(vocab_path: str) -> tuple[dict, dict]:
    """Load char vocab from JSON and remap so CTC blank is at index 0.

    The data_prep.py puts <blank> at index 4. For CTC, blank MUST be at 0.
    We remap: move <blank> to 0, shift everything else.

    Returns:
        char2idx: character -> index mapping (blank at 0)
        idx2char: index -> character mapping
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        original_vocab = json.load(f)

    # Find the original blank index
    blank_key = "<blank>"
    if blank_key not in original_vocab:
        raise ValueError(f"<blank> not found in vocabulary at {vocab_path}")

    original_blank_idx = original_vocab[blank_key]

    # Build new mapping with blank at 0
    # Strategy: assign blank->0, then assign all others in original order
    # skipping the blank slot
    char2idx = {}
    char2idx[blank_key] = 0

    new_idx = 1
    for char, old_idx in sorted(original_vocab.items(), key=lambda x: x[1]):
        if char == blank_key:
            continue
        char2idx[char] = new_idx
        new_idx += 1

    idx2char = {v: k for k, v in char2idx.items()}

    return char2idx, idx2char


def text_to_indices(text: str, char2idx: dict) -> list[int]:
    """Convert text string to list of character indices.

    Unknown characters map to <unk>.
    Skips <blank>, <pad>, <sos>, <eos> tokens (those are special).
    """
    unk_idx = char2idx.get("<unk>", 1)
    indices = []
    for ch in text:
        if ch in char2idx:
            indices.append(char2idx[ch])
        else:
            indices.append(unk_idx)
    return indices


def indices_to_text(indices: list[int], idx2char: dict) -> str:
    """Convert list of token indices back to text string.

    Skips special tokens (<blank>, <pad>, <sos>, <eos>, <unk>).
    """
    special = {"<blank>", "<pad>", "<sos>", "<eos>", "<unk>"}
    chars = []
    for idx in indices:
        ch = idx2char.get(idx, "")
        if ch not in special:
            chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdjaASRDataset(torch.utils.data.Dataset):
    """Load audio + text pairs from a TSV manifest.

    Manifest columns: id, audio_path, text, duration_sec, sampling_rate

    Optionally applies speed perturbation at load time.
    """

    def __init__(
        self,
        manifest_path: str,
        char2idx: dict,
        speed_factors: list[float] | None = None,
        max_samples: int | None = None,
    ):
        self.char2idx = char2idx
        self.speed_factors = speed_factors
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            assert header == ["id", "audio_path", "text", "duration_sec", "sampling_rate"], \
                f"Unexpected manifest header: {header}"

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                self.samples.append({
                    "id": parts[0],
                    "audio_path": parts[1],
                    "text": parts[2],
                    "duration": float(parts[3]),
                    "sr": int(parts[4]),
                })

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load audio
        waveform, sr = torchaudio.load(sample["audio_path"])

        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # (samples,)

        # Speed perturbation (only if factors provided)
        if self.speed_factors is not None and len(self.speed_factors) > 1:
            factor = random.choice(self.speed_factors)
            if factor != 1.0:
                waveform = self._speed_perturb(waveform, sr, factor)

        # Text to indices
        token_ids = text_to_indices(sample["text"], self.char2idx)

        return {
            "id": sample["id"],
            "waveform": waveform,
            "wav_length": waveform.size(0),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_length": len(token_ids),
            "text": sample["text"],
        }

    @staticmethod
    def _speed_perturb(
        waveform: torch.Tensor, sample_rate: int, factor: float
    ) -> torch.Tensor:
        """Apply speed perturbation by resampling.

        Speed factor > 1.0 = faster (shorter), < 1.0 = slower (longer).
        Uses torchaudio.functional.resample (no SoX dependency).
        """
        # Resample: treat original audio as if sampled at sr*factor,
        # then resample back to sr. This changes speed without pitch correction.
        new_sr = int(sample_rate * factor)
        augmented = torchaudio.functional.resample(waveform, new_sr, sample_rate)
        return augmented


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length audio + text into padded batches.

    Returns:
        waveforms: (batch, max_samples) zero-padded
        wav_lengths: (batch,) original sample counts
        targets: (batch, max_label_len) zero-padded token indices
        target_lengths: (batch,) original label lengths
        ids: list of utterance IDs
        texts: list of reference texts
    """
    # Sort by waveform length descending (helps pack_padded_sequence)
    batch = sorted(batch, key=lambda x: x["wav_length"], reverse=True)

    wav_lengths = torch.tensor([s["wav_length"] for s in batch], dtype=torch.long)
    target_lengths = torch.tensor([s["token_length"] for s in batch], dtype=torch.long)

    # Pad waveforms
    max_wav_len = wav_lengths[0].item()
    waveforms = torch.zeros(len(batch), max_wav_len)
    for i, s in enumerate(batch):
        waveforms[i, : s["wav_length"]] = s["waveform"]

    # Pad targets
    max_target_len = target_lengths.max().item()
    targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, s in enumerate(batch):
        targets[i, : s["token_length"]] = s["token_ids"]

    return {
        "waveforms": waveforms,
        "wav_lengths": wav_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "ids": [s["id"] for s in batch],
        "texts": [s["text"] for s in batch],
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def greedy_decode_to_text(
    model: BiLSTMCTCModel,
    log_probs: torch.Tensor,
    feat_lengths: torch.Tensor,
    idx2char: dict,
) -> list[str]:
    """Run greedy CTC decoding and convert to text."""
    decoded_indices = model.greedy_decode(log_probs, feat_lengths)
    return [indices_to_text(seq, idx2char) for seq in decoded_indices]


def compute_epoch_cer(references: list[str], hypotheses: list[str]) -> float:
    """Compute CER using the shared metrics module. Returns CER as a percentage."""
    if not references:
        return 100.0
    result = compute_cer(references, hypotheses)
    return result["cer"]


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BiLSTMCTCModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    log_interval: int,
    epoch: int,
    dry_run: bool = False,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        waveforms = batch["waveforms"].to(device)
        wav_lengths = batch["wav_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()

        log_probs, feat_lengths = model(waveforms, wav_lengths)

        # Safety: ensure feat_lengths >= target_lengths for CTC
        # If any sample violates this, skip the batch
        if (feat_lengths < target_lengths).any():
            print(f"  [WARNING] Skipping batch {batch_idx}: feat_lengths < target_lengths")
            continue

        loss = model.compute_loss(log_probs, feat_lengths, targets, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARNING] Skipping batch {batch_idx}: loss is nan/inf")
            continue

        loss.backward()

        # Gradient clipping
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / num_batches
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")

        if dry_run and batch_idx >= 1:
            break

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: BiLSTMCTCModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    idx2char: dict,
    decode_samples: int = 10,
    dry_run: bool = False,
) -> tuple[float, float, list[tuple[str, str, str]]]:
    """Evaluate on dev/test set.

    Returns:
        avg_loss: average CTC loss
        cer: character error rate (percentage)
        samples: list of (id, reference, hypothesis) for display
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_refs = []
    all_hyps = []
    sample_triples = []  # (id, ref, hyp)

    for batch_idx, batch in enumerate(dataloader):
        waveforms = batch["waveforms"].to(device)
        wav_lengths = batch["wav_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs, feat_lengths = model(waveforms, wav_lengths)

        # Skip batches where CTC constraint is violated
        if (feat_lengths < target_lengths).any():
            continue

        loss = model.compute_loss(log_probs, feat_lengths, targets, target_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

        # Greedy decode
        hyps = greedy_decode_to_text(model, log_probs, feat_lengths, idx2char)
        refs = batch["texts"]
        ids = batch["ids"]

        all_refs.extend(refs)
        all_hyps.extend(hyps)

        for uid, ref, hyp in zip(ids, refs, hyps):
            if len(sample_triples) < decode_samples:
                sample_triples.append((uid, ref, hyp))

        if dry_run and batch_idx >= 1:
            break

    avg_loss = total_loss / max(num_batches, 1)
    cer = compute_epoch_cer(all_refs, all_hyps)

    return avg_loss, cer, sample_triples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B1 BiLSTM-CTC ASR Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory (contains manifests/, char_vocab.json)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps on 5 samples for sanity check")
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Paths ---
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = data_dir / "manifests" / "train.tsv"
    dev_manifest = data_dir / "manifests" / "dev.tsv"
    vocab_path = data_dir / "char_vocab.json"

    for p in [train_manifest, dev_manifest, vocab_path]:
        if not p.exists():
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    # --- Seed ---
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    set_seed(seed)
    print(f"Random seed: {seed}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Vocabulary ---
    print(f"Loading vocabulary from {vocab_path}")
    char2idx, idx2char = load_and_remap_vocab(str(vocab_path))
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size} (blank at index 0)")

    # Save remapped vocab for reference
    with open(output_dir / "vocab_remapped.json", "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)

    # --- Datasets ---
    dry_run = args.dry_run
    max_samples = 5 if dry_run else None

    speed_factors = cfg["speed_perturb"]["factors"] if cfg["speed_perturb"]["enabled"] else None

    print(f"Loading training data from {train_manifest}")
    train_dataset = AdjaASRDataset(
        manifest_path=str(train_manifest),
        char2idx=char2idx,
        speed_factors=speed_factors,
        max_samples=max_samples,
    )
    print(f"  Train samples: {len(train_dataset)}")

    print(f"Loading dev data from {dev_manifest}")
    dev_dataset = AdjaASRDataset(
        manifest_path=str(dev_manifest),
        char2idx=char2idx,
        speed_factors=None,  # No augmentation for dev
        max_samples=max_samples,
    )
    print(f"  Dev samples: {len(dev_dataset)}")

    batch_size = cfg["training"]["batch_size"]
    if dry_run:
        batch_size = min(batch_size, 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --- Model ---
    audio_cfg = cfg["audio"]
    encoder_cfg = cfg["encoder"]
    sa_cfg = cfg["spec_augment"]

    model = BiLSTMCTCModel(
        vocab_size=vocab_size,
        n_mels=audio_cfg["n_mels"],
        encoder_dim=encoder_cfg["hidden_dim"],
        encoder_layers=encoder_cfg["num_layers"],
        dropout=encoder_cfg["dropout"],
        blank_index=cfg["ctc"]["blank_index"],
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        f_min=audio_cfg["f_min"],
        f_max=audio_cfg["f_max"],
        spec_augment=sa_cfg["enabled"],
        freq_mask_param=sa_cfg["freq_mask_param"],
        time_mask_param=sa_cfg["time_mask_param"],
        n_freq_masks=sa_cfg["n_freq_masks"],
        n_time_masks=sa_cfg["n_time_masks"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    # --- Optimizer ---
    train_cfg = cfg["training"]
    if train_cfg["optimizer"] == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            print("Using Adafactor optimizer")
        except ImportError:
            print("Adafactor not available, falling back to Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        print(f"Using Adam optimizer (lr={train_cfg['learning_rate']})")

    # --- Training state ---
    max_epochs = 3 if dry_run else train_cfg["max_epochs"]
    patience = cfg["early_stopping"]["patience"]
    log_interval = cfg["logging"]["log_interval"]
    decode_samples = cfg["logging"]["decode_samples"]
    grad_clip = train_cfg["grad_clip_norm"]

    best_cer = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "decode_samples.txt"
    timing_path = output_dir / "timing.json"

    # Save config to output dir for reproducibility
    with open(output_dir / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Starting training: {max_epochs} epochs, patience={patience}")
    print(f"Output: {output_dir}")
    if dry_run:
        print("*** DRY RUN MODE ***")
    print(f"{'='*60}\n")

    total_start = time.time()

    # --- Training loop ---
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip,
            log_interval=log_interval,
            epoch=epoch,
            dry_run=dry_run,
        )

        # Evaluate
        dev_loss, dev_cer, dev_samples = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            idx2char=idx2char,
            decode_samples=decode_samples,
            dry_run=dry_run,
        )

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch}/{max_epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Dev loss:   {dev_loss:.4f}")
        print(f"  Dev CER:    {dev_cer:.2f}%")

        # Decode samples
        if dev_samples:
            print(f"\n  --- Decode samples (epoch {epoch}) ---")
            for uid, ref, hyp in dev_samples[:5]:
                print(f"  [{uid}]")
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
            print()

        # History
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "dev_loss": round(dev_loss, 6),
            "dev_cer": round(dev_cer, 2),
            "epoch_time_sec": round(epoch_time, 1),
        }
        history.append(epoch_record)

        # Early stopping check
        if dev_cer < best_cer:
            best_cer = dev_cer
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_cer": dev_cer,
                "dev_loss": dev_loss,
                "vocab_size": vocab_size,
                "config": cfg,
            }, checkpoint_path)
            print(f"  ** New best CER: {best_cer:.2f}% (saved to {checkpoint_path})")

            # Save decode samples for best epoch
            with open(samples_path, "w", encoding="utf-8") as f:
                f.write(f"# Best decode samples (epoch {epoch}, CER={dev_cer:.2f}%)\n\n")
                for uid, ref, hyp in dev_samples:
                    f.write(f"[{uid}]\n")
                    f.write(f"  REF: {ref}\n")
                    f.write(f"  HYP: {hyp}\n\n")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs (best: {best_cer:.2f}% at epoch {best_epoch})")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        # Save metrics after each epoch (so partial results survive crashes)
        total_elapsed = time.time() - total_start
        metrics = {
            "experiment": "B1_bilstm_ctc",
            "best_epoch": best_epoch,
            "best_dev_cer": round(best_cer, 2),
            "final_epoch": epoch,
            "total_epochs_run": epoch,
            "seed": seed,
            "vocab_size": vocab_size,
            "num_params": num_params,
            "history": history,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # --- Done ---
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best dev CER: {best_cer:.2f}% at epoch {best_epoch}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*60}")

    # Save timing
    timing = {
        "total_time_sec": round(total_time, 1),
        "total_time_min": round(total_time / 60, 2),
        "total_epochs_run": len(history),
        "avg_epoch_time_sec": round(np.mean([h["epoch_time_sec"] for h in history]), 1) if history else 0,
        "device": str(device),
        "dry_run": dry_run,
    }
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    # Final metrics save
    metrics["timing"] = timing
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
