#!/usr/bin/env python3
from __future__ import annotations
"""
C5: Fine-tune wav2vec 2.0 (facebook/wav2vec2-large-960h) on Adja ASR data with CTC.

Control experiment: English-only pretrained model to measure whether multilingual
pretraining (C3: MMS, C4: XLS-R) actually helps for Adja ASR.

Custom training loop (no HuggingFace Trainer).
Loads data from TSV manifests, builds a custom CTC vocabulary for Adja characters,
trains with AdamW + linear warmup, evaluates with greedy CTC decoding.

Usage:
    python train.py --config config.yaml --data-dir /path/to/data --output-dir /path/to/output
    python train.py --config config.yaml --data-dir /path/to/data --output-dir /path/to/output --dry-run
"""

import sys

sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Add shared metrics to path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))
from metrics import compute_cer, compute_wer  # noqa: E402


# ===========================================================================
# CTC Vocabulary
# ===========================================================================
def build_ctc_vocab(char_vocab_path: str) -> tuple[dict, dict]:
    """Build CTC vocabulary from char_vocab.json.

    CTC requires blank at index 0. We remap the character vocabulary
    so that:
        0 -> <blank> (CTC blank)
        1 -> <pad>
        2 -> <unk>
        3+ -> actual characters

    Returns:
        char2idx: character -> index mapping
        idx2char: index -> character mapping
    """
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        original_vocab = json.load(f)

    # Extract just the characters (skip special tokens from original vocab)
    special_in_orig = {"<pad>", "<sos>", "<eos>", "<unk>", "<blank>"}
    chars = sorted([c for c in original_vocab if c not in special_in_orig])

    # Build new CTC-compatible vocab: blank=0, pad=1, unk=2, then chars
    char2idx = {
        "<blank>": 0,
        "<pad>": 1,
        "<unk>": 2,
    }
    for i, c in enumerate(chars, start=len(char2idx)):
        char2idx[c] = i

    # Add space/word boundary explicitly if not present
    if " " not in char2idx:
        char2idx[" "] = len(char2idx)

    idx2char = {v: k for k, v in char2idx.items()}

    return char2idx, idx2char


def encode_text(text: str, char2idx: dict) -> list[int]:
    """Encode text string to list of token indices for CTC."""
    unk_idx = char2idx.get("<unk>", 2)
    return [char2idx.get(c, unk_idx) for c in text]


def decode_ctc_greedy(logits: torch.Tensor, idx2char: dict, blank_id: int = 0) -> list[str]:
    """Greedy CTC decoding: argmax, remove blanks and repeated tokens.

    Args:
        logits: (batch, time, vocab) tensor of logits
        idx2char: index to character mapping
        blank_id: CTC blank token index

    Returns:
        List of decoded strings
    """
    predictions = torch.argmax(logits, dim=-1)  # (batch, time)
    decoded = []

    for pred in predictions:
        tokens = pred.tolist()
        # Remove consecutive duplicates, then remove blanks
        collapsed = []
        prev = None
        for t in tokens:
            if t != prev:
                if t != blank_id:
                    collapsed.append(t)
                prev = t
        text = "".join([idx2char.get(t, "") for t in collapsed])
        decoded.append(text)

    return decoded


# ===========================================================================
# Dataset
# ===========================================================================
class AdjaCTCDataset(Dataset):
    """Loads audio + text from a TSV manifest for CTC-based training."""

    def __init__(self, manifest_path: str, char2idx: dict, max_audio_length_sec: float = 30.0):
        self.char2idx = char2idx
        self.max_audio_length_sec = max_audio_length_sec
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            col = {name: idx for idx, name in enumerate(header)}
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < len(header):
                    continue
                duration = float(parts[col["duration_sec"]])
                if duration > max_audio_length_sec:
                    continue
                self.samples.append({
                    "id": parts[col["id"]],
                    "audio_path": parts[col["audio_path"]],
                    "text": parts[col["text"]],
                    "duration_sec": duration,
                    "sampling_rate": int(parts[col["sampling_rate"]]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample["audio_path"], dtype="float32")

        # Resample if needed (wav2vec2 expects 16kHz)
        if sr != 16000:
            import torchaudio

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
            audio = audio_tensor.squeeze(0).numpy()

        label_ids = encode_text(sample["text"], self.char2idx)

        return {
            "id": sample["id"],
            "audio": torch.from_numpy(audio),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "text": sample["text"],
        }


def collate_ctc(batch):
    """Collate function for CTC training with padding."""
    ids = [item["id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Pad audio to same length
    audios = [item["audio"] for item in batch]
    audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Pad labels to same length
    labels = [item["labels"] for item in batch]
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_values": audios_padded,
        "audio_lengths": audio_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths,
        "ids": ids,
        "texts": texts,
    }


# ===========================================================================
# Learning-rate schedule
# ===========================================================================
def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay to 0."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, dataloader, idx2char, device):
    """Run greedy CTC decoding on a dataset and compute CER/WER."""
    model.eval()
    all_refs = []
    all_hyps = []
    all_ids = []

    for batch in dataloader:
        input_values = batch["input_values"].to(device)

        # Create attention mask from audio lengths
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for i, length in enumerate(batch["audio_lengths"]):
            attention_mask[i, :length] = 1

        outputs = model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, time, vocab)

        decoded = decode_ctc_greedy(logits, idx2char, blank_id=0)
        all_hyps.extend(decoded)
        all_refs.extend(batch["texts"])
        all_ids.extend(batch["ids"])

    model.train()

    cer_result = compute_cer(all_refs, all_hyps)
    wer_result = compute_wer(all_refs, all_hyps)

    return {
        "cer": cer_result["cer"],
        "wer": wer_result["wer"],
        "cer_details": cer_result,
        "wer_details": wer_result,
        "references": all_refs,
        "hypotheses": all_hyps,
        "ids": all_ids,
    }


# ===========================================================================
# Main training function
# ===========================================================================
def train(config: dict, data_dir: str, output_dir: str, seed: int, dry_run: bool):
    # ---- Setup ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build CTC vocabulary ----
    data_path = Path(data_dir)
    char_vocab_path = data_path / "char_vocab.json"
    if not char_vocab_path.exists():
        raise FileNotFoundError(
            f"char_vocab.json not found at {char_vocab_path}. "
            f"Run experiments/asr/shared/data_prep.py first."
        )

    char2idx, idx2char = build_ctc_vocab(str(char_vocab_path))
    vocab_size = len(char2idx)
    print(f"CTC vocabulary: {vocab_size} tokens (blank=0)")

    # Save the CTC vocab for reproducibility
    with open(output_path / "ctc_vocab.json", "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False, indent=2)

    # ---- Load model ----
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

    model_name = config["model_name"]
    print(f"Loading model: {model_name}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,  # CTC head will be re-initialized
        ctc_loss_reduction="mean",
        pad_token_id=char2idx["<pad>"],
        ctc_zero_infinity=True,
    )

    # Freeze feature encoder (CNN layers) if requested
    if config.get("freeze_feature_encoder", True):
        print("Freezing feature encoder (CNN layers)")
        model.freeze_feature_encoder()

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Load data ----
    train_manifest = data_path / "manifests" / "train.tsv"
    dev_manifest = data_path / "manifests" / "dev.tsv"

    max_audio_sec = config.get("max_audio_length_sec", 30.0)
    train_dataset = AdjaCTCDataset(str(train_manifest), char2idx, max_audio_sec)
    dev_dataset = AdjaCTCDataset(str(dev_manifest), char2idx, max_audio_sec)

    if dry_run:
        train_dataset.samples = train_dataset.samples[:5]
        dev_dataset.samples = dev_dataset.samples[:5]

    print(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_ctc,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_ctc,
        drop_last=False,
    )

    # ---- Optimizer & scheduler ----
    max_epochs = 2 if dry_run else config["max_epochs"]
    grad_accum = config["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = config["warmup_steps"]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    max_grad_norm = config.get("max_grad_norm", 1.0)
    log_every = config.get("log_every_n_steps", 10)
    patience = config.get("early_stopping_patience", 5)
    num_decode_samples = config.get("num_decode_samples", 20)

    # CTC loss
    ctc_loss_fn = torch.nn.CTCLoss(blank=char2idx["<blank>"], reduction="mean", zero_infinity=True)

    # ---- Training loop ----
    best_cer = float("inf")
    patience_counter = 0
    global_step = 0
    all_epoch_metrics = []
    timing = {"train_start": time.time(), "epochs": []}

    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Epochs: {max_epochs}, Batch: {batch_size}, GradAccum: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  LR: {config['learning_rate']}, Warmup: {warmup_steps}")
    print(f"  Early stopping patience: {patience}")
    print(f"{'='*60}\n")

    max_train_steps = 2 if dry_run else None

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if max_train_steps is not None and batch_idx >= max_train_steps:
                break

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            # Create attention mask
            attention_mask = torch.zeros_like(input_values, dtype=torch.long)
            for i, length in enumerate(audio_lengths):
                attention_mask[i, :length] = 1

            # Forward pass
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, time, vocab)

            # Compute CTC loss manually
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab) for CTC

            # Compute output lengths (model downsamples audio)
            with torch.no_grad():
                output_lengths = model._get_feat_extract_output_lengths(audio_lengths)

            # Filter valid labels (remove padding -100)
            target_list = []
            target_length_list = []
            for i in range(labels.shape[0]):
                valid = labels[i][labels[i] != -100]
                target_list.append(valid)
                target_length_list.append(len(valid))
            targets = torch.cat(target_list)
            target_lengths = torch.tensor(target_length_list, dtype=torch.long, device=device)

            loss = ctc_loss_fn(log_probs, targets, output_lengths, target_lengths)
            loss = loss / grad_accum
            loss.backward()

            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Epoch {epoch+1}/{max_epochs}] Step {global_step}/{total_steps} "
                        f"| Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

        train_loss = epoch_loss / max(num_batches, 1)
        train_time = time.time() - epoch_start

        # ---- Evaluate ----
        eval_start = time.time()
        print(f"\n  Evaluating epoch {epoch+1}...")
        eval_results = evaluate(model, dev_loader, idx2char, device)
        eval_time = time.time() - eval_start

        dev_cer = eval_results["cer"]
        dev_wer = eval_results["wer"]

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "dev_cer": dev_cer,
            "dev_wer": dev_wer,
            "lr": scheduler.get_last_lr()[0],
            "train_time_sec": round(train_time, 1),
            "eval_time_sec": round(eval_time, 1),
        }
        all_epoch_metrics.append(epoch_metrics)
        timing["epochs"].append({
            "epoch": epoch + 1,
            "train_sec": round(train_time, 1),
            "eval_sec": round(eval_time, 1),
        })

        improved = dev_cer < best_cer
        if improved:
            best_cer = dev_cer
            patience_counter = 0
            # Save best checkpoint
            ckpt_dir = output_path / "best_checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            # Save vocab alongside checkpoint
            with open(ckpt_dir / "ctc_vocab.json", "w", encoding="utf-8") as f_v:
                json.dump(char2idx, f_v, ensure_ascii=False, indent=2)
            print(f"  ** New best CER: {dev_cer:.2f}% -- checkpoint saved **")
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch+1}/{max_epochs} | TrainLoss: {train_loss:.4f} "
            f"| DevCER: {dev_cer:.2f}% | DevWER: {dev_wer:.2f}% "
            f"| Patience: {patience_counter}/{patience} "
            f"| Time: {train_time:.0f}s train + {eval_time:.0f}s eval"
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    timing["train_end"] = time.time()
    timing["total_sec"] = round(timing["train_end"] - timing["train_start"], 1)
    timing["total_hours"] = round(timing["total_sec"] / 3600, 3)

    # ---- Final evaluation with best model ----
    print(f"\nLoading best checkpoint for final evaluation...")
    ckpt_dir = output_path / "best_checkpoint"
    if ckpt_dir.exists():
        model = Wav2Vec2ForCTC.from_pretrained(str(ckpt_dir))
        model.to(device)

    final_eval = evaluate(model, dev_loader, idx2char, device)

    # ---- Save decode samples ----
    samples_path = output_path / "decode_samples.txt"
    n_samples = min(num_decode_samples, len(final_eval["references"]))
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"C5 wav2vec 2.0 Fine-Tune -- Decode Samples (best checkpoint)\n")
        f.write(f"{'='*70}\n\n")
        for i in range(n_samples):
            f.write(f"[{final_eval['ids'][i]}]\n")
            f.write(f"  REF: {final_eval['references'][i]}\n")
            f.write(f"  HYP: {final_eval['hypotheses'][i]}\n\n")
    print(f"Saved {n_samples} decode samples to {samples_path}")

    # ---- Save metrics ----
    metrics = {
        "experiment": "C5_wav2vec2_finetune",
        "model_name": config["model_name"],
        "best_dev_cer": best_cer,
        "best_dev_wer": final_eval["wer"],
        "final_cer_details": final_eval["cer_details"],
        "final_wer_details": final_eval["wer_details"],
        "epoch_metrics": all_epoch_metrics,
        "config": config,
        "seed": seed,
        "dry_run": dry_run,
        "device": str(device),
        "num_train_samples": len(train_dataset),
        "num_dev_samples": len(dev_dataset),
        "vocab_size": vocab_size,
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved metrics to {metrics_path}")

    # ---- Save timing ----
    timing_path = output_path / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Saved timing to {timing_path}")

    print(f"\nTraining complete. Best dev CER: {best_cer:.2f}%")
    print(f"Total time: {timing['total_hours']:.2f}h")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="C5: Fine-tune wav2vec 2.0 on Adja ASR")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to prepared data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints and results")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 2 steps on 5 samples")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)

    print(f"C5: wav2vec 2.0 Fine-Tune on Adja ASR")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {seed}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")

    train(config, args.data_dir, args.output_dir, seed, args.dry_run)


if __name__ == "__main__":
    main()
