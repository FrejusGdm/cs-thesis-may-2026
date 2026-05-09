#!/usr/bin/env python3
# /// script
# dependencies = ["transformers>=4.48.0,<4.53.0", "torch", "torchaudio", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "huggingface-hub>=0.34.0", "hf_transfer"]
# ///
from __future__ import annotations
"""
E7 Stage 2: Adapt Ewe-tuned Whisper large-v3 to Adja ASR.

Loads the Stage 1 checkpoint (Whisper large-v3 fine-tuned on Ewe) and
continues training on Adja data with a lower learning rate. Hypothesis:
the Ewe Gbe-family prior from Stage 1 improves sample efficiency on Adja's
~1.6k labeled utterances relative to starting from the English-only checkpoint.

Prerequisites:
  - E7_whisper_largev3_ewe_stage1.py must have completed and pushed its model.
  - Set --stage1-checkpoint to the Hub path of the Stage 1 model.

References:
  - Whisper: https://cdn.openai.com/papers/whisper.pdf
  - WaxalNLP: https://huggingface.co/datasets/google/WaxalNLP
  - Ewe→Adja hypothesis: experiments/asr-tts-getting-right-2026-04-21.md, Track 2B
"""
import argparse, functools, json, os, sys, time, random, unicodedata
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    p = argparse.ArgumentParser(description="E7 Stage 2: Ewe-adapted Whisper → Adja")
    p.add_argument("--stage1-checkpoint",
                   default="JosueG/adja-asr-checkpoints/E7_whisper_largev3_ewe_stage1",
                   help="Hub path (repo_id/subfolder) to Stage 1 model")
    p.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6, help="Lower LR for Stage 2 adaptation")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-asr-results")
    p.add_argument("--results-prefix", default="E7_whisper_largev3_ewe_adja_stage2")
    return p.parse_args()


def normalize_text(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
    return dp[n][m]


def compute_cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    return round(edits / max(sum(len(r) for r in refs), 1) * 100, 2)


def compute_wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return round(edits / max(sum(len(r.split()) for r in refs), 1) * 100, 2)


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import librosa
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "A100 80GB required"
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # ===== Download Stage 1 checkpoint =====
    stage1_repo, stage1_subdir = args.stage1_checkpoint.rsplit("/", 1)
    print(f"Downloading Stage 1 from {args.stage1_checkpoint}...")
    snapshot_download(stage1_repo, local_dir="/tmp/stage1_e7",
                      allow_patterns=f"{stage1_subdir}/*", token=token)
    stage1_local = f"/tmp/stage1_e7/{stage1_subdir}"

    # ===== Dataset (Adja, same 80/10/10 split as all other experiments) =====
    print("Loading Adja dataset...")
    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds, dev_ds, test_ds = split2["train"], split2["test"], split1["test"]
    print(f"Train={len(train_ds)} | Dev={len(dev_ds)} | Test={len(test_ds)}\n")

    # ===== Model =====
    print(f"Loading Stage 1 checkpoint from {stage1_local}...")
    processor = WhisperProcessor.from_pretrained(stage1_local)
    model = WhisperForConditionalGeneration.from_pretrained(stage1_local)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.to(device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    # ===== Data pipeline (identical to whisper_finetune.py — proven path) =====
    class AdjaSRDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self.data = hf_split
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            s = self.data[idx]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            sr = s["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return {"audio": audio, "text": normalize_text(s["text"])}

    def collate_fn(batch, processor):
        audios = [b["audio"] for b in batch]
        texts  = [b["text"] for b in batch]
        feats = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        if feats.shape[-1] < 3000:
            feats = torch.nn.functional.pad(feats, (0, 3000 - feats.shape[-1]))
        else:
            feats = feats[..., :3000]
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        label_ids, attn_mask = labels.input_ids, labels.attention_mask
        bos_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if bos_id is not None and label_ids.size(1) > 0 and (label_ids[:, 0] == bos_id).all():
            label_ids, attn_mask = label_ids[:, 1:], attn_mask[:, 1:]
        label_ids = label_ids.masked_fill(attn_mask == 0, -100)
        return {"input_features": feats, "labels": label_ids, "texts": texts}

    collate = functools.partial(collate_fn, processor=processor)
    train_loader = torch.utils.data.DataLoader(AdjaSRDataset(train_ds), batch_size=args.batch_size,
                                               shuffle=True, num_workers=0, collate_fn=collate)
    dev_loader   = torch.utils.data.DataLoader(AdjaSRDataset(dev_ds), batch_size=args.batch_size,
                                               shuffle=False, num_workers=0, collate_fn=collate)

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.max_epochs

    def get_lr(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ===== Training =====
    best_cer = float("inf"); best_epoch = 0; no_improve = 0
    history = []; global_step = 0
    t_start = time.time()

    print(f"Training: up to {args.max_epochs} epochs, patience={args.patience}, lr={args.lr}")
    for epoch in range(1, args.max_epochs + 1):
        model.train(); epoch_loss = 0; n_batches = 0; optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_features=feats, labels=labels).loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss): continue
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        model.eval(); all_refs, all_hyps = [], []
        with torch.no_grad():
            for batch in dev_loader:
                feats = batch["input_features"].to(device)
                pred_ids = model.generate(feats, max_new_tokens=225, do_sample=False, num_beams=1)
                all_hyps.extend(h.strip() for h in processor.batch_decode(pred_ids, skip_special_tokens=True))
                all_refs.extend(batch["texts"])

        dev_cer = compute_cer(all_refs, all_hyps)
        dev_wer = compute_wer(all_refs, all_hyps)
        print(f"Epoch {epoch}/{args.max_epochs} | loss={avg_loss:.4f} | CER={dev_cer}% | WER={dev_wer}%")
        for i in range(min(2, len(all_refs))):
            print(f"  REF: {all_refs[i]}\n  HYP: {all_hyps[i]}")

        history.append({"epoch": epoch, "loss": round(avg_loss, 4), "dev_cer": dev_cer, "dev_wer": dev_wer})
        if dev_cer < best_cer:
            best_cer = dev_cer; best_epoch = epoch; no_improve = 0
            model.save_pretrained("/tmp/best_e7_adja"); processor.save_pretrained("/tmp/best_e7_adja")
            print(f"  ** New best CER: {best_cer}%")
        else:
            no_improve += 1
            print(f"  No improve {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print("Early stopping!"); break

    elapsed = time.time() - t_start
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    print(f"\nDone! Best CER={best_cer}% at epoch {best_epoch} | time={elapsed/60:.1f}min | VRAM={peak_mem}GB")

    results = {
        "experiment": "E7_whisper_largev3_ewe_adja_stage2",
        "stage1_checkpoint": args.stage1_checkpoint,
        "adja_dataset": args.dataset,
        "best_cer": best_cer, "best_epoch": best_epoch,
        "training_time_min": round(elapsed / 60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "history": history,
    }

    if args.push_to_hub:
        api = HfApi(token=token)
        prefix = args.results_prefix
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
            path_in_repo=f"{prefix}/metrics.json",
            repo_id=args.results_repo, token=token,
        )
        api.upload_folder(folder_path="/tmp/best_e7_adja",
                          path_in_repo=f"{prefix}/best_model",
                          repo_id=args.results_repo, token=token)
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    main()
