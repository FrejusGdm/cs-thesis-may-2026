#!/usr/bin/env python3
# /// script
# dependencies = ["transformers==4.48.3", "torch==2.5.1", "torchaudio==2.5.1", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "huggingface-hub>=0.34.0", "hf_transfer"]
# ///
from __future__ import annotations
"""
E7 Stage 1: Fine-tune Whisper large-v3 on WaxalNLP Ewe ASR (labeled split).

Hypothesis: Pre-adapting Whisper large-v3 on Ewe (Adja's closest Gbe relative,
15k labeled utterances) before Adja adaptation gives the model Gbe-family
phonological priors and improves sample efficiency on Adja in Stage 2.

Same training loop as whisper_finetune.py (proven E4 path):
  - Custom optimizer + LR schedule (not Trainer)
  - padding=True + manual pad-to-3000 (NOT padding="max_length" — see gotchas)
  - attention_mask-based label masking (NOT equality mask — preserves EOS)
  - Early stopping on dev CER (not loss)

Compute: Whisper large-v3 is 1.5B params — requires A100 80GB.

Data:
  google/WaxalNLP, config=ewe_asr
  train: ~15k labeled  |  validation: ~1.9k  |  test: ~1.9k
  Audio: 48kHz → resampled to 16kHz in collate

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
    p = argparse.ArgumentParser(description="E7 Stage 1: Whisper large-v3 on Ewe ASR")
    p.add_argument("--dataset", default="google/WaxalNLP")
    p.add_argument("--dataset-config", default="ewe_asr")
    p.add_argument("--base-model", default="openai/whisper-large-v3")
    p.add_argument("--output-dir", default="/tmp/e7_ewe_stage1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-asr-results")
    p.add_argument("--results-prefix", default="E7_whisper_largev3_ewe_stage1")
    p.add_argument("--checkpoint-repo", default="JosueG/adja-asr-checkpoints",
                   help="Separate repo for best model (loaded by Stage 2)")
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
    total = sum(len(r) for r in refs)
    return round(edits / max(total, 1) * 100, 2)


def compute_wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits / max(total, 1) * 100, 2)


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import librosa
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "A100 80GB required for Whisper large-v3"
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # ===== Dataset =====
    print("Loading WaxalNLP ewe_asr...")
    train_ds = load_dataset(args.dataset, name=args.dataset_config, split="train", token=token)
    dev_ds   = load_dataset(args.dataset, name=args.dataset_config, split="validation", token=token)
    test_ds  = load_dataset(args.dataset, name=args.dataset_config, split="test", token=token)

    text_col = "text" if "text" in train_ds.column_names else "sentence"
    print(f"Columns: {train_ds.column_names} → text column: '{text_col}'")
    print(f"Train={len(train_ds)} | Dev={len(dev_ds)} | Test={len(test_ds)}\n")

    # ===== Model =====
    print(f"Loading {args.base_model}...")
    processor = WhisperProcessor.from_pretrained(args.base_model, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, token=token)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.0f}M\n")

    # ===== Data pipeline =====
    class EweASRDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self.data = hf_split
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            s = self.data[idx]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            sr = s["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return {"audio": audio, "text": normalize_text(s[text_col])}

    def collate_fn(batch, processor):
        audios = [b["audio"] for b in batch]
        texts  = [b["text"] for b in batch]

        input_features = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        if input_features.shape[-1] < 3000:
            input_features = torch.nn.functional.pad(input_features, (0, 3000 - input_features.shape[-1]))
        else:
            input_features = input_features[..., :3000]

        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        label_ids, attn_mask = labels.input_ids, labels.attention_mask

        bos_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if bos_id is not None and label_ids.size(1) > 0 and (label_ids[:, 0] == bos_id).all():
            label_ids, attn_mask = label_ids[:, 1:], attn_mask[:, 1:]

        label_ids = label_ids.masked_fill(attn_mask == 0, -100)
        return {"input_features": input_features, "labels": label_ids, "texts": texts}

    collate = functools.partial(collate_fn, processor=processor)
    train_loader = torch.utils.data.DataLoader(EweASRDataset(train_ds), batch_size=args.batch_size,
                                               shuffle=True, num_workers=0, collate_fn=collate)
    dev_loader   = torch.utils.data.DataLoader(EweASRDataset(dev_ds), batch_size=args.batch_size,
                                               shuffle=False, num_workers=0, collate_fn=collate)

    # ===== Optimizer + schedule =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.max_epochs

    def get_lr(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ===== Training loop =====
    best_cer = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    global_step = 0
    t_start = time.time()
    output_dir = args.output_dir

    print(f"Training: up to {args.max_epochs} epochs, patience={args.patience}, lr={args.lr}")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0; n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_features=feats, labels=labels).loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        all_refs, all_hyps = [], []
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
            print(f"  REF: {all_refs[i]}")
            print(f"  HYP: {all_hyps[i]}")

        history.append({"epoch": epoch, "loss": round(avg_loss, 4), "dev_cer": dev_cer, "dev_wer": dev_wer})

        if dev_cer < best_cer:
            best_cer = dev_cer; best_epoch = epoch; no_improve = 0
            model.save_pretrained(f"/tmp/best_e7_ewe")
            processor.save_pretrained(f"/tmp/best_e7_ewe")
            print(f"  ** New best CER: {best_cer}%")
        else:
            no_improve += 1
            print(f"  No improve {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print("Early stopping!")
                break

    elapsed = time.time() - t_start
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    print(f"\nDone! Best CER={best_cer}% at epoch {best_epoch} | time={elapsed/60:.1f}min | VRAM={peak_mem}GB")

    results = {
        "experiment": "E7_whisper_largev3_ewe_stage1",
        "base_model": args.base_model,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "best_cer": best_cer, "best_epoch": best_epoch,
        "training_time_min": round(elapsed / 60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "history": history,
        "stage2_checkpoint": f"{args.checkpoint_repo}/E7_whisper_largev3_ewe_stage1",
    }

    if args.push_to_hub:
        api = HfApi(token=token)
        prefix = args.results_prefix
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        metrics_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode()
        api.upload_file(path_or_fileobj=metrics_bytes,
                        path_in_repo=f"{prefix}/metrics.json",
                        repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path="/tmp/best_e7_ewe",
                          path_in_repo=f"{prefix}/best_model",
                          repo_id=args.results_repo, token=token)

        # Push checkpoint for Stage 2
        api.create_repo(args.checkpoint_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path="/tmp/best_e7_ewe",
                          path_in_repo="E7_whisper_largev3_ewe_stage1",
                          repo_id=args.checkpoint_repo, token=token)
        print(f"Checkpoint pushed to {args.checkpoint_repo}/E7_whisper_largev3_ewe_stage1")
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")
    else:
        print("Skipping Hub upload. Best model at /tmp/best_e7_ewe/")


if __name__ == "__main__":
    main()
