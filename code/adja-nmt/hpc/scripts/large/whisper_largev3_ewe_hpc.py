#!/usr/bin/env python3
from __future__ import annotations
"""
Whisper large-v3 Ewe → Adja ASR — HPC version.

Last updated: 2026-04-21

This is the HPC scale-up of scripts/hf_jobs/E7_whisper_largev3_ewe_stage1.py.
Differences vs the HF Jobs version:
  - 50+ epochs instead of 20 (we have more clock time here)
  - Larger effective batch (8 x 16 = 128) with gradient accumulation
  - Reads pre-cached model from /models (no HF Hub download at runtime)
  - Reads pre-cached WaxalNLP from /data
  - Saves checkpoints to /results (read-write bind mount)

Two-stage:
  Stage A: Whisper large-v3 → 15k Ewe labeled ASR (WaxalNLP ewe_asr)
  Stage B: Ewe-adapted Whisper → Adja (JosueG/adja-tts-orpheus audio split)

Runtime (A100 80GB): Stage A ~18-24h, Stage B ~4-8h. Fits in the 36h wall-time
budget of submit_whisper_largev3_ewe.sbatch.

References:
  - Whisper: https://cdn.openai.com/papers/whisper.pdf
  - Cross-lingual transfer for Whisper: E4 experiment in registry.md (CER 24.90%)
  - Whisper training gotchas: ../../docs/whisper-training-gotchas.md
    (pad-to-3000-mel, BOS strip, attention-mask-based label masking, NOT
    padding="max_length" on raw audio, NOT label_ids == pad_id masking —
    that bug broke E4v2 and E4 redo attempts with CER 76-400%). All three
    gotchas are encoded in make_collate() below — do not "simplify" them.
"""
import argparse, functools, json, os, random, sys, time, unicodedata
from pathlib import Path
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretraining"))
from waxalnlp_loader import detect_transcript_col, load_waxal_split, safe_load_audio_array


def parse_args():
    p = argparse.ArgumentParser(description="Whisper large-v3 Ewe → Adja (HPC)")
    p.add_argument("--models-dir", default="/models")
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--output-dir", default="/results/whisper_largev3_ewe")
    p.add_argument("--base", default="openai/whisper-large-v3")
    p.add_argument("--stage", choices=["ewe", "adja", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--ewe-lr", type=float, default=1e-5)
    p.add_argument("--adja-lr", type=float, default=5e-6)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dry run: 4 samples, 1 step, Ewe stage only, no save.")
    return p.parse_args()


def resolve(args, repo):
    safe = repo.replace("/", "__")
    candidate = Path(args.models_dir) / safe
    return str(candidate) if candidate.exists() else repo


def normalize(t):
    return " ".join(unicodedata.normalize("NFKC", t.strip()).split())


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
    return dp[n][m]


def compute_cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    return round(edits / max(sum(len(r) for r in refs), 1) * 100, 2)


def compute_wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return round(edits / max(sum(len(r.split()) for r in refs), 1) * 100, 2)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, text_col):
        self.data = hf_split
        self.text_col = text_col
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        audio = safe_load_audio_array(s.get("audio"), target_sr=16000)
        if audio is None or len(audio) == 0:
            return None
        return {"audio": audio, "text": normalize(s[self.text_col])}


def make_collate(processor):
    def collate(batch):
        batch = [b for b in batch if b is not None and b.get("text")]
        if not batch:
            return None
        audios = [b["audio"] for b in batch]
        texts = [b["text"] for b in batch]
        feats = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        if feats.shape[-1] < 3000:
            feats = torch.nn.functional.pad(feats, (0, 3000 - feats.shape[-1]))
        else:
            feats = feats[..., :3000]
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        label_ids, attn = labels.input_ids, labels.attention_mask
        bos = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if bos is not None and label_ids.size(1) > 0 and (label_ids[:, 0] == bos).all():
            label_ids, attn = label_ids[:, 1:], attn[:, 1:]
        label_ids = label_ids.masked_fill(attn == 0, -100)
        return {"input_features": feats, "labels": label_ids, "texts": texts}
    return collate


def train_stage(*, model, processor, train_ds, dev_ds, text_col, args, lr, save_dir, stage_name):
    device = torch.device("cuda")
    collate = make_collate(processor)
    train_loader = torch.utils.data.DataLoader(AudioDataset(train_ds, text_col),
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=4, collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(AudioDataset(dev_ds, text_col),
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, collate_fn=collate)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs

    def get_lr(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    best_cer = float("inf"); best_epoch = 0; no_improve = 0
    history = []
    t0 = time.time()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0; n = 0; optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                loss = model(input_features=feats, labels=labels).loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            epoch_loss += loss.item(); n += 1
        avg_loss = epoch_loss / max(n, 1)

        model.eval()
        all_refs, all_hyps = [], []
        with torch.no_grad():
            for batch in dev_loader:
                if batch is None:
                    continue
                feats = batch["input_features"].to(device)
                pred_ids = model.generate(feats, max_new_tokens=225, do_sample=False, num_beams=1)
                all_hyps.extend(h.strip() for h in processor.batch_decode(pred_ids, skip_special_tokens=True))
                all_refs.extend(batch["texts"])
        dev_cer = compute_cer(all_refs, all_hyps)
        dev_wer = compute_wer(all_refs, all_hyps)
        print(f"[{stage_name}] Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | CER={dev_cer}% | WER={dev_wer}%")
        if all_refs:
            print(f"  REF: {all_refs[0]}\n  HYP: {all_hyps[0]}")
        history.append({"epoch": epoch, "loss": round(avg_loss, 4), "dev_cer": dev_cer, "dev_wer": dev_wer})

        if dev_cer < best_cer:
            best_cer = dev_cer; best_epoch = epoch; no_improve = 0
            model.save_pretrained(str(save_dir))
            processor.save_pretrained(str(save_dir))
            print(f"  ** new best CER: {best_cer}%")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping."); break

    return {
        "best_cer": best_cer, "best_epoch": best_epoch,
        "time_min": round((time.time() - t0) / 60, 1),
        "history": history,
    }


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    from datasets import Audio, load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB\n")

    # Auto-fit for MIG 40GB slices (Discovery often allocates 3g.40gb).
    # Prefer keeping the job alive with a smaller effective batch.
    if vram_gb < 60:
        old_bs, old_ga = args.batch_size, args.grad_accum
        args.batch_size = min(args.batch_size, 1)
        args.grad_accum = min(args.grad_accum, 8)
        print(f"[auto-fit] low VRAM detected ({vram_gb:.1f}GB): "
              f"batch_size {old_bs}→{args.batch_size}, grad_accum {old_ga}→{args.grad_accum}")
    use_lora = vram_gb < 60

    base_path = resolve(args, args.base)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ewe_out = output_dir / "stage_ewe"
    adja_out = output_dir / "stage_adja"

    all_results = {}

    def prepare_model(model):
        model.config.use_cache = False
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.generation_config.task = "transcribe"
        model.generation_config.language = None
        model.gradient_checkpointing_enable()
        if use_lora and not isinstance(model, PeftModel):
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            ))
            model.print_trainable_parameters()
        return model.to(device)

    if args.smoke:
        print(">>> SMOKE MODE: 4 samples, 1 epoch, Ewe stage only, no save <<<\n")
        args.stage = "ewe"
        args.epochs = 1
        args.batch_size = 1 if vram_gb < 60 else 2
        args.grad_accum = 1
        args.warmup_steps = 0

    # ---- Stage Ewe ----
    if args.stage in ("ewe", "both"):
        processor = WhisperProcessor.from_pretrained(base_path)
        model = WhisperForConditionalGeneration.from_pretrained(base_path)
        model = prepare_model(model)

        train = load_waxal_split(
            config="ewe_asr",
            split="train",
            cache_dir=str(Path(args.data_dir) / "WaxalNLP_ewe_asr"),
            token=hf_token,
        )
        val = load_waxal_split(
            config="ewe_asr",
            split="validation",
            cache_dir=str(Path(args.data_dir) / "WaxalNLP_ewe_asr"),
            token=hf_token,
        )
        if args.smoke:
            train = train.select(range(4))
            val = val.select(range(2))
        train = train.cast_column("audio", Audio(decode=False))
        val = val.cast_column("audio", Audio(decode=False))
        text_col = detect_transcript_col(train)
        print(f"Ewe: train={len(train)} val={len(val)} | text_col={text_col}")

        all_results["ewe"] = train_stage(
            model=model, processor=processor,
            train_ds=train, dev_ds=val, text_col=text_col,
            args=args, lr=args.ewe_lr, save_dir=ewe_out, stage_name="Ewe",
        )
        if args.smoke:
            peak_gb = torch.cuda.max_memory_reserved() / 1e9
            print(f"SMOKE PASS: whisper_largev3_ewe_hpc | peak VRAM {peak_gb:.1f}GB")
            return

    # ---- Stage Adja ----
    if args.stage in ("adja", "both"):
        start = str(ewe_out) if args.stage == "both" else base_path
        processor = WhisperProcessor.from_pretrained(start)
        if use_lora and args.stage == "both":
            base_model = WhisperForConditionalGeneration.from_pretrained(base_path)
            model = PeftModel.from_pretrained(base_model, start, is_trainable=True)
            model = prepare_model(model)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(start)
            model = prepare_model(model)

        adja = load_dataset("JosueG/adja-tts-orpheus", split="train",
                            cache_dir=str(Path(args.data_dir) / "JosueG_adja-tts-orpheus"),
                            token=hf_token)
        adja = adja.cast_column("audio", Audio(decode=False))
        s1 = adja.train_test_split(test_size=0.1, seed=args.seed)
        s2 = s1["train"].train_test_split(test_size=0.1/0.9, seed=args.seed)
        print(f"Adja: train={len(s2['train'])} dev={len(s2['test'])}")

        args_adja = argparse.Namespace(**vars(args))
        args_adja.epochs = min(args.epochs, 30)  # Adja needs fewer epochs
        all_results["adja"] = train_stage(
            model=model, processor=processor,
            train_ds=s2["train"], dev_ds=s2["test"], text_col="text",
            args=args_adja, lr=args.adja_lr, save_dir=adja_out, stage_name="Adja",
        )

    all_results["gpu"] = torch.cuda.get_device_name(0)
    all_results["peak_vram_gb"] = round(torch.cuda.max_memory_reserved()/1e9, 2)
    (output_dir / "metrics.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
