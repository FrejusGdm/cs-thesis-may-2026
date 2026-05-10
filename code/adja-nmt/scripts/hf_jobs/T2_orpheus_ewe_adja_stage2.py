#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.6.0", "torchaudio==2.6.0", "transformers==4.56.2", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "numpy", "scipy", "sentencepiece", "protobuf", "bitsandbytes", "snac", "trl==0.22.2"]
# ///
from __future__ import annotations
"""
T2-ewe Stage 2: Adapt an Ewe-tuned Orpheus checkpoint to Adja.

Last updated: 2026-04-21

Loads the Stage 1 checkpoint (Orpheus fine-tuned on WaxalNLP Ewe TTS) and
continues training on the Adja TTS dataset at a lower learning rate. Runs once
per base variant (English / Chinese / French) so we can compare bases.

Prerequisites:
  - T2_orpheus_ewe_stage1.py must have completed and pushed its merged model
    to JosueG/adja-tts-checkpoints/T2_orpheus_{en|zh|fr}_ewe_stage1

References:
  - Orpheus TTS: https://huggingface.co/canopylabs
  - SNAC codec: https://github.com/hubertsiuzdak/snac
  - Cross-lingual TTS transfer: experiments/asr-tts-getting-right-2026-04-21.md, Track 1B
"""
import argparse, json, os, random, subprocess, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

TOKENISER_LENGTH = 128256
END_OF_TEXT      = 128009
START_OF_SPEECH  = TOKENISER_LENGTH + 1
END_OF_SPEECH    = TOKENISER_LENGTH + 2
START_OF_HUMAN   = TOKENISER_LENGTH + 3
END_OF_HUMAN     = TOKENISER_LENGTH + 4
START_OF_AI      = TOKENISER_LENGTH + 5
END_OF_AI        = TOKENISER_LENGTH + 6
PAD_TOKEN        = TOKENISER_LENGTH + 7
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10


def parse_args():
    p = argparse.ArgumentParser(description="T2-ewe Stage 2: Orpheus Ewe → Adja")
    p.add_argument("--stage1-checkpoint",
                   default="JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1",
                   help="Hub path (repo_id/subfolder) to Stage 1 merged model")
    p.add_argument("--checkpoint-tag", default="en", help="en|zh|fr for result naming")
    p.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    p.add_argument("--output-dir", default="/tmp/orpheus_ewe_adja_stage2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5,
                   help="Lower LR for Stage 2 adaptation (vs 2e-4 in Stage 1)")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T2_orpheus_en_ewe_adja_stage2")
    return p.parse_args()


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def build_snac_codes(audio_array, snac_model, orig_sr, device):
    import numpy as np, torch
    import torchaudio.transforms as T
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if audio.size:
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 2.0:
            scale = 65536.0 if max_abs <= 65536.0 * 1.1 else max_abs
            audio = audio / scale
    waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    if orig_sr != 24000:
        waveform = T.Resample(orig_freq=orig_sr, new_freq=24000)(waveform)
    with torch.inference_mode():
        codes = snac_model.encode(waveform.unsqueeze(0).to(device))
    flat = []
    for i in range(codes[0].shape[1]):
        flat.append(codes[0][0][i].item() + AUDIO_TOKENS_START)
        flat.append(codes[1][0][2*i].item() + AUDIO_TOKENS_START + 4096)
        flat.append(codes[2][0][4*i].item() + AUDIO_TOKENS_START + 2*4096)
        flat.append(codes[2][0][4*i+1].item() + AUDIO_TOKENS_START + 3*4096)
        flat.append(codes[1][0][2*i+1].item() + AUDIO_TOKENS_START + 4*4096)
        flat.append(codes[2][0][4*i+2].item() + AUDIO_TOKENS_START + 5*4096)
        flat.append(codes[2][0][4*i+3].item() + AUDIO_TOKENS_START + 6*4096)
    return flat


def dedupe_frames(codes):
    if len(codes) % 7 != 0:
        raise ValueError(f"codes length {len(codes)} not divisible by 7")
    kept = codes[:7]
    for i in range(7, len(codes), 7):
        if codes[i] != kept[-7]:
            kept.extend(codes[i:i+7])
    return kept


def redistribute_codes(flat):
    import torch
    l1, l2, l3 = [], [], []
    for i in range((len(flat)+1)//7):
        b = 7*i
        l1.append(flat[b]   - AUDIO_TOKENS_START)
        l2.append(flat[b+1] - AUDIO_TOKENS_START - 4096)
        l3.append(flat[b+2] - AUDIO_TOKENS_START - 2*4096)
        l3.append(flat[b+3] - AUDIO_TOKENS_START - 3*4096)
        l2.append(flat[b+4] - AUDIO_TOKENS_START - 4*4096)
        l3.append(flat[b+5] - AUDIO_TOKENS_START - 5*4096)
        l3.append(flat[b+6] - AUDIO_TOKENS_START - 6*4096)
    return [torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0), torch.tensor(l3).unsqueeze(0)]


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np, soundfile as sf, torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               DataCollatorForLanguageModeling,
                               EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ===== Download Stage 1 checkpoint =====
    stage1_repo, stage1_subdir = args.stage1_checkpoint.rsplit("/", 1)
    print(f"Downloading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    snapshot_download(stage1_repo, local_dir="/tmp/stage1_orpheus_ewe",
                      allow_patterns=f"{stage1_subdir}/*", token=token)
    stage1_local = f"/tmp/stage1_orpheus_ewe/{stage1_subdir}"

    # ===== Adja dataset (80/10/10 split, seed=42) =====
    ds = load_dataset(args.dataset, token=token, split="train")
    s1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    s2 = s1["train"].train_test_split(test_size=0.1/0.9, seed=args.seed)
    train_ds, dev_ds, test_ds = s2["train"], s2["test"], s1["test"]
    orig_sr = train_ds[0]["audio"]["sampling_rate"]
    print(f"Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)} | orig_sr={orig_sr}Hz\n")

    # ===== Model (from Stage 1 merged checkpoint) =====
    tokenizer = AutoTokenizer.from_pretrained(stage1_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(stage1_local, torch_dtype=torch.float32).to("cuda")
    total_params = sum(p.numel() for p in model.parameters())

    if args.full_finetune:
        training_mode = "full_finetune"
    else:
        training_mode = f"lora_r{args.lora_r}"
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable/1e6:.2f}M / {total_params/1e6:.1f}M\n")

    # ===== SNAC encode =====
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
    dropped = {"empty_text":0,"short_audio":0,"encode_fail":0,"too_long":0}

    def preprocess(ex):
        text = normalize_text(ex.get("text") or "")
        if not text:
            dropped["empty_text"] += 1
            return None
        arr = ex.get("audio", {}).get("array")
        if arr is None or len(arr) < 10000:
            dropped["short_audio"] += 1
            return None
        try:
            flat = build_snac_codes(arr, snac_model, orig_sr, "cuda")
        except Exception as e:
            print(f"  skip: {e}")
            dropped["encode_fail"] += 1
            return None
        flat = dedupe_frames(flat)
        text_ids = tokenizer.encode(text, add_special_tokens=True) + [END_OF_TEXT]
        ids = [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH] + flat + [END_OF_SPEECH] + [END_OF_AI]
        if len(ids) > args.max_seq_length:
            dropped["too_long"] += 1
            return None
        return {"input_ids": ids, "labels": ids, "attention_mask": [1]*len(ids)}

    def proc(split, name):
        out = split.map(preprocess, remove_columns=split.column_names, desc=f"Encode {name}")
        return out.filter(lambda x: x.get("input_ids") is not None)

    processed_train = proc(train_ds, "train")
    processed_dev   = proc(dev_ds, "dev")
    print(f"Encoded: train={len(processed_train)} dev={len(processed_dev)} | dropped={dropped}\n")
    snac_model.to("cpu")
    torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Train =====
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10, num_train_epochs=args.num_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5, optim="adamw_torch" if args.full_finetune else "adamw_8bit",
        weight_decay=0.001, lr_scheduler_type="cosine", seed=args.seed,
        report_to="none", eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, train_dataset=processed_train, eval_dataset=processed_dev,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved()/1e9, 2)
    eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
    best_eval = min((l["eval_loss"] for l in eval_history), default=None)
    print(f"Done {elapsed/60:.1f}min | train_loss={train_loss:.4f} | best_eval={best_eval}\n")

    # ===== Quick generation (5 Adja sentences) =====
    model.eval()
    snac_model.to("cpu")
    generated = []
    start_tok = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_toks = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    for i in range(min(5, len(test_ds))):
        text = normalize_text(test_ds[i]["text"])
        try:
            ids = tokenizer(text, return_tensors="pt").input_ids
            prompt = torch.cat([start_tok, ids, end_toks], dim=1).to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                     max_new_tokens=1200, do_sample=True, temperature=0.6,
                                     top_p=0.95, repetition_penalty=1.1, eos_token_id=END_OF_SPEECH)
            row = out[0].tolist()
            if START_OF_SPEECH in row:
                row = row[len(row) - 1 - row[::-1].index(START_OF_SPEECH):][1:]
            row = [t for t in row if t != END_OF_SPEECH]
            row = row[:(len(row)//7)*7]
            if not row:
                raise RuntimeError("no audio frames")
            codes = redistribute_codes(row)
            # Re-upload SNAC to GPU for decoding
            snac_model.to("cuda")
            with torch.inference_mode():
                wav = snac_model.decode([c.to("cuda") for c in codes]).detach().squeeze().cpu().numpy().astype(np.float32)
            snac_model.to("cpu")
            p = generated_dir / f"adja_{i:02d}.wav"
            sf.write(str(p), wav, 24000)
            generated.append({"text": text, "file": p.name, "duration_sec": round(float(wav.size)/24000, 2)})
        except Exception as e:
            generated.append({"text": text, "error": str(e)})

    # ===== Save =====
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    results = {
        "experiment": f"T2_orpheus_{args.checkpoint_tag}_ewe_adja_stage2",
        "stage1_checkpoint": args.stage1_checkpoint,
        "checkpoint_tag": args.checkpoint_tag,
        "adja_dataset": args.dataset,
        "training_mode": training_mode,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval, 4) if best_eval else None,
        "training_time_min": round(elapsed/60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
        "dropped": dropped,
    }
    # Write metrics — wrapped because a single UnicodeEncodeError here historically lost the
    # whole job's output (generated_audio never pushed). Don't let one failure block the rest.
    metrics_path = output_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to write metrics.json ({type(exc).__name__}: {exc}). Continuing so audio still pushes.")

    if args.push_to_hub:
        api = HfApi(token=token)
        prefix = args.results_prefix
        try:
            api.create_repo(args.results_repo, private=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] create_repo failed ({type(exc).__name__}: {exc})")
        # Push GENERATED AUDIO FIRST — highest-signal artifact for the listening test.
        # If anything below fails we still have the audio on Hub.
        try:
            api.upload_folder(folder_path=str(generated_dir),
                              path_in_repo=f"{prefix}/generated_audio",
                              repo_id=args.results_repo, token=token)
            print(f"[push] generated_audio -> {prefix}/generated_audio")
        except Exception as exc:
            print(f"[WARN] generated_audio upload failed ({type(exc).__name__}: {exc})")
        try:
            if metrics_path.exists():
                api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                                path_in_repo=f"{prefix}/metrics.json",
                                repo_id=args.results_repo, token=token)
                print(f"[push] metrics.json")
        except Exception as exc:
            print(f"[WARN] metrics.json upload failed ({type(exc).__name__}: {exc})")
        try:
            api.upload_folder(folder_path=str(adapter_dir),
                              path_in_repo=f"{prefix}/adapter",
                              repo_id=args.results_repo, token=token)
            print(f"[push] adapter")
        except Exception as exc:
            print(f"[WARN] adapter upload failed ({type(exc).__name__}: {exc})")
        print(f"Done: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
