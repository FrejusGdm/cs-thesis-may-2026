#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1", "transformers==4.56.2", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "numpy", "scipy", "sentencepiece", "protobuf", "bitsandbytes", "snac"]
# ///
from __future__ import annotations
"""
T2-tokfix: Test the 'Llama BPE tokenizer is the bottleneck' hypothesis for Orpheus.

Last updated: 2026-04-21

Orpheus uses the same Llama 3.2 BPE backbone as CSM. Hypothesis: expanding the
tokenizer with native Adja tokens + resizing embeddings lets the LM learn
phoneme→audio mapping from real characters, not byte fragments. Otherwise
identical to T2_orpheus_finetune.py (English base).

Expected outcome: if T2-tokfix dev loss beats vanilla T2 English (best was 5.42),
the tokenizer WAS the bottleneck. If not, another layer is also broken.

References:
  - Three-layer TTS architecture: ideas/tts-models-guide.md
  - Failure analysis correction: learnings-from-the-past/training-gotchas.md
  - Orpheus: https://huggingface.co/canopylabs
  - SNAC codec: https://github.com/hubertsiuzdak/snac
"""
import argparse, json, os, random, sys, time, unicodedata
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

# Empirical Adja character set — see T1_csm_tokfix.py for derivation.
# IMPORTANT: In Orpheus, text token IDs 0..128255 are base vocab and special
# tokens start at 128256 (PAD at 128263, audio tokens at 128266). Added tokens
# land in the reserved range (Llama 3.2 reserves 256 slots from 128000-128255),
# so they should NOT collide with Orpheus's special-token block. Worst case
# the tokenizer extends past 128256 — we handle this with a resize + diagnostic.
ADJA_CHARS = [
    "ɔ", "ɛ", "ɖ", "ŋ", "Ŋ", "Ɖ", "Ɔ", "Ɛ",
    "é", "à", "è", "ê", "ó", "á", "ç", "ú", "ì", "ù", "ò", "ô", "â", "î", "û",
    "œ", "Ç", "É", "À", "Ê", "Á", "Ò", "Ô", "È", "ï", "ë", "í",
    "ɣ", "Ɣ", "ɤ", "ʒ", "Ʒ", "ƒ", "ɲ",
    "ǒ", "ǎ", "ǔ", "ǹ", "ń", "ĩ", "ă", "ī", "ŭ", "ö", "õ", "ĭ", "ū", "Ô",
    "\u0300", "\u0301", "\u0302", "\u0303", "\u0304", "\u030C", "\u0330",
    "ː", "ˈ",
    "kp", "gb", "ts", "dz", "ny", "ŋk", "ɖe", "ɖi",
]


def parse_args():
    p = argparse.ArgumentParser(description="T2-tokfix: Orpheus + expanded tokenizer → Adja")
    p.add_argument("--base-model", default="canopylabs/orpheus-3b-0.1-ft")
    p.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    p.add_argument("--output-dir", default="/tmp/orpheus_tokfix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T2_orpheus_tokfix")
    return p.parse_args()


def normalize_text(text):
    # NFKC: matches the normalization used to derive ADJA_CHARS.
    return " ".join(unicodedata.normalize("NFKC", text.strip()).split())


def build_snac_codes(audio_array, snac_model, orig_sr, device):
    import numpy as np, torch
    import torchaudio.transforms as T
    waveform = torch.from_numpy(np.asarray(audio_array, dtype=np.float32)).unsqueeze(0)
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


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np, soundfile as sf, torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               DataCollatorForLanguageModeling,
                               EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"Base: {args.base_model}\nGPU: {torch.cuda.get_device_name(0)}\n")

    # ===== Dataset =====
    ds = load_dataset(args.dataset, token=token, split="train")
    s1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    s2 = s1["train"].train_test_split(test_size=0.1/0.9, seed=args.seed)
    train_ds, dev_ds, test_ds = s2["train"], s2["test"], s1["test"]
    orig_sr = train_ds[0]["audio"]["sampling_rate"]
    print(f"Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)} | orig_sr={orig_sr}Hz\n")

    # ===== Tokenizer + expansion =====
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample = "Mì ɖo alɔ ji ŋkɔ nyé Tom ɛnyi wɛ dze è"
    tokens_before = tokenizer.tokenize(sample)
    vocab_before = len(tokenizer)
    print(f"[DIAGNOSTIC] Before expansion: vocab={vocab_before}")
    print(f"  '{sample}' → {tokens_before}\n")

    n_added = tokenizer.add_tokens(ADJA_CHARS)
    vocab_after = len(tokenizer)
    tokens_after = tokenizer.tokenize(sample)
    print(f"[DIAGNOSTIC] After expansion: added={n_added}, vocab={vocab_before}→{vocab_after}")
    print(f"  '{sample}' → {tokens_after}\n")

    tokenizer.pad_token_id = PAD_TOKEN  # keep Orpheus' special-token convention

    # ===== Model + embedding resize =====
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, token=token).to("cuda")
    # Orpheus vocab includes special/audio tokens at fixed offsets above 128256.
    # After add_tokens(), the new text token IDs go into the tokenizer's reserved
    # range (Llama 3.2 has 256 reserved slots starting at 128000) — they do NOT
    # collide with the audio-token block at AUDIO_TOKENS_START=128266.
    # Sanity-check and resize only if genuinely needed.
    needed_vocab = max(vocab_after, model.get_input_embeddings().num_embeddings)
    if needed_vocab > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(needed_vocab)
        print(f"Resized embeddings to {needed_vocab}\n")
    else:
        print(f"Embeddings already accommodate {needed_vocab} tokens (reserved range reused)\n")

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

    # ===== SNAC =====
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
    dropped = {"empty_text":0,"short_audio":0,"encode_fail":0,"too_long":0}

    def preprocess(ex):
        text = normalize_text(ex.get("text") or "")
        if not text:
            dropped["empty_text"] += 1
            return None
        arr = ex.get("audio",{}).get("array")
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
        ids = [START_OF_HUMAN]+text_ids+[END_OF_HUMAN]+[START_OF_AI]+[START_OF_SPEECH]+flat+[END_OF_SPEECH]+[END_OF_AI]
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
    snac_model.to("cpu"); torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir/"trainer_output"),
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

    # Save
    adapter_dir = output_dir/"adapter"
    model.save_pretrained(str(adapter_dir)); tokenizer.save_pretrained(str(adapter_dir))
    results = {
        "experiment": "T2_orpheus_tokfix",
        "base_model": args.base_model,
        "training_mode": training_mode,
        "tokenizer_expansion": {
            "chars_attempted": len(ADJA_CHARS),
            "chars_actually_added": n_added,
            "vocab_before": vocab_before,
            "vocab_after": vocab_after,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        },
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval, 4) if best_eval else None,
        "training_time_min": round(elapsed/60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "dropped": dropped,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.push_to_hub:
        api = HfApi(token=token)
        prefix = args.results_prefix
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                        path_in_repo=f"{prefix}/metrics.json",
                        repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(adapter_dir),
                          path_in_repo=f"{prefix}/adapter",
                          repo_id=args.results_repo, token=token)
        print(f"Done: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
