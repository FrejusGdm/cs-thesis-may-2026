#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.6.0", "torchaudio==2.6.0", "transformers==4.56.2", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "numpy", "scipy", "sentencepiece", "protobuf", "bitsandbytes", "snac"]
# ///
from __future__ import annotations
"""
T2-ewe Stage 1: Fine-tune an Orpheus 3B variant on WaxalNLP Ewe TTS data.

Last updated: 2026-04-21

This is the Orpheus analog of T1_csm_ewe_stage1.py and T3_spark_ewe_stage1.py.
Ewe is Adja's closest Gbe-family relative. Stage 1 hypothesis: priming Orpheus
on Ewe TTS before Adja should give the LM backbone Gbe-family phonological
priors that are missing from the English/French base, and complementary to the
tonal-only prior the Chinese base already has.

Parameterized for all three Orpheus bases:
  - English:  canopylabs/orpheus-3b-0.1-ft                    (tag: en)
  - Chinese:  canopylabs/3b-zh-ft-research_release (tag: zh)
  - French:   canopylabs/3b-fr-ft-research_release            (tag: fr)

After training, the merged model is pushed to
  JosueG/adja-tts-checkpoints/T2_orpheus_{tag}_ewe_stage1
so Stage 2 (T2_orpheus_ewe_adja_stage2.py) can load it.

References:
  - Orpheus TTS: https://huggingface.co/canopylabs
  - SNAC codec (language-agnostic): https://github.com/hubertsiuzdak/snac
  - WaxalNLP: https://huggingface.co/datasets/google/WaxalNLP
  - LoRA: https://arxiv.org/abs/2106.09685
  - Gbe cross-lingual hypothesis: experiments/asr-tts-getting-right-2026-04-21.md, Track 1B
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


def parse_args():
    p = argparse.ArgumentParser(description="T2-ewe Stage 1: Orpheus → Ewe TTS")
    p.add_argument("--base-model", default="canopylabs/orpheus-3b-0.1-ft",
                   help="Orpheus base checkpoint; use en/zh/fr variants")
    p.add_argument("--checkpoint-tag", default="en",
                   help="Tag used in the output checkpoint path (en|zh|fr|...)")
    p.add_argument("--dataset", default="google/WaxalNLP")
    p.add_argument("--dataset-config", default="ewe_tts")
    p.add_argument("--output-dir", default="/tmp/orpheus_ewe_stage1")
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
    p.add_argument("--results-prefix", default="T2_orpheus_en_ewe_stage1")
    p.add_argument("--checkpoint-repo", default="JosueG/adja-tts-checkpoints",
                   help="Repo to push merged model to for Stage 2 loading")
    return p.parse_args()


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


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

    print(f"Base model: {args.base_model}  (tag: {args.checkpoint_tag})")
    print(f"Dataset: {args.dataset}/{args.dataset_config}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # ===== Dataset (WaxalNLP has train/validation/test splits for ewe_tts) =====
    train_ds = load_dataset(args.dataset, name=args.dataset_config, split="train", token=token)
    dev_ds   = load_dataset(args.dataset, name=args.dataset_config, split="validation", token=token)
    test_ds  = load_dataset(args.dataset, name=args.dataset_config, split="test", token=token)

    text_col = "text" if "text" in train_ds.column_names else "sentence"
    print(f"Columns: {train_ds.column_names} → text column: '{text_col}'")
    orig_sr = train_ds[0]["audio"]["sampling_rate"]
    print(f"Train={len(train_ds)} | Dev={len(dev_ds)} | Test={len(test_ds)} | orig_sr={orig_sr}Hz\n")

    # ===== Model =====
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, token=token).to("cuda")
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
        model.enable_input_require_grads()  # required: gradient_checkpointing + PEFT breaks grad graph without this
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable/1e6:.2f}M / {total_params/1e6:.1f}M\n")

    # ===== SNAC encode (language-agnostic, proven by Mandarin Orpheus) =====
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
    dropped = {"empty_text":0,"short_audio":0,"encode_fail":0,"too_long":0}

    def preprocess(ex):
        text = normalize_text(ex.get(text_col) or "")
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
    if len(processed_dev) == 0 and len(processed_train) > 0:
        print(
            f"[WARN] Encoded dev=0 (all sequences > --max-seq-length={args.max_seq_length}). "
            "Falling back to a small eval slice from train so HF Trainer can compute eval_loss."
        )
        with_len = processed_train.map(lambda ex: {"seq_len": len(ex["input_ids"])}, desc="Compute seq_len")
        processed_dev = with_len.sort("seq_len").select(range(min(64, len(with_len))))
        processed_train = with_len.remove_columns(["seq_len"])
        processed_dev = processed_dev.remove_columns(["seq_len"])

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
        gradient_checkpointing=True,
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

    # ===== Save adapter + merged model for Stage 2 =====
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if not args.full_finetune:
        print("Merging LoRA for Stage 2 checkpoint...")
        merged = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
    else:
        merged_dir = adapter_dir

    results = {
        "experiment": f"T2_orpheus_{args.checkpoint_tag}_ewe_stage1",
        "base_model": args.base_model,
        "checkpoint_tag": args.checkpoint_tag,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "training_mode": training_mode,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval, 4) if best_eval else None,
        "training_time_min": round(elapsed/60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "dropped": dropped,
        "stage2_checkpoint": f"{args.checkpoint_repo}/T2_orpheus_{args.checkpoint_tag}_ewe_stage1",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Metrics saved to {metrics_path}")

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

        # Push merged model to checkpoint repo for Stage 2
        api.create_repo(args.checkpoint_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(merged_dir),
                          path_in_repo=f"T2_orpheus_{args.checkpoint_tag}_ewe_stage1",
                          repo_id=args.checkpoint_repo, token=token)
        print(f"Checkpoint pushed to {args.checkpoint_repo}/T2_orpheus_{args.checkpoint_tag}_ewe_stage1")
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
