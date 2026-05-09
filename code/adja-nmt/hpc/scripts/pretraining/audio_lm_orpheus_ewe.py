#!/usr/bin/env python3
from __future__ import annotations
"""
Audio-LM pretraining: Orpheus backbone on SNAC codes from unlabeled Ewe → fine-tune.

Last updated: 2026-04-21

Pipeline mirrors audio_lm_csm_ewe.py but uses Orpheus + SNAC instead of CSM + Mimi.
SNAC is proven language-agnostic by the working Mandarin Orpheus variant
(`canopylabs/3b-zh-ft-research_release`), so we have more
confidence that the audio-LM objective will produce clean tokens for Adja.

Stage A: encode 183k unlabeled Ewe utterances with SNAC and train the Orpheus
LM backbone on pure next-token prediction over SNAC codes.
Stage B: supervised fine-tune on Ewe TTS + Adja TTS (text-to-audio).

Which Orpheus base: we default to the Mandarin tonal pretrain, since Adja is
tonal. English or French base can be set via --base if you want to compare.

References:
  - Orpheus: https://huggingface.co/canopylabs
  - SNAC: https://github.com/hubertsiuzdak/snac
  - Audio LM pretraining rationale: AudioLM/VALL-E family

Runtime (A100 80GB): Stage A ~18-24h for 183k utts, Stage B ~6-8h.
"""
import argparse, json, os, random, subprocess, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

from waxalnlp_loader import load_waxal_split, safe_load_audio_array

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
    p = argparse.ArgumentParser(description="Orpheus audio-LM pretraining on Ewe")
    p.add_argument("--models-dir", default="/models")
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--output-dir", default="/results/audiolm_orpheus_ewe")
    p.add_argument("--base", default="canopylabs/3b-zh-ft-research_release",
                   help="Orpheus base; zh for tonal prior, en/fr as controls")
    p.add_argument("--snac", default="hubertsiuzdak/snac_24khz")
    p.add_argument("--stage", choices=["pretrain", "finetune", "both"], default="both")
    p.add_argument("--pretrain-epochs", type=int, default=2)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--pretrain-batch", type=int, default=1)
    p.add_argument("--pretrain-grad-accum", type=int, default=8)
    p.add_argument("--pretrain-lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--finetune-grad-accum", type=int, default=8)
    p.add_argument("--max-audio-sec", type=float, default=10.0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--low-vram-max-seq-length", type=int, default=1024,
                   help="Max sequence length to use automatically on 40GB MIG slices.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-key", default="orpheus-3b")
    p.add_argument("--preprocess-proc", type=int, default=min(8, (os.cpu_count() or 1)),
                   help="Processes for dataset preprocessing (decode+SNAC encode). "
                        "Falls back to 1 if multiprocessing fails.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dry run: 4 samples, 2 steps, Stage A only, no save.")
    return p.parse_args()


def resolve(args, repo):
    safe = repo.replace("/", "__")
    candidate = Path(args.models_dir) / safe
    return str(candidate) if candidate.exists() else repo


def build_snac_codes(audio_array, snac_model, orig_sr, device):
    import numpy as np, torch
    import librosa  # lighter than torchaudio, works without a .so at runtime
    arr = np.asarray(audio_array, dtype=np.float32)
    if orig_sr != 24000:
        arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=24000)
    waveform = torch.from_numpy(arr).unsqueeze(0)
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


def dedupe(codes):
    if len(codes) % 7 != 0:
        raise ValueError("codes not divisible by 7")
    kept = codes[:7]
    for i in range(7, len(codes), 7):
        if codes[i] != kept[-7]:
            kept.extend(codes[i:i+7])
    return kept


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    import numpy as np
    import torch
    from datasets import Audio, Features, Sequence, Value, concatenate_datasets, load_dataset
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_out = output_dir / "stage_a_pretrain"
    finetune_out = output_dir / "stage_b_finetune"

    base_path = resolve(args, args.base)
    snac_path = resolve(args, args.snac)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Base: {base_path}\nSNAC: {snac_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB\n")

    if vram_gb < 60:
        old_seq, old_r = args.max_seq_length, args.lora_r
        args.max_seq_length = min(args.max_seq_length, args.low_vram_max_seq_length)
        args.lora_r = min(args.lora_r, 16)
        print(f"[auto-fit] low VRAM detected ({vram_gb:.1f}GB): "
              f"max_seq_length {old_seq}->{args.max_seq_length}, lora_r {old_r}->{args.lora_r}")

    if args.smoke:
        print(">>> SMOKE MODE: 4 samples, 2 steps, Stage A only, no save <<<\n")
        args.stage = "pretrain"
        args.pretrain_epochs = 1

    def normalize(t):
        return " ".join(unicodedata.normalize("NFKC", t.strip()).split())

    def configure_lm_for_training(model):
        model.config.use_cache = False
        if vram_gb < 60:
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
            model.gradient_checkpointing_enable()
        return model

    # --------------------------------------------------------------
    # Stage A — audio-only LM pretraining on unlabeled Ewe SNAC codes
    # --------------------------------------------------------------
    if args.stage in ("pretrain", "both"):
        print("=" * 60)
        print("STAGE A — Orpheus LM on SNAC codes (unlabeled Ewe)")
        print("=" * 60)

        unlab = load_waxal_split(
            config="ewe_asr",
            split="unlabeled",
            cache_dir=str(Path(args.data_dir) / "WaxalNLP_ewe_asr"),
            token=hf_token,
        )
        if args.smoke:
            unlab = unlab.select(range(4))
        max_samples = int(args.max_audio_sec * 24000)
        # Avoid eager Audio decoding: some unlabeled blobs fail libsndfile.
        unlab = unlab.cast_column("audio", Audio(decode=False))
        n_total = len(unlab)
        print(f"Unlabeled Ewe utts (raw): {n_total}")

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = PAD_TOKEN
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).to("cuda")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        ))
        model = configure_lm_for_training(model)

        snac_model = SNAC.from_pretrained(snac_path).to("cuda").eval()

        out_features = Features({
            "_ok": Value("bool"),
            "input_ids": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
        })

        def encode_ssl(ex):
            arr = safe_load_audio_array(ex.get("audio"), target_sr=24000)
            if arr is None or len(arr) < 10000 or len(arr) > max_samples:
                return {"_ok": False,
                        "input_ids": np.asarray([], dtype=np.int64),
                        "labels": np.asarray([], dtype=np.int64),
                        "attention_mask": np.asarray([], dtype=np.int64)}
            try:
                flat = build_snac_codes(arr, snac_model, 24000, "cuda")
            except Exception:
                return {"_ok": False,
                        "input_ids": np.asarray([], dtype=np.int64),
                        "labels": np.asarray([], dtype=np.int64),
                        "attention_mask": np.asarray([], dtype=np.int64)}
            flat = dedupe(flat)
            # No text; just the audio frames wrapped with speech delimiters.
            budget = args.max_seq_length - 4
            if budget <= 0:
                return {"_ok": False,
                        "input_ids": np.asarray([], dtype=np.int64),
                        "labels": np.asarray([], dtype=np.int64),
                        "attention_mask": np.asarray([], dtype=np.int64)}
            flat = flat[:budget]
            ids = [START_OF_AI, START_OF_SPEECH] + flat + [END_OF_SPEECH, END_OF_AI]
            ids_arr = np.asarray(ids, dtype=np.int64)
            return {"_ok": True,
                    "input_ids": ids_arr,
                    "labels": ids_arr,
                    "attention_mask": np.ones((len(ids),), dtype=np.int64)}

        try:
            encoded = unlab.map(
                encode_ssl,
                remove_columns=unlab.column_names,
                features=out_features,
                num_proc=max(1, int(args.preprocess_proc)),
                desc="decode+SNAC encode",
            )
        except Exception as exc:
            print(f"[warn] preprocessing multiprocessing failed ({exc}); retrying with --preprocess-proc 1",
                  file=__import__('sys').stderr)
            encoded = unlab.map(
                encode_ssl,
                remove_columns=unlab.column_names,
                features=out_features,
                num_proc=1,
                desc="decode+SNAC encode",
            )
        encoded = encoded.filter(lambda x: x["_ok"], desc="drop bad/long")
        encoded = encoded.remove_columns(["_ok"])
        n_ok = len(encoded)
        print(f"Usable unlabeled samples: {n_ok}/{n_total}")
        if n_ok == 0:
            try:
                a = unlab[0].get("audio") if n_total else None
                raw = (a or {}).get("bytes") or b""
                magic = raw[:8]
                path = (a or {}).get("path")
                keys = list((a or {}).keys()) if isinstance(a, dict) else []
                print(f"[fatal] decoded 0/{n_total}. audio keys={keys} path={path!r} bytes_magic={magic!r}",
                      file=__import__('sys').stderr)
            except Exception:
                print(f"[fatal] decoded 0/{n_total}. Could not inspect first example.",
                      file=__import__('sys').stderr)
            raise SystemExit("No usable unlabeled audio after decoding; check ffmpeg bytes decode.")
        snac_model.to("cpu"); torch.cuda.empty_cache()

        training_args = TrainingArguments(
            output_dir=str(pretrain_out / "trainer"),
            per_device_train_batch_size=args.pretrain_batch,
            gradient_accumulation_steps=1 if args.smoke else args.pretrain_grad_accum,
            num_train_epochs=args.pretrain_epochs,
            max_steps=2 if args.smoke else -1,
            learning_rate=args.pretrain_lr,
            warmup_steps=0 if args.smoke else 1000,
            lr_scheduler_type="linear",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1 if args.smoke else 100,
            save_strategy="no" if args.smoke else "epoch", save_total_limit=2,
            optim="adamw_8bit", seed=args.seed, report_to="none",
            remove_unused_columns=False, label_names=["labels"],
        )
        trainer = Trainer(
            model=model, args=training_args, train_dataset=encoded,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        if args.smoke:
            peak_gb = torch.cuda.max_memory_reserved() / 1e9
            print(f"SMOKE PASS: audio_lm_orpheus_ewe | peak VRAM {peak_gb:.1f}GB | {elapsed:.1f}s")
            return
        # Merge LoRA before Stage B starts
        merged = model.merge_and_unload()
        merged.save_pretrained(str(pretrain_out / "backbone"))
        tokenizer.save_pretrained(str(pretrain_out / "backbone"))
        print(f"Stage A done: {elapsed/60:.1f}min\n")

    # --------------------------------------------------------------
    # Stage B — supervised text→audio on labeled Ewe + Adja
    # --------------------------------------------------------------
    if args.stage in ("finetune", "both"):
        print("=" * 60)
        print("STAGE B — supervised Orpheus on Ewe + Adja TTS")
        print("=" * 60)

        start_path = str(pretrain_out / "backbone") if args.stage == "both" else base_path

        ewe = load_waxal_split(
            config="ewe_tts",
            split="train",
            cache_dir=str(Path(args.data_dir) / "WaxalNLP_ewe_tts"),
            token=hf_token,
        )
        ewe = ewe.cast_column("audio", Audio(sampling_rate=24000))
        if "sentence" in ewe.column_names and "text" not in ewe.column_names:
            ewe = ewe.rename_column("sentence", "text")
        adja = load_dataset("JosueG/adja-tts-orpheus", split="train",
                            cache_dir=str(Path(args.data_dir) / "JosueG_adja-tts-orpheus"),
                            token=hf_token)
        adja = adja.cast_column("audio", Audio(sampling_rate=24000))

        combined = concatenate_datasets([
            ewe.remove_columns([c for c in ewe.column_names if c not in ("audio", "text")]),
            adja.remove_columns([c for c in adja.column_names if c not in ("audio", "text")]),
        ])
        split = combined.train_test_split(test_size=0.1, seed=args.seed)
        train_ds, dev_ds = split["train"], split["test"]

        tokenizer = AutoTokenizer.from_pretrained(start_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = PAD_TOKEN
        model = AutoModelForCausalLM.from_pretrained(start_path, torch_dtype=torch.float32).to("cuda")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        ))
        model = configure_lm_for_training(model)

        snac_model = SNAC.from_pretrained(snac_path).to("cuda").eval()

        def prep(ex):
            text = normalize(ex["text"])
            if not text:
                return None
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            if len(arr) < 10000:
                return None
            try:
                flat = build_snac_codes(arr, snac_model, 24000, "cuda")
            except Exception:
                return None
            flat = dedupe(flat)
            text_ids = tokenizer.encode(text, add_special_tokens=True) + [END_OF_TEXT]
            audio_budget = args.max_seq_length - (len(text_ids) + 6)
            if audio_budget <= 0:
                return None
            flat = flat[:audio_budget]
            ids = [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH] + flat + [END_OF_SPEECH, END_OF_AI]
            if len(ids) < 8:
                return None
            return {"input_ids": ids, "labels": ids, "attention_mask": [1]*len(ids)}

        train_p = train_ds.map(prep, remove_columns=train_ds.column_names, desc="prep train")
        train_p = train_p.filter(lambda x: x.get("input_ids") is not None)
        dev_p = dev_ds.map(prep, remove_columns=dev_ds.column_names, desc="prep dev")
        dev_p = dev_p.filter(lambda x: x.get("input_ids") is not None)

        snac_model.to("cpu"); torch.cuda.empty_cache()

        do_eval = vram_gb >= 60
        training_args = TrainingArguments(
            output_dir=str(finetune_out / "trainer"),
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if args.smoke else args.finetune_grad_accum,
            num_train_epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            warmup_steps=50, lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            eval_strategy="steps" if do_eval else "no",
            eval_steps=50,
            save_strategy="steps" if do_eval else "epoch",
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=do_eval,
            metric_for_best_model="eval_loss",
            greater_is_better=False, optim="adamw_8bit",
            seed=args.seed, report_to="none", remove_unused_columns=False,
            label_names=["labels"],
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_p, eval_dataset=dev_p,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if do_eval else [],
        )
        t0 = time.time()
        stats = trainer.train()
        elapsed = time.time() - t0
        train_loss = float(stats.metrics.get("train_loss", 0.0))
        eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
        best_eval = min((l["eval_loss"] for l in eval_history), default=None)

        model.save_pretrained(str(finetune_out / "adapter"))
        tokenizer.save_pretrained(str(finetune_out / "adapter"))

        results = {
            "experiment": "audiolm_orpheus_ewe",
            "base": args.base,
            "stage": args.stage,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "train_loss": round(train_loss, 4),
            "best_eval_loss": round(best_eval, 4) if best_eval else None,
            "finetune_time_min": round(elapsed/60, 1),
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_reserved()/1e9, 2),
        }
        (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print("All done.")


if __name__ == "__main__":
    main()
