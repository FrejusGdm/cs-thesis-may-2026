#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "transformers==4.52.3", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "scipy", "huggingface-hub>=0.34.0", "hf_transfer", "sentencepiece", "protobuf", "torchcodec", "bitsandbytes"]
# ///
from __future__ import annotations
"""
T1-ewe Stage 2: Adapt an Ewe-tuned Sesame CSM to Adja TTS.

Last updated: 2026-04-21

Loads the Stage 1 merged checkpoint (CSM fine-tuned on WaxalNLP Ewe TTS) and
continues training on JosueG/adja-tts-orpheus. Hypothesis: Gbe-family acoustic
prior acquired in Stage 1 improves Adja adaptation quality.

Prerequisites:
  - T1_csm_ewe_stage1.py must have completed and pushed its merged model to
    JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1

References:
  - Sesame CSM: https://github.com/SesameAILabs/csm
  - Mimi codec: https://arxiv.org/abs/2410.00037
  - Cross-lingual TTS transfer: experiments/asr-tts-getting-right-2026-04-21.md, Track 1A
"""

import argparse, json, os, random, subprocess, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    p = argparse.ArgumentParser(description="T1-ewe Stage 2: CSM Ewe → Adja")
    p.add_argument("--stage1-checkpoint",
                   default="JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1",
                   help="Hub path (repo_id/subfolder) to Stage 1 merged model")
    p.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    p.add_argument("--output-dir", default="/tmp/csm_ewe_adja_stage2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5,
                   help="Lower LR for Stage 2 (vs 2e-4 in Stage 1)")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T1_csm_ewe_adja_stage2")
    return p.parse_args()


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, CsmForConditionalGeneration,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # ===== Download Stage 1 merged checkpoint =====
    stage1_repo, stage1_subdir = args.stage1_checkpoint.rsplit("/", 1)
    print(f"Downloading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    snapshot_download(stage1_repo, local_dir="/tmp/stage1_csm_ewe",
                      allow_patterns=f"{stage1_subdir}/*", token=token)
    stage1_local = f"/tmp/stage1_csm_ewe/{stage1_subdir}"

    # ===== Adja dataset (80/10/10 split, seed=42) =====
    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"].cast_column("audio", Audio(sampling_rate=24000))
    dev_ds = split2["test"].cast_column("audio", Audio(sampling_rate=24000))
    test_ds = split1["test"].cast_column("audio", Audio(sampling_rate=24000))
    print(f"Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)}\n")

    # ===== Load model from Stage 1 =====
    print(f"Loading Stage 1 model from {stage1_local}...")
    model = CsmForConditionalGeneration.from_pretrained(stage1_local, torch_dtype=torch.float32).to("cuda")
    # Load processor from the BASE model, not the merged Stage 1 save. The merged
    # save loses config that prevents text_kwargs (pad_to_multiple_of) from being
    # forwarded to the EncodecFeatureExtractor — causes every example to be skipped
    # with: EncodecFeatureExtractor.__call__() got an unexpected keyword argument 'pad_to_multiple_of'.
    processor = AutoProcessor.from_pretrained("unsloth/csm-1b", token=token)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 model loaded ({total_params / 1e6:.1f}M params)")

    if args.full_finetune:
        training_mode = "full_finetune"
    else:
        training_mode = f"lora_r{args.lora_r}"
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none",
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M\n")

    # ===== Preprocess (same as Stage 1) =====
    MAX_AUDIO_SAMPLES = 240001

    def filter_by_length(ds, name):
        n_before = len(ds)
        ds_filtered = ds.filter(lambda ex: len(ex["audio"]["array"]) <= MAX_AUDIO_SAMPLES,
                                desc=f"Filtering over-long {name}")
        if len(ds_filtered) < n_before:
            print(f"  {name}: dropped {n_before - len(ds_filtered)} clips > 10s")
        return ds_filtered

    train_ds = filter_by_length(train_ds, "train")
    dev_ds = filter_by_length(dev_ds, "dev")

    def preprocess_example(example):
        text = normalize_text(example["text"])
        conversation = [{
            "role": "0",
            "content": [
                {"type": "text", "text": text},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }]
        try:
            model_inputs = processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, output_labels=True,
                text_kwargs={"padding": "max_length", "max_length": 256,
                             "pad_to_multiple_of": 8, "padding_side": "right"},
                audio_kwargs={"sampling_rate": 24000, "max_length": 240001, "padding": "max_length"},
                common_kwargs={"return_tensors": "pt"},
            )
        except Exception as exc:
            print(f"  Skip '{text[:40]}': {exc}")
            return None
        required = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        if any(k not in model_inputs for k in required):
            return None
        return {k: model_inputs[k][0] for k in required}

    processed_train = train_ds.map(preprocess_example, remove_columns=train_ds.column_names, desc="Preprocessing train")
    processed_train = processed_train.filter(lambda x: x.get("input_ids") is not None)
    processed_dev = dev_ds.map(preprocess_example, remove_columns=dev_ds.column_names, desc="Preprocessing dev")
    processed_dev = processed_dev.filter(lambda x: x.get("input_ids") is not None)
    print(f"Preprocessed: train={len(processed_train)} dev={len(processed_dev)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10, num_train_epochs=args.num_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5, optim="adamw_torch", weight_decay=0.001,
        lr_scheduler_type="cosine", seed=args.seed, report_to="none",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, train_dataset=processed_train, eval_dataset=processed_dev,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
    best_eval = min((l["eval_loss"] for l in eval_history), default=None)
    print(f"Done {elapsed/60:.1f}min | train_loss={train_loss:.4f} | best_eval={best_eval}\n")

    # ===== Generate Adja samples =====
    model.eval()
    generated = []
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    for i in range(min(5, len(test_ds))):
        text = normalize_text(test_ds[i]["text"])
        try:
            plain_inputs = processor(f"[0]{text}", add_special_tokens=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                audio_values = model.generate(**plain_inputs, max_new_tokens=125, output_audio=True)
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / f"adja_{i:02d}.wav"
            sf.write(str(path), audio, 24000)
            generated.append({"text": text, "file": path.name, "duration_sec": round(len(audio)/24000, 2)})
        except Exception as exc:
            generated.append({"text": text, "error": str(exc)})

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    results = {
        "experiment": "T1_csm_ewe_adja_stage2",
        "stage1_checkpoint": args.stage1_checkpoint,
        "adja_dataset": args.dataset,
        "training_mode": training_mode,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval, 4) if best_eval else None,
        "training_time_min": round(elapsed/60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
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
    main()
