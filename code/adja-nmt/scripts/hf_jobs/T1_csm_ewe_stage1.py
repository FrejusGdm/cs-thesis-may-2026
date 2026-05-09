#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "transformers==4.52.3", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "scipy", "huggingface-hub>=0.34.0", "hf_transfer", "sentencepiece", "protobuf", "torchcodec", "bitsandbytes"]
# ///
from __future__ import annotations

"""
T1-ewe Stage 1: Fine-tune Sesame CSM (1B) on WaxalNLP Ewe TTS data.

Ewe is Adja's closest Gbe-family relative (same tonal system, similar phoneme
inventory). Stage 1 hypothesis: pre-adapting CSM to any Gbe-family language
before Adja adaptation should ease the phonological mapping in Stage 2.

Key decisions (same as T1_sesame_csm_finetune.py — proven working path):
  - CsmForConditionalGeneration + vanilla HF Trainer (no Unsloth)
  - LoRA on attention + MLP layers (gradient checkpointing OFF with PEFT)
  - Audio cast to 24kHz via datasets.Audio (Mimi target rate)
  - label_names=["labels"] fix so eval_loss is visible to Trainer

After training, the merged model is pushed to JosueG/adja-tts-checkpoints
so Stage 2 (T1_csm_ewe_adja_stage2.py) can load it without redownloading
the base checkpoint.

Data:
  google/WaxalNLP, config=ewe_tts
  train: 1215 rows  |  validation: 152 rows  |  test: 152 rows
  Audio: 48kHz raw → resampled to 24kHz via datasets cast_column

References:
  - Sesame CSM: https://github.com/SesameAILabs/csm
  - WaxalNLP: https://huggingface.co/datasets/google/WaxalNLP
  - Mimi codec: https://arxiv.org/abs/2410.00037
  - LoRA: https://arxiv.org/abs/2106.09685
  - Gbe cross-lingual hypothesis: experiments/asr-tts-getting-right-2026-04-21.md, Track 1A
"""

import argparse
import json
import os
import random
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1-ewe Stage 1: CSM on Ewe TTS")
    parser.add_argument("--dataset", default="google/WaxalNLP")
    parser.add_argument("--dataset-config", default="ewe_tts")
    parser.add_argument("--output-dir", default="/tmp/csm_ewe_stage1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T1_csm_ewe_stage1")
    parser.add_argument("--checkpoint-repo", default="JosueG/adja-tts-checkpoints",
                        help="Separate repo for merged model (used by Stage 2)")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        CsmForConditionalGeneration,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("T1-ewe Stage 1: Sesame CSM (1B) on WaxalNLP Ewe TTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset} / {args.dataset_config}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # ===== Load dataset (WaxalNLP has pre-existing splits) =====
    print("===== STAGE 1: Dataset =====")
    train_ds = load_dataset(args.dataset, name=args.dataset_config, split="train", token=token)
    dev_ds   = load_dataset(args.dataset, name=args.dataset_config, split="validation", token=token)
    test_ds  = load_dataset(args.dataset, name=args.dataset_config, split="test", token=token)

    # Detect text column — WaxalNLP may use "text" or "sentence"
    text_col = "text" if "text" in train_ds.column_names else "sentence"
    print(f"Columns: {train_ds.column_names} → text column: '{text_col}'")

    # Resample from 48kHz to 24kHz (Mimi target)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=24000))
    dev_ds   = dev_ds.cast_column("audio", Audio(sampling_rate=24000))
    test_ds  = test_ds.cast_column("audio", Audio(sampling_rate=24000))
    print(f"Train={len(train_ds)} | Dev={len(dev_ds)} | Test={len(test_ds)}\n")

    # ===== Load model =====
    print("===== STAGE 2: Model =====")
    model = CsmForConditionalGeneration.from_pretrained(
        "unsloth/csm-1b", torch_dtype=torch.float32,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained("unsloth/csm-1b")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: unsloth/csm-1b ({total_params / 1e6:.1f}M params)")

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

    # ===== Preprocess =====
    print("===== STAGE 3: Preprocess =====")
    MAX_AUDIO_SAMPLES = 240001  # 10s at 24kHz — clips longer than this are dropped

    def filter_by_length(ds, name: str):
        n_before = len(ds)
        ds_filtered = ds.filter(
            lambda ex: len(ex["audio"]["array"]) <= MAX_AUDIO_SAMPLES,
            desc=f"Filtering over-long {name} audio",
        )
        if len(ds_filtered) < n_before:
            print(f"  {name}: dropped {n_before - len(ds_filtered)} clips > {MAX_AUDIO_SAMPLES/24000:.1f}s")
        return ds_filtered

    train_ds = filter_by_length(train_ds, "train")
    dev_ds   = filter_by_length(dev_ds, "dev")

    def preprocess_example(example: dict) -> dict | None:
        text = normalize_text(example[text_col])
        conversation = [{
            "role": "0",
            "content": [
                {"type": "text", "text": text},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }]
        try:
            model_inputs = processor.apply_chat_template(
                conversation,
                tokenize=True, return_dict=True, output_labels=True,
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
    processed_dev   = dev_ds.map(preprocess_example, remove_columns=dev_ds.column_names, desc="Preprocessing dev")
    processed_dev   = processed_dev.filter(lambda x: x.get("input_ids") is not None)
    print(f"Preprocessed: train={len(processed_train)} dev={len(processed_dev)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Train =====
    print("===== STAGE 4: Train =====")
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
        greater_is_better=False, remove_unused_columns=False,
        label_names=["labels"],
        # gradient_checkpointing OFF — breaks grad flow with PEFT
    )

    trainer = Trainer(
        model=model, train_dataset=processed_train, eval_dataset=processed_dev,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(trainer_stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
    best_eval_loss = min((l["eval_loss"] for l in eval_history), default=None)
    final_eval_loss = float(trainer.evaluate().get("eval_loss", 0.0))

    print(f"\nDone in {elapsed / 60:.1f} min | train_loss={train_loss:.4f} | "
          f"best_eval={best_eval_loss} | VRAM={peak_mem}GB\n")

    # ===== Generate Ewe samples =====
    print("===== STAGE 5: Generate =====")
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    model.eval()

    generated: list[dict] = []
    for i in range(min(5, len(test_ds))):
        text = normalize_text(test_ds[i][text_col])
        try:
            plain_inputs = processor(
                f"[0]{text}", add_special_tokens=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                audio_values = model.generate(**plain_inputs, max_new_tokens=125, output_audio=True)
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / f"ewe_{i:02d}.wav"
            sf.write(str(path), audio, 24000)
            generated.append({"text": text, "file": path.name, "duration_sec": round(len(audio) / 24000, 2)})
            print(f"  [{i}] '{text}' -> {len(audio) / 24000:.1f}s")
        except Exception as exc:
            print(f"  [{i}] FAILED: {exc}")
            generated.append({"text": text, "error": str(exc)})

    # ===== Save adapter =====
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    # Merge LoRA weights into base model for Stage 2 loading
    if not args.full_finetune:
        print("Merging LoRA weights for Stage 2 checkpoint...")
        merged = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        print(f"Merged model saved to {merged_dir}")
    else:
        merged_dir = adapter_dir  # full finetune is already merged

    results = {
        "experiment": "T1_csm_ewe_stage1",
        "base_model": "unsloth/csm-1b",
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "training_mode": training_mode,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval_loss, 4) if best_eval_loss else None,
        "final_eval_loss": round(final_eval_loss, 4),
        "training_time_min": round(elapsed / 60, 1),
        "peak_vram_gb": peak_mem,
        "n_train": len(processed_train),
        "n_dev": len(processed_dev),
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
        "stage2_checkpoint": f"{args.checkpoint_repo}/T1_csm_ewe_stage1",
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Metrics saved to {metrics_path}")

    if args.push_to_hub:
        print("\n===== STAGE 6: Push to Hub =====")
        api = HfApi(token=token)
        prefix = args.results_prefix

        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                        path_in_repo=f"{prefix}/metrics.json",
                        repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(adapter_dir),
                          path_in_repo=f"{prefix}/adapter",
                          repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(generated_dir),
                          path_in_repo=f"{prefix}/generated_audio",
                          repo_id=args.results_repo, token=token)

        # Push merged model to checkpoint repo for Stage 2
        api.create_repo(args.checkpoint_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(merged_dir),
                          path_in_repo="T1_csm_ewe_stage1",
                          repo_id=args.checkpoint_repo, token=token)
        print(f"Checkpoint pushed to {args.checkpoint_repo}/T1_csm_ewe_stage1")
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    main()
