#!/usr/bin/env python3
from __future__ import annotations

"""
T1 canonical: Fine-tune Sesame CSM (1B) on Adja TTS via HuggingFace Jobs.

This is the WORKING path. It does NOT use Unsloth, because Unsloth's CSM
forward-pass patch (as of 2026.4.x) was broken for our data shape and the
older pinned versions (2025.5.4) keep hitting Colab/HF-Jobs resolver issues.

Instead it uses vanilla HuggingFace:
    - CsmForConditionalGeneration (first-class transformers model)
    - peft.LoraConfig + get_peft_model (standard LoRA)
    - transformers.Trainer (standard training loop)

Run shape:
    - Container: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
    - GPU: any with >= 24 GB VRAM (tested: L40S 48GB, A100 80GB)
    - Model loaded in fp32, Trainer handles bf16 mixed precision
    - Gradient checkpointing OFF (PEFT + checkpointing = grad flow breakage)

For the short 2-step sanity check use scripts/hf_jobs/T1_csm_vanilla.py.
This script does the FULL training run + generation + Hub upload.

References:
    - Sesame CSM: https://github.com/SesameAILabs/csm
    - Mimi codec (24 kHz): https://arxiv.org/abs/2410.00037
    - LoRA: https://arxiv.org/abs/2106.09685
    - Upstream Unsloth notebook:
      references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full Sesame CSM training for Adja")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus", help="Private HF dataset id")
    parser.add_argument("--output-dir", default="/tmp/csm_adja_full", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Split + training seed")
    parser.add_argument("--max-steps", type=int, default=-1, help="Hard step cap (default -1 = use --num-epochs)")
    parser.add_argument("--num-epochs", type=int, default=20, help="Max epochs (early stopping usually fires first)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = bs * grad_accum)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (ignored if --full-finetune)")
    parser.add_argument("--full-finetune", action="store_true", help="Skip LoRA, update all 1.66B params. Tests if LoRA r=32 was the ceiling.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate on dev set every N steps")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Stop if eval_loss doesn't improve for N evals")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload metrics, adapter, and audio samples")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results", help="HF repo for outputs")
    parser.add_argument("--results-prefix", default="T1", help="Subdirectory inside the results repo")
    return parser.parse_args()


def pip_install(packages: list[str], no_deps: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_env() -> None:
    pip_install(
        [
            "transformers==4.52.3",
            "peft>=0.11.0,<0.16.0",
            "accelerate",
            "datasets>=3.4.1,<4.0.0",
            "soundfile",
            "librosa",
            "numpy",
            "scipy",
            "huggingface_hub>=0.34.0",
            "hf_transfer",
            "sentencepiece",
            "protobuf",
            "torchcodec",
            "bitsandbytes",
        ]
    )


def normalize_text(text: str) -> str:
    """NFC normalize — critical for Adja tone marks (ɛ, ɔ, ŋ, ɖ, é, è)."""
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    print("Installing environment...")
    install_env()
    print("Environment ready.\n")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        CsmForConditionalGeneration,
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
    print("T1 CANONICAL: Sesame CSM (1B) full training on Adja")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {args.seed}")
    print(f"LoRA r: {args.lora_r}")
    steps_desc = f"{args.max_steps} hard-capped" if args.max_steps > 0 else f"up to {args.num_epochs} epochs"
    print(f"Training: {steps_desc}, batch {args.batch_size} x grad_accum {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Eval every {args.eval_steps} steps, early stop patience {args.early_stopping_patience}")
    print()

    # ===== Load + split dataset =====
    print("===== STAGE 1: Dataset =====")
    ds = load_dataset(args.dataset, token=token, split="train")
    # Same 80/10/10 protocol as ASR experiments (seed=42)
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"].cast_column("audio", Audio(sampling_rate=24000))
    dev_ds = split2["test"].cast_column("audio", Audio(sampling_rate=24000))
    test_ds = split1["test"].cast_column("audio", Audio(sampling_rate=24000))
    print(f"Train samples: {len(train_ds)}")
    print(f"Dev samples: {len(dev_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print()

    # ===== Load CSM base model in fp32 =====
    # NOTE: loading in bf16 causes index_put_ dtype mismatch inside CSM audio processing.
    #   Trainer handles bf16 mixed precision via autocast; base weights stay fp32.
    print("===== STAGE 2: Model =====")
    model = CsmForConditionalGeneration.from_pretrained(
        "unsloth/csm-1b",
        torch_dtype=torch.float32,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained("unsloth/csm-1b")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: unsloth/csm-1b ({total_params / 1e6:.1f}M params)")

    if args.full_finetune:
        print("Mode: FULL fine-tune (all params trainable, no LoRA)")
        training_mode = "full_finetune"
        # All params already require_grad=True after from_pretrained; nothing to do.
    else:
        training_mode = f"lora_r{args.lora_r}"
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M ({100 * trainable / total_params:.2f}%)\n")

    # ===== Preprocess =====
    print("===== STAGE 3: Preprocess =====")

    # Max audio length that fits the `audio_kwargs["max_length"]=240001` slot (10s at 24kHz).
    # Adja clips can reach ~14s; those get dropped so the default data collator can stack tensors.
    MAX_AUDIO_SAMPLES = 240001

    # Drop over-long clips BEFORE map(). Filtering inside map() via `return None` is
    # unreliable with HF Datasets — the schema wants a real dict back every time.
    def filter_by_length(ds, name: str):
        n_before = len(ds)
        ds_filtered = ds.filter(
            lambda ex: len(ex["audio"]["array"]) <= MAX_AUDIO_SAMPLES,
            desc=f"Filtering over-long {name} audio",
        )
        n_after = len(ds_filtered)
        if n_after < n_before:
            print(f"  {name}: dropped {n_before - n_after} clips > {MAX_AUDIO_SAMPLES/24000:.1f}s (kept {n_after})")
        return ds_filtered

    train_ds = filter_by_length(train_ds, "train")
    dev_ds = filter_by_length(dev_ds, "dev")

    def preprocess_example(example: dict) -> dict | None:
        text = normalize_text(example["text"])
        conversation = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "path": example["audio"]["array"]},
                ],
            }
        ]
        try:
            model_inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                text_kwargs={
                    "padding": "max_length",
                    "max_length": 256,
                    "pad_to_multiple_of": 8,
                    "padding_side": "right",
                },
                audio_kwargs={
                    "sampling_rate": 24000,
                    "max_length": 240001,
                    "padding": "max_length",
                },
                common_kwargs={"return_tensors": "pt"},
            )
        except Exception as exc:
            print(f"  Skip '{text[:40]}': {exc}")
            return None

        required = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        missing = [key for key in required if key not in model_inputs]
        if missing:
            return None
        return {key: model_inputs[key][0] for key in required}

    processed_train = train_ds.map(
        preprocess_example,
        remove_columns=train_ds.column_names,
        desc="Preprocessing train",
    )
    processed_train = processed_train.filter(lambda x: x.get("input_ids") is not None)
    processed_dev = dev_ds.map(
        preprocess_example,
        remove_columns=dev_ds.column_names,
        desc="Preprocessing dev",
    )
    processed_dev = processed_dev.filter(lambda x: x.get("input_ids") is not None)
    print(f"Preprocessed: train={len(processed_train)} dev={len(processed_dev)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Train =====
    print("===== STAGE 4: Train =====")
    from transformers import EarlyStoppingCallback

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,  # -1 means use num_train_epochs
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",  # cosine > linear for longer runs
        seed=args.seed,
        report_to="none",
        # Evaluation + checkpointing for early stopping
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,  # save with eval so best checkpoint is always on disk
        save_total_limit=3,           # keep best + 2 most recent (disk hygiene)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        # PeftModel hides the base model's forward signature, so Trainer needs an
        # explicit hint that "labels" is the label key. Without this, eval_loss
        # is never computed and early stopping crashes with KeyError: 'eval_loss'.
        label_names=["labels"],
        # gradient_checkpointing INTENTIONALLY OFF — with PEFT it breaks grad flow
        # (UserWarning: None of the inputs have requires_grad=True).
    )

    trainer = Trainer(
        model=model,
        train_dataset=processed_train,
        eval_dataset=processed_dev,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(trainer_stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)

    # Pull best eval_loss (on dev set) — lower is better, this is our overfitting signal
    eval_history = [log for log in trainer.state.log_history if "eval_loss" in log]
    best_eval_loss = min((log["eval_loss"] for log in eval_history), default=None)
    final_eval = trainer.evaluate()
    final_eval_loss = float(final_eval.get("eval_loss", 0.0))

    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Best dev loss: {best_eval_loss:.4f}" if best_eval_loss is not None else "No eval history")
    print(f"Final dev loss: {final_eval_loss:.4f} (model reloaded at best checkpoint)")
    print(f"Peak VRAM: {peak_mem} GB\n")

    # ===== Generate samples =====
    print("===== STAGE 5: Generate =====")
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # One plain (text-only) and a few more for variety.
    plain_targets = []
    for i in range(min(5, len(test_ds))):
        plain_targets.append(normalize_text(test_ds[i]["text"]))
    # Deduplicate while preserving order.
    seen = set()
    plain_targets = [t for t in plain_targets if not (t in seen or seen.add(t))]

    generated: list[dict] = []

    for i, text in enumerate(plain_targets):
        try:
            plain_inputs = processor(
                f"[0]{text}",
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                audio_values = model.generate(
                    **plain_inputs,
                    max_new_tokens=125,
                    output_audio=True,
                )
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / f"plain_{i:02d}.wav"
            sf.write(str(path), audio, 24000)
            generated.append(
                {
                    "type": "plain",
                    "text": text,
                    "file": path.name,
                    "duration_sec": round(len(audio) / 24000, 2),
                }
            )
            print(f"  plain[{i}]: '{text}' -> {len(audio) / 24000:.1f}s")
        except Exception as exc:
            print(f"  plain[{i}] FAILED: {exc}")
            generated.append({"type": "plain", "text": text, "error": str(exc)})

    # Speaker-conditioned: use test[0] audio as reference, synthesize test[1] text.
    if len(test_ds) >= 2:
        try:
            ref_text = normalize_text(test_ds[0]["text"])
            ref_audio = test_ds[0]["audio"]["array"]
            gen_text = normalize_text(test_ds[1]["text"])
            conversation = [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": ref_text},
                        {"type": "audio", "path": ref_audio},
                    ],
                },
                {"role": "0", "content": [{"type": "text", "text": gen_text}]},
            ]
            conditioned_inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                common_kwargs={"return_tensors": "pt"},
            ).to(model.device)
            with torch.no_grad():
                audio_values = model.generate(
                    **conditioned_inputs,
                    max_new_tokens=125,
                    output_audio=True,
                )
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / "conditioned.wav"
            sf.write(str(path), audio, 24000)
            generated.append(
                {
                    "type": "speaker_conditioned",
                    "reference_text": ref_text,
                    "target_text": gen_text,
                    "file": path.name,
                    "duration_sec": round(len(audio) / 24000, 2),
                }
            )
            print(f"  conditioned: ref='{ref_text[:40]}...' -> gen='{gen_text[:40]}...' -> {len(audio) / 24000:.1f}s")
        except Exception as exc:
            print(f"  conditioned FAILED: {exc}")
            generated.append({"type": "speaker_conditioned", "error": str(exc)})

    # ===== Save adapter =====
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    results = {
        "experiment": "T1",
        "path": "vanilla_csm_peft_full",
        "seed": args.seed,
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "eval_steps": args.eval_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval_loss, 4) if best_eval_loss is not None else None,
        "final_eval_loss": round(final_eval_loss, 4),
        "global_step_at_stop": trainer.state.global_step,
        "epochs_completed": round(trainer.state.epoch, 2) if trainer.state.epoch is not None else None,
        "training_time_min": round(elapsed / 60, 1),
        "peak_vram_gb": peak_mem,
        "trainable_params_M": round(trainable / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 1),
        "n_train_samples": len(processed_train),
        "n_dev_samples": len(processed_dev),
        "n_test_samples": len(test_ds),
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
        "eval_history": eval_history,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved metrics to {metrics_path}")

    # ===== Push to Hub =====
    if args.push_to_hub:
        print("\n===== STAGE 6: Push to Hub =====")
        api = HfApi(token=token)
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        prefix = args.results_prefix.rstrip("/")

        api.upload_file(
            path_or_fileobj=metrics_path.read_bytes(),
            path_in_repo=f"{prefix}/metrics.json",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(adapter_dir),
            path_in_repo=f"{prefix}/adapter",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(generated_dir),
            path_in_repo=f"{prefix}/generated_audio",
            repo_id=args.results_repo,
            token=token,
        )
        print(f"Uploaded to https://huggingface.co/{args.results_repo} under {prefix}/")

    print("\nT1 canonical training finished.")


if __name__ == "__main__":
    main()
