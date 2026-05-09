#!/usr/bin/env python3
from __future__ import annotations

"""
Short diagnostic for Sesame CSM without the Unsloth patch path.

Use this only if the canonical CSM path fails and you need to answer:
"Is the blocker specifically in Unsloth's CSM integration, or in CSM/data itself?"

This script keeps:
- the same `apply_chat_template(...)` preprocessing shape
- PEFT LoRA adapters
- a 2-step dry-run
- one plain and one speaker-conditioned waveform generation check
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
    parser = argparse.ArgumentParser(description="Diagnostic Sesame CSM run without Unsloth patching")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus", help="Private HF dataset id")
    parser.add_argument("--output-dir", default="/tmp/csm_adja_vanilla_diag", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Split + training seed")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload metrics, adapter, and audio samples")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results", help="HF repo for outputs")
    return parser.parse_args()


def pip_install(packages: list[str], no_deps: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_diagnostic_env() -> None:
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
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    print("Installing diagnostic environment...")
    install_diagnostic_env()
    print("Environment ready.\n")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, CsmForConditionalGeneration, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this diagnostic")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("T1 diagnostic: vanilla CSM + PEFT")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"].cast_column("audio", Audio(sampling_rate=24000)).select(range(5))
    test_ds = split1["test"].cast_column("audio", Audio(sampling_rate=24000)).select(range(2))

    print("Loading base CSM directly from transformers...")
    model = CsmForConditionalGeneration.from_pretrained(
        "unsloth/csm-1b",
        torch_dtype=torch.float32,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained("unsloth/csm-1b")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    def preprocess_example(example: dict) -> dict:
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

        required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        missing = [key for key in required_keys if key not in model_inputs]
        if missing:
            raise KeyError(f"Missing keys from apply_chat_template: {missing}")

        return {key: model_inputs[key][0] for key in required_keys}

    processed_train = [preprocess_example(train_ds[i]) for i in range(len(train_ds))]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=processed_train,
        args=TrainingArguments(
            output_dir=str(output_dir / "trainer_output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=1,
            max_steps=2,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
            save_strategy="no",
            remove_unused_columns=False,
        ),
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(trainer_stats.metrics.get("train_loss", 0.0))

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    plain_text = normalize_text(test_ds[0]["text"])
    plain_inputs = processor(f"[0]{plain_text}", add_special_tokens=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        plain_audio_values = model.generate(**plain_inputs, max_new_tokens=125, output_audio=True)
    plain_audio = plain_audio_values[0].to(torch.float32).cpu().numpy()
    if plain_audio.size == 0:
        raise RuntimeError("Plain-text generation returned an empty waveform")
    plain_path = generated_dir / "plain.wav"
    sf.write(str(plain_path), plain_audio, 24000)

    conditioned_text = normalize_text(test_ds[1]["text"]) if len(test_ds) > 1 else plain_text
    conditioned_inputs = processor.apply_chat_template(
        [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": normalize_text(test_ds[0]["text"])},
                    {"type": "audio", "path": test_ds[0]["audio"]["array"]},
                ],
            },
            {"role": "0", "content": [{"type": "text", "text": conditioned_text}]},
        ],
        tokenize=True,
        return_dict=True,
        common_kwargs={"return_tensors": "pt"},
    ).to(model.device)
    with torch.no_grad():
        conditioned_audio_values = model.generate(
            **conditioned_inputs,
            max_new_tokens=125,
            output_audio=True,
        )
    conditioned_audio = conditioned_audio_values[0].to(torch.float32).cpu().numpy()
    if conditioned_audio.size == 0:
        raise RuntimeError("Speaker-conditioned generation returned an empty waveform")
    conditioned_path = generated_dir / "conditioned.wav"
    sf.write(str(conditioned_path), conditioned_audio, 24000)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    results = {
        "experiment": "T1-diagnostic",
        "path": "vanilla_csm_peft",
        "seed": args.seed,
        "dataset": args.dataset,
        "train_loss": round(train_loss, 4),
        "training_time_min": round(elapsed / 60, 1),
        "generated": {
            "plain": {"text": plain_text, "file": plain_path.name},
            "conditioned": {"text": conditioned_text, "file": conditioned_path.name},
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved diagnostic metrics to {metrics_path}")

    if args.push_to_hub:
        api = HfApi(token=token)
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=metrics_path.read_bytes(),
            path_in_repo="T1_diagnostic/metrics.json",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(adapter_dir),
            path_in_repo="T1_diagnostic/adapter",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(generated_dir),
            path_in_repo="T1_diagnostic/generated_audio",
            repo_id=args.results_repo,
            token=token,
        )
        print(f"Uploaded diagnostic outputs to https://huggingface.co/{args.results_repo}")

    print("Vanilla CSM diagnostic finished successfully.")


if __name__ == "__main__":
    main()
