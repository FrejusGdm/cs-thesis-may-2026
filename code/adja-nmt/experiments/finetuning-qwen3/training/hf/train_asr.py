#!/usr/bin/env python3
"""
Qwen3-ASR Fine-tuning Script (HuggingFace-compatible)
=====================================================
This script fine-tunes Qwen3-ASR on your custom dataset using the
HuggingFace Trainer API. It wraps the official qwen3_asr_sft.py with:
- YAML config support
- Better logging
- Validation evaluation
- HF Hub integration

Based on the official qwen3_asr_sft.py from:
    https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning

Usage:
    # Using config:
    python train_asr.py --config ../../configs/asr_config.yaml

    # Direct:
    python train_asr.py \
        --model_path Qwen/Qwen3-ASR-1.7B \
        --train_file splits/train.jsonl \
        --eval_file splits/val.jsonl \
        --output_dir ./asr_output

    # Multi-GPU:
    torchrun --nproc_per_node=2 train_asr.py \
        --config ../../configs/asr_config.yaml
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
import yaml
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Model patching (required for HF Trainer compatibility)
# ---------------------------------------------------------------------------

def patch_outer_forward(model):
    """Patch the outer model's forward to delegate to the thinker.

    WHY THIS IS NEEDED:
    Qwen3-ASR wraps its core model inside a 'thinker' module.
    The HF Trainer calls model.forward() and expects standard
    (input_ids, attention_mask, labels) signature. This patch
    makes the outer model delegate to thinker.forward() which
    has the correct signature.
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in output_dir."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000):
    """Load audio at target sample rate, mono."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------

def build_prefix_messages(prompt: str, audio_array):
    """Build the chat template messages for ASR."""
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    """Create preprocessing function for the dataset.

    This function applies the chat template to create the input prefix
    (system prompt + audio placeholder), which the model must attend to
    before generating the transcription.
    """
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASR:
    """Custom data collator that handles audio loading and label masking.

    KEY CONCEPT (label masking):
    We only want the model to learn to predict the TRANSCRIPTION tokens,
    not the prefix tokens (system prompt, audio features). So we set
    labels = -100 for all prefix positions. PyTorch's CrossEntropyLoss
    ignores -100 labels automatically.
    """
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Mask prefix tokens in labels (set to -100 = ignore)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class CastFloatInputsTrainer(Trainer):
    """Trainer that casts float inputs to model dtype.

    Without this, float32 audio features would be fed to a bf16 model,
    causing dtype mismatches and potential precision issues.
    """
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


# ---------------------------------------------------------------------------
# Callback: make checkpoints self-contained
# ---------------------------------------------------------------------------

def copy_hf_files(src_dir: str, dst_dir: str):
    """Copy tokenizer/config files to checkpoint for standalone inference."""
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class InferableCheckpointCallback(TrainerCallback):
    """Copies config files to each checkpoint so they can be loaded directly."""
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            copy_hf_files(self.base_model_path, ckpt_dir)
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Fine-tuning")

    p.add_argument("--config", type=str, help="YAML config file")

    # Paths
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    # Audio
    p.add_argument("--sr", type=int, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_acc", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--log_steps", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--lr_scheduler_type", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--report_to", type=str, default=None)

    # Save
    p.add_argument("--save_strategy", type=str, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=None)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    # Load and merge config
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            raw_config = yaml.safe_load(f)
        # Flatten
        for section in raw_config.values():
            if isinstance(section, dict):
                config.update(section)

    # Apply CLI overrides
    overrides = {
        "model_path": args.model_path,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "output_dir": args.output_dir,
        "sample_rate": args.sr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_acc,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "log_steps": args.log_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "report_to": args.report_to,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
    }
    for k, v in overrides.items():
        if v is not None:
            config[k] = v

    # Defaults
    config.setdefault("model_path", "Qwen/Qwen3-ASR-0.6B")
    config.setdefault("output_dir", "./asr_output")
    config.setdefault("sample_rate", 16000)
    config.setdefault("batch_size", 32)
    config.setdefault("gradient_accumulation_steps", 4)
    config.setdefault("learning_rate", 2e-5)
    config.setdefault("num_epochs", 1)
    config.setdefault("log_steps", 10)
    config.setdefault("warmup_ratio", 0.02)
    config.setdefault("lr_scheduler_type", "linear")
    config.setdefault("save_strategy", "steps")
    config.setdefault("save_steps", 200)
    config.setdefault("save_total_limit", 5)
    config.setdefault("num_workers", 4)
    config.setdefault("pin_memory", True)
    config.setdefault("persistent_workers", True)
    config.setdefault("prefetch_factor", 2)
    config.setdefault("max_steps", -1)
    config.setdefault("seed", 42)
    config.setdefault("report_to", "tensorboard")

    if not config.get("train_file"):
        print("ERROR: --train_file is required")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        config["model_path"],
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    data_files = {"train": config["train_file"]}
    if config.get("eval_file"):
        data_files["validation"] = config["eval_file"]

    raw_ds = load_dataset("json", data_files=data_files)
    ds = raw_ds.map(make_preprocess_fn(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASR(
        processor=processor,
        sampling_rate=config["sample_rate"],
    )

    # -----------------------------------------------------------------------
    # Setup Trainer
    # -----------------------------------------------------------------------
    nw = config["num_workers"]
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        logging_steps=config["log_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        dataloader_num_workers=nw,
        dataloader_pin_memory=config["pin_memory"],
        dataloader_persistent_workers=config["persistent_workers"],
        dataloader_prefetch_factor=config["prefetch_factor"] if nw > 0 else None,
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        max_steps=config["max_steps"],
        save_safetensors=True,
        evaluation_strategy="steps" if config.get("eval_file") else "no",
        eval_steps=config["save_steps"] if config.get("eval_file") else None,
        do_eval=bool(config.get("eval_file")),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=config["seed"],
        data_seed=config["seed"],
        report_to=config["report_to"],
        logging_dir=os.path.join(config["output_dir"], "tensorboard"),
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[InferableCheckpointCallback(base_model_path=config["model_path"])],
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        print(f"[resume] Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
