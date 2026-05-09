#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0", "numpy", "sentencepiece", "protobuf", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "bitsandbytes", "accelerate", "xformers", "peft", "trl==0.22.2", "triton", "transformers==4.56.2", "omegaconf", "einx", "einops", "torchcodec", "setuptools", "wheel", "pillow", "psutil", "msgspec", "tyro", "cut_cross_entropy"]
# ///
from __future__ import annotations
"""
T3-ewe Stage 1: Fine-tune Spark TTS (0.5B) on WaxalNLP Ewe TTS data.

Ewe is Adja's closest Gbe-family relative. Fine-tuning Spark on Ewe first
gives the model Gbe-family prosody and phonology before Adja adaptation.

Stage 2 (T3_spark_ewe_adja_stage2.py) loads the saved checkpoint from this run.

Data: google/WaxalNLP, config=ewe_tts
  train: 1215 rows  val: 152 rows  test: 152 rows
  Audio: 48kHz → resampled to Spark's target SR inside formatting_audio_func
  Text: Ewe Latin orthography with diacritics (same family as Adja)

References:
  - Spark TTS: https://github.com/SparkAudio/Spark-TTS
  - WaxalNLP: https://huggingface.co/datasets/google/WaxalNLP
  - Gbe language family cross-lingual transfer hypothesis:
      experiments/asr-tts-getting-right-2026-04-21.md, Track 1B
"""
import argparse, json, os, re, subprocess, sys, time, unicodedata
sys.stdout.reconfigure(line_buffering=True)

# --- triton JIT pre-flight (torch 2.6 ships triton 3.2+, which compiles a CUDA
# driver shim via `gcc ... -lcuda` the first time triton is imported; that
# happens transitively through bitsandbytes -> transformers.integrations during
# `from unsloth import FastModel`). Two env gaps in HF Jobs' uv-script container
# break this:
#
#   (1) Python dev headers are missing. The shim's C source does `#include <Python.h>`;
#       without `libpython3.11-dev` installed gcc fails with "Python.h: No such file".
#   (2) libcuda.so is unversioned. Only `libcuda.so.1` is mounted (from the host
#       NVIDIA driver at /usr/lib64-nvidia on HF Jobs L40S), and gcc's `-lcuda`
#       search can't resolve that.
#
# Fix (1) by apt-installing python3-dev below in the deps block. Fix (2) here by
# symlinking and adding the driver dir to LIBRARY_PATH so the linker can find it.
for _libdir in ("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"):
    _src = f"{_libdir}/libcuda.so.1"
    _dst = f"{_libdir}/libcuda.so"
    if os.path.exists(_src):
        if not os.path.exists(_dst):
            try:
                os.symlink(_src, _dst)
                print(f"[cuda-shim] symlinked {_dst} -> {_src}")
            except OSError as _e:
                print(f"[cuda-shim] symlink skip {_dst} ({_e})")
        os.environ["LIBRARY_PATH"] = _libdir + os.pathsep + os.environ.get("LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _libdir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        print(f"[cuda-shim] added {_libdir} to LIBRARY_PATH/LD_LIBRARY_PATH")
        break
else:
    print("[cuda-shim] WARNING: no libcuda.so.1 found in standard paths")

_parser = argparse.ArgumentParser(description="T3-ewe Stage 1: Spark TTS on Ewe")
_parser.add_argument("--push-to-hub", action="store_true")
_parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
_parser.add_argument("--results-prefix", default="T3_spark_ewe_stage1")
_parser.add_argument("--max-steps", type=int, default=-1)
_parser.add_argument("--num-epochs", type=int, default=20)
_parser.add_argument("--checkpoint-repo", default="JosueG/adja-tts-checkpoints",
                     help="Separate repo to push the full merged model for Stage 2")
_args = _parser.parse_args()

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
print("===== Installing dependencies =====")
os.system("apt-get update -q && apt-get install -y -q git python3-dev libpython3.11-dev >/dev/null 2>&1")

def uv_pip_install_no_deps(packages: list[str]) -> None:
    print(f"$ uv pip install -q --no-deps {' '.join(packages)}")
    subprocess.check_call(["uv", "pip", "install", "-q", "--no-deps", *packages])

# Unsloth currently pulls in torchao, which can break against the PyTorch builds
# available on HF Jobs. Install it without deps and rely on the explicit uv
# dependency list above for the runtime stack.
uv_pip_install_no_deps(["unsloth_zoo", "unsloth"])
uv_pip_install_no_deps(["trl==0.22.2"])
os.system("git clone --depth 1 https://github.com/SparkAudio/Spark-TTS /tmp/Spark-TTS")
print("Dependencies installed.\n")

import numpy as np
import torch
print(f"torch={torch.__version__}, CUDA={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

from unsloth import FastModel
from huggingface_hub import snapshot_download

snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="/tmp/Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name="/tmp/Spark-TTS-0.5B/LLM",
    max_seq_length=2048,
    dtype=torch.bfloat16,      # bf16 halves model + activation memory vs fp32; L40S supports it natively
    full_finetuning=False,     # we're applying LoRA below — Unsloth was keeping grad state on all 500M base params otherwise
    load_in_4bit=False,
)

model = FastModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("===== Loading Ewe TTS dataset (google/WaxalNLP, ewe_tts) =====")
from datasets import load_dataset, Audio
import torchaudio.transforms as T

sys.path.insert(0, "/tmp/Spark-TTS")
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

audio_tokenizer = BiCodecTokenizer("/tmp/Spark-TTS-0.5B", "cuda")

# WaxalNLP ewe_tts — use existing splits
ewe_train = load_dataset("google/WaxalNLP", name="ewe_tts", split="train", token=token)
ewe_val   = load_dataset("google/WaxalNLP", name="ewe_tts", split="validation", token=token)

# Detect text column name (WaxalNLP may use "text" or "sentence")
text_col = "text" if "text" in ewe_train.column_names else "sentence"
print(f"Dataset columns: {ewe_train.column_names}")
print(f"Using text column: '{text_col}'")
print(f"Ewe train: {len(ewe_train)} | val: {len(ewe_val)}")


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def extract_wav2vec2_features(wavs):
    wav_np = wavs.squeeze(0).cpu().numpy()
    processed = audio_tokenizer.processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = processed.input_values.to(audio_tokenizer.feature_extractor.device)
    model_output = audio_tokenizer.feature_extractor(input_values)
    feats_mix = (model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]) / 3
    return feats_mix


def formatting_audio_func(example):
    text = normalize_text(example[text_col])
    audio_array = np.asarray(example["audio"]["array"], dtype=np.float32)
    sampling_rate = example["audio"]["sampling_rate"]

    target_sr = audio_tokenizer.config["sample_rate"]
    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio_array = resampler(torch.from_numpy(audio_array).float()).numpy()

    if audio_tokenizer.config["volume_normalize"]:
        audio_array = audio_volume_normalize(audio_array)

    ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
    ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)
    feat = extract_wav2vec2_features(audio_tensor)

    batch = {"wav": audio_tensor, "ref_wav": ref_wav_tensor, "feat": feat.to(audio_tokenizer.device)}
    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()])
    semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()])

    inputs = "".join([
        "<|task_tts|>", "<|start_content|>", text, "<|end_content|>",
        "<|start_global_token|>", global_tokens, "<|end_global_token|>",
        "<|start_semantic_token|>", semantic_tokens, "<|end_semantic_token|>", "<|im_end|>",
    ])
    return {"text": inputs}


print("Tokenizing Ewe audio (train)...")
train_dataset = ewe_train.map(formatting_audio_func, remove_columns=["audio"], desc="Tokenizing Ewe train")
print("Tokenizing Ewe audio (val)...")
eval_dataset = ewe_val.map(formatting_audio_func, remove_columns=["audio"], desc="Tokenizing Ewe val")
print(f"Tokenized: train={len(train_dataset)} eval={len(eval_dataset)}")

audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()

print("===== Training on Ewe =====")
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

if _args.max_steps > 0:
    _sft_kwargs = dict(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=10, max_steps=_args.max_steps, learning_rate=2e-4,
        fp16=False, bf16=True, logging_steps=10, optim="adamw_8bit",
        weight_decay=0.001, lr_scheduler_type="linear", seed=42,
        output_dir="/tmp/outputs", report_to="none",
    )
    _callbacks, _eval_ds = None, None
else:
    _sft_kwargs = dict(
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, warmup_steps=20,
        num_train_epochs=_args.num_epochs, max_steps=-1, learning_rate=2e-4,
        fp16=False, bf16=True, logging_steps=10, optim="adamw_8bit",
        weight_decay=0.001, lr_scheduler_type="cosine", seed=42,
        output_dir="/tmp/outputs", report_to="none",
        eval_strategy="steps", eval_steps=50, save_strategy="steps",
        save_steps=50, save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
    )
    _callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    _eval_ds = eval_dataset

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=_eval_ds,
    dataset_text_field="text", max_seq_length=2048, packing=False,
    callbacks=_callbacks, args=SFTConfig(**_sft_kwargs),
)

t0 = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - t0
peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
print(f"Training done in {elapsed/60:.1f} min | loss={trainer_stats.metrics['train_loss']:.4f} | VRAM={peak_mem}GB\n")

# Save merged model for Stage 2
print("Saving merged model for Stage 2...")
FastModel.for_inference(model)
model.save_pretrained("/tmp/spark_ewe_merged")
tokenizer.save_pretrained("/tmp/spark_ewe_merged")

results = {
    "experiment": "T3_spark_ewe_stage1",
    "model": "unsloth/Spark-TTS-0.5B",
    "dataset": "google/WaxalNLP ewe_tts",
    "train_samples": len(train_dataset),
    "eval_samples": len(eval_dataset),
    "train_loss": round(trainer_stats.metrics["train_loss"], 4),
    "training_time_min": round(elapsed / 60, 1),
    "peak_vram_gb": peak_mem,
    "gpu": torch.cuda.get_device_name(0),
    "stage2_checkpoint": f"{_args.checkpoint_repo}/T3_spark_ewe_stage1",
}

if _args.push_to_hub:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    prefix = _args.results_prefix

    # Push metrics
    api.create_repo(_args.results_repo, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
        path_in_repo=f"{prefix}/metrics.json",
        repo_id=_args.results_repo, token=token,
    )

    # Push merged model to checkpoint repo for Stage 2 to load
    api.create_repo(_args.checkpoint_repo, private=True, exist_ok=True)
    api.upload_folder(
        folder_path="/tmp/spark_ewe_merged",
        path_in_repo="T3_spark_ewe_stage1",
        repo_id=_args.checkpoint_repo, token=token,
    )
    print(f"Checkpoint pushed to {_args.checkpoint_repo}/T3_spark_ewe_stage1")
    print(f"Results: https://huggingface.co/{_args.results_repo}/tree/main/{prefix}")
else:
    print("Skipping Hub upload. Artifacts at /tmp/spark_ewe_merged/")
