#!/usr/bin/env python3
from __future__ import annotations
"""
T3-ewe Stage 2: Fine-tune Ewe-adapted Spark TTS on Adja.

Loads the Stage 1 checkpoint (Spark fine-tuned on Ewe) and continues training
on Adja data. The hypothesis is that the Ewe stage gives the model Gbe-family
acoustic priors that make Adja adaptation easier.

Prerequisites:
  - T3_spark_ewe_stage1.py must have completed and pushed its checkpoint.
  - Set --stage1-checkpoint to the Hub path of the Stage 1 model.

References:
  - Cross-lingual TTS transfer: experiments/asr-tts-getting-right-2026-04-21.md, Track 1B
"""
import argparse, json, os, sys, time, re, unicodedata
sys.stdout.reconfigure(line_buffering=True)

_parser = argparse.ArgumentParser(description="T3-ewe Stage 2: Spark Ewe→Adja")
_parser.add_argument("--push-to-hub", action="store_true")
_parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
_parser.add_argument("--results-prefix", default="T3_spark_ewe_adja_stage2")
_parser.add_argument("--stage1-checkpoint",
                     default="JosueG/adja-tts-checkpoints/T3_spark_ewe_stage1",
                     help="Hub path (repo_id/subfolder) to Stage 1 merged model")
_parser.add_argument("--adja-dataset", default="JosueG/adja-tts-orpheus")
_parser.add_argument("--max-steps", type=int, default=-1)
_parser.add_argument("--num-epochs", type=int, default=20)
_args = _parser.parse_args()

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
print("===== Installing dependencies =====")
os.system("apt-get update -q && apt-get install -y -q git >/dev/null 2>&1")
os.system("pip install -q sentencepiece protobuf 'datasets>=3.4.1,<4.0.0' 'huggingface_hub>=0.34.0' hf_transfer soundfile librosa")
os.system("pip install -q --no-deps unsloth_zoo bitsandbytes accelerate xformers peft trl triton unsloth")
os.system("pip install -q transformers==4.56.2")
os.system("pip install -q --no-deps trl==0.22.2")
os.system("pip install -q omegaconf einx einops torchcodec")
os.system("git clone --depth 1 https://github.com/SparkAudio/Spark-TTS /tmp/Spark-TTS")
print("Dependencies installed.\n")

import numpy as np
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

from unsloth import FastModel
from huggingface_hub import snapshot_download, HfApi

# Download Stage 1 checkpoint (Ewe-adapted Spark)
# Stage 1 pushes to checkpoint_repo/T3_spark_ewe_stage1, which is a flat folder
stage1_repo, stage1_subdir = _args.stage1_checkpoint.rsplit("/", 1)
print(f"Downloading Stage 1 checkpoint from {_args.stage1_checkpoint}...")
snapshot_download(stage1_repo, local_dir="/tmp/stage1_checkpoint",
                  allow_patterns=f"{stage1_subdir}/*", token=token)
stage1_local = f"/tmp/stage1_checkpoint/{stage1_subdir}"

# Also download the Spark-TTS-0.5B for BiCodecTokenizer assets
snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="/tmp/Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name=stage1_local,
    max_seq_length=2048,
    dtype=torch.float32,
    full_finetuning=True,
    load_in_4bit=False,
)

model = FastModel.get_peft_model(
    model, r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

print("===== Loading Adja dataset =====")
from datasets import load_dataset, Audio
import torchaudio.transforms as T

sys.path.insert(0, "/tmp/Spark-TTS")
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

audio_tokenizer = BiCodecTokenizer("/tmp/Spark-TTS-0.5B", "cuda")

raw_ds = load_dataset(_args.adja_dataset, token=token, split="train")
print(f"Loaded {len(raw_ds)} Adja samples")


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
    text = normalize_text(example["text"])
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
    return {"text": "".join(["<|task_tts|>", "<|start_content|>", text, "<|end_content|>",
                              "<|start_global_token|>", global_tokens, "<|end_global_token|>",
                              "<|start_semantic_token|>", semantic_tokens, "<|end_semantic_token|>", "<|im_end|>"])}


print("Tokenizing Adja audio...")
dataset = raw_ds.map(formatting_audio_func, remove_columns=["audio"], desc="Tokenizing Adja")
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]
print(f"train={len(train_dataset)} eval={len(eval_dataset)}")

audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()

print("===== Training on Adja =====")
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

if _args.max_steps > 0:
    _sft_kwargs = dict(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=10, max_steps=_args.max_steps, learning_rate=1e-4,
        fp16=False, bf16=False, logging_steps=10, optim="adamw_8bit",
        weight_decay=0.001, lr_scheduler_type="linear", seed=42,
        output_dir="/tmp/outputs", report_to="none",
    )
    _callbacks, _eval_ds = None, None
else:
    _sft_kwargs = dict(
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, warmup_steps=10,
        num_train_epochs=_args.num_epochs, max_steps=-1, learning_rate=1e-4,
        fp16=False, bf16=False, logging_steps=10, optim="adamw_8bit",
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
print(f"Done in {elapsed/60:.1f} min | loss={trainer_stats.metrics['train_loss']:.4f} | VRAM={peak_mem}GB\n")

# Generate Adja samples
print("===== Generating Adja speech =====")
import soundfile as sf
FastModel.for_inference(model)
os.makedirs("/tmp/generated", exist_ok=True)
audio_tokenizer.device = torch.device("cuda")
audio_tokenizer.model.to("cuda")

test_sentences = ["Tom trɔ yi kpanŋkɔ wezexu", "Ele lɔ awu ehoci lɔ sa",
                  "Ɛ yi gbɛ̀", "Mì ɖo alɔ ji", "Nye ŋkɔ nyé Tom"]
generated = []
sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
for i, text in enumerate(test_sentences):
    text = unicodedata.normalize("NFC", text)
    try:
        prompt = "".join(["<|task_tts|>", "<|start_content|>", text, "<|end_content|>", "<|start_global_token|>"])
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        gen_ids = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True,
                                 temperature=0.8, top_k=50, top_p=1.0,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        gen_trimmed = gen_ids[:, model_inputs.input_ids.shape[1]:]
        gen_text = tokenizer.batch_decode(gen_trimmed, skip_special_tokens=False)[0]
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", gen_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", gen_text)
        if not semantic_matches:
            generated.append({"text": text, "error": "no semantic tokens"})
            continue
        pred_semantic = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
        pred_global = (torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0)
                       if global_matches else torch.zeros((1, 1, 1), dtype=torch.long))
        wav_np = audio_tokenizer.detokenize(pred_global.to("cuda").squeeze(0), pred_semantic.to("cuda"))
        wav_path = f"/tmp/generated/adja_{i:02d}.wav"
        sf.write(wav_path, wav_np, sample_rate)
        dur = len(wav_np) / sample_rate
        generated.append({"text": text, "file": f"adja_{i:02d}.wav", "duration_sec": round(dur, 2)})
        print(f"  [{i}] '{text}' -> {dur:.1f}s")
    except Exception as e:
        print(f"  [{i}] FAILED: {e}")
        generated.append({"text": text, "error": str(e)})

model.save_pretrained("/tmp/spark_ewe_adja_lora")
tokenizer.save_pretrained("/tmp/spark_ewe_adja_lora")

results = {
    "experiment": "T3_spark_ewe_adja_stage2",
    "stage1_checkpoint": _args.stage1_checkpoint,
    "adja_dataset": _args.adja_dataset,
    "train_loss": round(trainer_stats.metrics["train_loss"], 4),
    "training_time_min": round(elapsed / 60, 1),
    "peak_vram_gb": peak_mem,
    "gpu": torch.cuda.get_device_name(0),
    "generated": generated,
}

if _args.push_to_hub:
    api = HfApi(token=token)
    prefix = _args.results_prefix
    api.create_repo(_args.results_repo, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
        path_in_repo=f"{prefix}/metrics.json", repo_id=_args.results_repo, token=token,
    )
    api.upload_folder(folder_path="/tmp/spark_ewe_adja_lora",
                      path_in_repo=f"{prefix}/adapter", repo_id=_args.results_repo, token=token)
    api.upload_folder(folder_path="/tmp/generated",
                      path_in_repo=f"{prefix}/generated_audio", repo_id=_args.results_repo, token=token)
    print(f"Done: https://huggingface.co/{_args.results_repo}/tree/main/{prefix}")
