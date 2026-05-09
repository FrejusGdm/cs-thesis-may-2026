"""
T3: Fine-tune Spark TTS (0.5B) on Adja TTS data with Unsloth.
Runs inside pytorch/pytorch Docker image on HF Jobs.

Adapted from Unsloth's Spark TTS notebook:
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Spark_TTS_(0_5B).ipynb

Key differences from CSM:
    - Uses BiCodecTokenizer (semantic + global tokens)
    - Full float32 training (no bf16/fp16)
    - Requires cloning SparkAudio/Spark-TTS repo for tokenizer code
    - 0.5B params — smallest TTS model, interesting for deployment

References:
    - Spark TTS: https://github.com/SparkAudio/Spark-TTS
    - Unsloth: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
"""
from __future__ import annotations
import argparse, json, os, sys, time, re, unicodedata
sys.stdout.reconfigure(line_buffering=True)

# ---- CLI args (so --push-to-hub and --results-prefix are not silently ignored) ----
_parser = argparse.ArgumentParser(description="T3 Spark TTS fine-tune on Adja")
_parser.add_argument("--push-to-hub", action="store_true",
                     help="Upload adapter + audio to the results repo")
_parser.add_argument("--results-repo", default="JosueG/adja-tts-results",
                     help="HF repo to push results to")
_parser.add_argument("--results-prefix", default="T3",
                     help="Subfolder inside the results repo (default: T3)")
_parser.add_argument("--max-steps", type=int, default=-1,
                     help="If > 0, use max_steps (disables eval / early stopping). If -1, "
                          "use num_train_epochs=20 with eval + early stopping.")
_parser.add_argument("--num-epochs", type=int, default=20,
                     help="Max epochs when --max-steps is not set")
_args = _parser.parse_args()

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

# ===== Install — follow the notebook: "pip install unsloth" for non-Colab =====
# Disable auto-update so unsloth_zoo stays consistent
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
print("===== Installing dependencies =====")
os.system("apt-get update -q && apt-get install -y -q git >/dev/null 2>&1")
os.system("pip install -q sentencepiece protobuf 'datasets>=3.4.1,<4.0.0' 'huggingface_hub>=0.34.0' hf_transfer soundfile librosa")
# Install Unsloth with --no-deps so it does NOT upgrade the container's torch/torchaudio.
# Container ships torch==2.6.0+cu124 with a matching torchaudio — keep that combo.
os.system("pip install -q --no-deps unsloth_zoo bitsandbytes accelerate xformers peft trl triton unsloth")
os.system("pip install -q transformers==4.56.2")
os.system("pip install -q --no-deps trl==0.22.2")
os.system("pip install -q omegaconf einx einops torchcodec")
# Clone Spark-TTS repo (needed for BiCodecTokenizer)
os.system("git clone --depth 1 https://github.com/SparkAudio/Spark-TTS /tmp/Spark-TTS")
print("Dependencies installed.\n")

import numpy as np
import torch
print(f"torch={torch.__version__}, CUDA={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ===== Load Model =====
print("===== Loading Spark TTS (0.5B) with Unsloth =====")
from unsloth import FastModel
from huggingface_hub import snapshot_download

snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="/tmp/Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name="/tmp/Spark-TTS-0.5B/LLM",
    max_seq_length=2048,
    dtype=torch.float32,  # Spark requires float32
    full_finetuning=True,
    load_in_4bit=False,
)

model = FastModel.get_peft_model(
    model,
    r=128,  # Higher rank for small model (matches notebook)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)\n")

# ===== Load & Tokenize Adja Dataset =====
print("===== Loading Adja Dataset =====")
from datasets import load_dataset, Audio
import torchaudio.transforms as T

sys.path.insert(0, "/tmp/Spark-TTS")
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

audio_tokenizer = BiCodecTokenizer("/tmp/Spark-TTS-0.5B", "cuda")

raw_ds = load_dataset("JosueG/adja-tts-orpheus", token=token, split="train")
print(f"Loaded {len(raw_ds)} samples")

def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())

def extract_wav2vec2_features(wavs):
    if wavs.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {wavs.shape}")
    wav_np = wavs.squeeze(0).cpu().numpy()
    processed = audio_tokenizer.processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = processed.input_values.to(audio_tokenizer.feature_extractor.device)
    model_output = audio_tokenizer.feature_extractor(input_values)
    feats_mix = (model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]) / 3
    return feats_mix

def formatting_audio_func(example):
    text = normalize_text(example["text"])
    # datasets >=3.4 can return audio["array"] as a Python list; torchaudio needs np.ndarray.
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

print("Tokenizing audio (this takes a while)...")
dataset = raw_ds.map(formatting_audio_func, remove_columns=["audio"], desc="Tokenizing Adja audio")
print(f"Tokenized: {len(dataset)} samples")

# Split 90/10 for train/dev so we can early-stop on eval loss.
print("Splitting 90/10 train/dev seed=42")
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  train={len(train_dataset)} dev={len(eval_dataset)}")

# Free GPU memory from audio tokenizer
print("Moving audio tokenizer to CPU...")
audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()
print()

# ===== Train =====
print("===== Training =====")
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

# Two training modes controlled by --max-steps:
#   --max-steps > 0  → original short-run shape (no eval, no early stop, linear schedule)
#                      Mirrors what produced the intelligible 120-step checkpoint.
#   --max-steps -1   → long run, 20 epochs, eval every 50 steps, early-stop patience 5,
#                      cosine schedule. Matches the T1 canonical treatment.
if _args.max_steps > 0:
    print(f"Short-run mode: max_steps={_args.max_steps}, no eval, no early stopping")
    _sft_kwargs = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=_args.max_steps,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="/tmp/outputs",
        report_to="none",
    )
    _callbacks = None
    _eval_ds = None
else:
    print(f"Long-run mode: num_train_epochs={_args.num_epochs}, eval every 50 steps, patience=5")
    _sft_kwargs = dict(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=_args.num_epochs,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="/tmp/outputs",
        report_to="none",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    _callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    _eval_ds = eval_dataset

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=_eval_ds,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    callbacks=_callbacks,
    args=SFTConfig(**_sft_kwargs),
)

start_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
t0 = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - t0
peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)

print(f"\nTraining done in {elapsed/60:.1f} min")
print(f"Loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM: {peak_mem} GB\n")

# ===== Generate Audio =====
print("===== Generating Adja Speech =====")
import soundfile as sf

FastModel.for_inference(model)
os.makedirs("/tmp/generated", exist_ok=True)

# Move audio tokenizer back to GPU for decoding
audio_tokenizer.device = torch.device("cuda")
audio_tokenizer.model.to("cuda")

test_sentences = [
    "Tom trɔ yi kpanŋkɔ wezexu",
    "Ele lɔ awu ehoci lɔ sa",
    "Ɛ yi gbɛ̀",
    "Mì ɖo alɔ ji",
    "Nye ŋkɔ nyé Tom",
]

generated = []
sample_rate = audio_tokenizer.config.get("sample_rate", 16000)

for i, text in enumerate(test_sentences):
    text = unicodedata.normalize("NFC", text)
    try:
        prompt = "".join(["<|task_tts|>", "<|start_content|>", text, "<|end_content|>", "<|start_global_token|>"])
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_trimmed = gen_ids[:, model_inputs.input_ids.shape[1]:]
        gen_text = tokenizer.batch_decode(gen_trimmed, skip_special_tokens=False)[0]

        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", gen_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", gen_text)

        if not semantic_matches:
            print(f"  [{i}] No semantic tokens for '{text}'")
            generated.append({"text": text, "error": "no semantic tokens"})
            continue

        pred_semantic = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
        pred_global = torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0) if global_matches else torch.zeros((1, 1, 1), dtype=torch.long)

        wav_np = audio_tokenizer.detokenize(pred_global.to("cuda").squeeze(0), pred_semantic.to("cuda"))

        wav_path = f"/tmp/generated/spark_{i:02d}.wav"
        sf.write(wav_path, wav_np, sample_rate)
        dur = len(wav_np) / sample_rate
        generated.append({"text": text, "file": f"spark_{i:02d}.wav", "duration_sec": round(dur, 2), "semantic_tokens": len(semantic_matches), "global_tokens": len(global_matches)})
        print(f"  [{i}] '{text}' -> {dur:.1f}s ({len(semantic_matches)} sem, {len(global_matches)} glob tokens)")

    except Exception as e:
        print(f"  [{i}] FAILED: {e}")
        generated.append({"text": text, "error": str(e)})

# ===== Save & Push =====
print("\n===== Saving & Pushing to Hub =====")
model.save_pretrained("/tmp/spark_adja_lora")
tokenizer.save_pretrained("/tmp/spark_adja_lora")

from huggingface_hub import HfApi
api = HfApi(token=token)
RESULTS_REPO = "JosueG/adja-tts-results"
try:
    api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except: pass

results = {
    "experiment": "T3",
    "model": "unsloth/Spark-TTS-0.5B",
    "lora_r": 128,
    "max_steps": 120,
    "train_loss": round(trainer_stats.metrics["train_loss"], 4),
    "training_time_min": round(elapsed / 60, 1),
    "peak_vram_gb": peak_mem,
    "trainable_params": trainable,
    "total_params": total_params,
    "n_samples": len(dataset),
    "gpu": torch.cuda.get_device_name(0),
    "generated": generated,
}

if _args.push_to_hub:
    prefix = _args.results_prefix.rstrip("/")
    RESULTS_REPO = _args.results_repo
    print(f"\nUploading to {RESULTS_REPO} under {prefix}/")

    api.upload_file(path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
                    path_in_repo=f"{prefix}/metrics.json", repo_id=RESULTS_REPO, token=token)

    try:
        api.upload_folder(folder_path="/tmp/spark_adja_lora", path_in_repo=f"{prefix}/adapter", repo_id=RESULTS_REPO, token=token)
        print("Adapter pushed.")
    except Exception as e:
        print(f"Adapter push: {e}")

    try:
        api.upload_folder(folder_path="/tmp/generated", path_in_repo=f"{prefix}/generated_audio", repo_id=RESULTS_REPO, token=token)
        print("Audio pushed.")
    except Exception as e:
        print(f"Audio push: {e}")

    print(f"\nDone! Results: https://huggingface.co/{RESULTS_REPO}/tree/main/{prefix}")
else:
    print("\nSkipping Hub upload (--push-to-hub not set). Artifacts in /tmp/.")
