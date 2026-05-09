# /// script
# dependencies = ["transformers", "torch==2.5.1", "torchaudio==2.5.1", "datasets", "soundfile", "librosa", "numpy", "huggingface-hub"]
# ///
"""
C1: Whisper zero-shot ASR inference on Adja.
Self-contained for HuggingFace Jobs.
What does Whisper know about Adja with no training?
"""
from __future__ import annotations
import json, os, sys, time, unicodedata, random
import numpy as np
import torch
import soundfile as sf
sys.stdout.reconfigure(line_buffering=True)

# ---- Config ----
DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
MODEL_SIZES = ["tiny", "small", "large-v3"]
SEED = 42
TEST_RATIO = 0.1
DEV_RATIO = 0.1

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

# ---- Load data ----
print("Loading dataset...")
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
print(f"Loaded {len(ds)} samples")

# Create test split (same as data_prep.py: seed=42, 10% test)
split1 = ds.train_test_split(test_size=TEST_RATIO, seed=SEED)
split2 = split1["train"].train_test_split(test_size=DEV_RATIO / (1 - TEST_RATIO), seed=SEED)
test_data = split1["test"]
dev_data = split2["test"]
print(f"Test: {len(test_data)}, Dev: {len(dev_data)}")

# ---- Metrics ----
def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
    return dp[n][m]

def compute_wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits / max(total, 1) * 100, 2)

def compute_cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return round(edits / max(total, 1) * 100, 2)

def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())

# ---- Run inference for each model size ----
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

all_results = {}
for model_size in MODEL_SIZES:
    model_name = f"openai/whisper-{model_size}"
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    for split_name, split_data in [("test", test_data), ("dev", dev_data)]:
        refs, hyps = [], []
        t0 = time.time()

        for i, sample in enumerate(split_data):
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            text = normalize_text(sample["text"])

            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(inputs)

            hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            refs.append(text)
            hyps.append(hyp)

            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(split_data)}]")

        elapsed = time.time() - t0
        wer = compute_wer(refs, hyps)
        cer = compute_cer(refs, hyps)

        key = f"{model_size}_{split_name}"
        all_results[key] = {"wer": wer, "cer": cer, "time_sec": round(elapsed, 1), "n_samples": len(refs)}

        print(f"  {split_name}: WER={wer}%, CER={cer}%, time={elapsed:.1f}s")

        # Print some samples
        indices = random.sample(range(len(refs)), min(5, len(refs)))
        for idx in indices:
            print(f"    REF: {refs[idx]}")
            print(f"    HYP: {hyps[idx]}")
            print()

    del model, processor
    torch.cuda.empty_cache()

# ---- Save results to Hub ----
from huggingface_hub import HfApi
api = HfApi(token=token)

# Create results repo if needed
try:
    api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except Exception as e:
    print(f"Repo creation: {e}")

results_json = json.dumps(all_results, indent=2)
print(f"\n{'='*60}")
print("RESULTS:")
print(results_json)
print(f"{'='*60}")

api.upload_file(
    path_or_fileobj=results_json.encode(),
    path_in_repo="C1_whisper_zeroshot/metrics.json",
    repo_id=RESULTS_REPO,
    token=token,
)
print(f"Results pushed to {RESULTS_REPO}")
