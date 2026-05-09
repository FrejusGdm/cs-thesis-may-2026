# /// script
# dependencies = [
#   "datasets",
#   "soundfile",
#   "librosa",
#   "numpy",
#   "huggingface-hub",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "fairseq2==0.6.0",
#   "pyarrow",
#   "pandas",
#   "numba",
#   "polars>=1.29.0",
#   "retrying",
#   "xxhash",
# ]
# ///
"""
Omnilingual-ASR in-context learning (ICL) inference for Adja.

Uses omniASR_LLM_7B_ZS with context audio/transcript pairs.
This script is separate from zero-shot language-conditioned runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import threading
import gc

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)


def ensure_omnilingual_asr_installed() -> None:
    if importlib.util.find_spec("omnilingual_asr") is not None:
        return

    print("Installing omnilingual-asr==0.2.0 (--no-deps)...")
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "omnilingual-asr==0.2.0",
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "omnilingual-asr==0.2.0",
            ],
            check=True,
        )


ensure_omnilingual_asr_installed()

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, ContextExample

DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
MODEL_CARD = os.environ.get("MODEL_CARD", "omniASR_LLM_7B_ZS")
SEED = int(os.environ.get("SEED", "42"))
MAX_SAMPLES_PER_SPLIT = int(os.environ.get("MAX_SAMPLES_PER_SPLIT", "0"))
CONTEXT_POOL = os.environ.get("CONTEXT_POOL", "train")  # train | train+dev
TARGET_MAX_SEC = float(os.environ.get("TARGET_MAX_SEC", "25"))
CONTEXT_MAX_SEC = float(os.environ.get("CONTEXT_MAX_SEC", "8"))
CONTEXT_K_LIST_RAW = os.environ.get("CONTEXT_K_LIST")
if CONTEXT_K_LIST_RAW:
    CONTEXT_K_LIST = [int(x.strip()) for x in CONTEXT_K_LIST_RAW.split(",") if x.strip()]
else:
    CONTEXT_K_LIST = [int(os.environ.get("CONTEXT_K", "3"))]

for context_k in CONTEXT_K_LIST:
    if context_k < 1 or context_k > 10:
        raise ValueError(f"CONTEXT_K must be in [1, 10], got {context_k}")

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"


def normalize_text(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
    return dp[n][m]


def cer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return round(edits / max(total, 1) * 100, 2)


def wer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits / max(total, 1) * 100, 2)


print("Loading dataset...")
from datasets import load_dataset

ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=SEED)
train_data = split2["train"]
dev_data = split2["test"]
test_data = split1["test"]

if MAX_SAMPLES_PER_SPLIT > 0:
    dev_data = dev_data.select(range(min(MAX_SAMPLES_PER_SPLIT, len(dev_data))))
    test_data = test_data.select(range(min(MAX_SAMPLES_PER_SPLIT, len(test_data))))
    print(f"Using capped split size: dev={len(dev_data)} test={len(test_data)}")

print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
print(f"ICL config: CONTEXT_K_LIST={CONTEXT_K_LIST}, CONTEXT_POOL={CONTEXT_POOL}")
print(f"Audio caps: TARGET_MAX_SEC={TARGET_MAX_SEC}, CONTEXT_MAX_SEC={CONTEXT_MAX_SEC}")

print(f"Loading {MODEL_CARD} (ICL)...")
_load_started = time.time()
_load_stop = threading.Event()


def _load_heartbeat() -> None:
    while not _load_stop.wait(30):
        elapsed = int(time.time() - _load_started)
        print(f"Still loading {MODEL_CARD}... elapsed={elapsed}s")


heartbeat_thread = threading.Thread(target=_load_heartbeat, daemon=True)
heartbeat_thread.start()
try:
    pipeline = ASRInferencePipeline(model_card=MODEL_CARD)
finally:
    _load_stop.set()
    heartbeat_thread.join(timeout=1)
print("Model loaded.")


def sample_to_temp_wav(sample, max_sec: float) -> tuple[str, str]:
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    text = normalize_text(sample["text"])
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if max_sec > 0:
        max_samples = int(max_sec * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        return tmp.name, text


def build_context_pool(split_name: str, split_data):
    pool = list(train_data)
    if CONTEXT_POOL == "train+dev" and split_name != "dev":
        pool = pool + list(dev_data)
    # Keep context separate from evaluation split to avoid leakage.
    return pool


def run_split(split_name: str, split_data, context_k: int):
    print(f"\n{'='*50}")
    print(f"Split: {split_name} ({len(split_data)} samples)")
    print(f"{'='*50}")

    refs: list[str] = []
    hyps: list[str] = []
    total_audio_sec = 0.0
    t0 = time.time()
    rng = random.Random(SEED + context_k * 100 + (0 if split_name == "dev" else 1))
    context_pool = build_context_pool(split_name, split_data)

    for i, sample in enumerate(split_data):
        target_wav, target_text = sample_to_temp_wav(sample, max_sec=TARGET_MAX_SEC)
        # Best-effort audio duration estimate from raw array + sr.
        total_audio_sec += len(sample["audio"]["array"]) / max(sample["audio"]["sampling_rate"], 1)

        context_samples = rng.sample(context_pool, k=min(context_k, len(context_pool)))
        context_examples = []
        context_paths = []
        for c in context_samples:
            c_wav, c_txt = sample_to_temp_wav(c, max_sec=CONTEXT_MAX_SEC)
            context_paths.append(c_wav)
            context_examples.append(ContextExample(audio=c_wav, text=c_txt))

        try:
            result = pipeline.transcribe_with_context(
                [target_wav],
                [context_examples],
                batch_size=1,
            )
            hyp = normalize_text(str(result[0]).strip()) if result else ""
        except Exception as e:
            print(f"  [{i}] Inference error: {e}")
            hyp = ""
        finally:
            for p in context_paths:
                if os.path.exists(p):
                    os.unlink(p)
            if os.path.exists(target_wav):
                os.unlink(target_wav)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        refs.append(target_text)
        hyps.append(hyp)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(split_data)}]")

    elapsed = time.time() - t0
    split_cer = cer(refs, hyps)
    split_wer = wer(refs, hyps)

    print(
        f"\n  {split_name.upper()}: CER={split_cer}%, "
        f"WER={split_wer}%, time={elapsed:.1f}s, "
        f"RTF={round(elapsed / max(total_audio_sec, 1.0), 3)}"
    )
    print("\n  Decode samples:")
    for idx in random.sample(range(len(refs)), min(5, len(refs))):
        print(f"    REF: {refs[idx]}")
        print(f"    HYP: {hyps[idx]}")
        print()

    return {
        "cer": split_cer,
        "wer": split_wer,
        "num_samples": len(refs),
        "inference_time_sec": round(elapsed, 1),
        "audio_duration_sec": round(total_audio_sec, 1),
        "rtf": round(elapsed / max(total_audio_sec, 1.0), 3),
    }


from huggingface_hub import HfApi

api = HfApi(token=token)
api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
all_payloads = []
for context_k in CONTEXT_K_LIST:
    experiment_name = f"Omni_ICL_k{context_k}"
    print(f"\nRunning experiment: {experiment_name}")
    results = {
        "dev": run_split("dev", dev_data, context_k),
        "test": run_split("test", test_data, context_k),
    }
    payload = {
        "experiment": experiment_name,
        "model": MODEL_CARD,
        "approach": "in-context learning (no fine-tuning)",
        "context_k": context_k,
        "context_pool": CONTEXT_POOL,
        "seed": SEED,
        "results": results,
    }
    api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=2).encode(),
        path_in_repo=f"{experiment_name}/metrics.json",
        repo_id=RESULTS_REPO,
        token=token,
    )
    print(f"\nResults pushed to {RESULTS_REPO}/{experiment_name}/")
    print(json.dumps(payload, indent=2))
    all_payloads.append(payload)

print("\nAll ICL experiment payloads:")
print(json.dumps(all_payloads, indent=2))
