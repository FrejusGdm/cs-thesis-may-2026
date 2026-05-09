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
#
# KNOWN ISSUE (2026-04-16): omnilingual-asr pins fairseq2 which requires
# torch==2.8.0 (CUDA 13). HF Jobs standard runners have CUDA 12 only, so
# torchaudio fails to load libcudart.so.13. To run this, use:
#   --image omnilingual/asr:latest   (if Meta publishes one), or
#   run on HPC with CUDA 13 drivers, or
#   run locally on a GPU with CUDA 13.
"""
Omnilingual-ASR zero-shot inference on Adja.

Meta released Omnilingual ASR in Nov 2025 with zero-shot support for
1,600+ languages via language tags. For Aja (Benin), use `ajg_Latn`.

This is NOT fine-tuning — we just run the pre-trained model on our
test + dev splits and measure CER/WER. If it works well out of the box,
it's a massive baseline. If not, we still learn something about
Meta's zero-shot claims.

Requires: pip install omnilingual-asr (built on fairseq2, not transformers)
Model: omniASR_LLM_7B_ZS (the zero-shot variant)
"""
from __future__ import annotations
import importlib.util
import json, os, shutil, subprocess, sys, time, unicodedata, random, tempfile, threading
import numpy as np
import soundfile as sf
sys.stdout.reconfigure(line_buffering=True)


def ensure_omnilingual_asr_installed() -> None:
    """Install omnilingual-asr without kenlm to avoid build-time failures on HF Jobs."""
    if importlib.util.find_spec("omnilingual_asr") is not None:
        return

    print("Installing omnilingual-asr==0.2.0 (--no-deps)...")
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, "--no-deps", "omnilingual-asr==0.2.0"],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "omnilingual-asr==0.2.0"],
            check=True,
        )


ensure_omnilingual_asr_installed()

DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
SEED = 42
LANG_CODE = os.environ.get("LANG_CODE", "ajg_Latn")
MODEL_CARD = os.environ.get("MODEL_CARD", "omniASR_LLM_300M")
MAX_SAMPLES_PER_SPLIT = int(os.environ.get("MAX_SAMPLES_PER_SPLIT", "0"))

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

LANG_ALIASES = {
    # Common project tags for Aja -> OmniASR map key
    "aj_Latn": "ajg_Latn",
    "ajg_Latn": "ajg_Latn",
}
requested_lang_code = LANG_CODE
LANG_CODE = LANG_ALIASES.get(LANG_CODE, LANG_CODE)
if requested_lang_code != LANG_CODE:
    print(f"Mapped LANG_CODE {requested_lang_code} -> {LANG_CODE}")
if LANG_CODE not in supported_langs:
    raise ValueError(
        f"LANG_CODE={LANG_CODE} is not supported by OmniASR. "
        "Please choose a value from omnilingual_asr.models.wav2vec2_llama.lang_ids.supported_langs."
    )

# ---- Load data ----
print("Loading dataset...")
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=SEED)
test_data = split1["test"]
dev_data = split2["test"]

if MAX_SAMPLES_PER_SPLIT > 0:
    test_data = test_data.select(range(min(MAX_SAMPLES_PER_SPLIT, len(test_data))))
    dev_data = dev_data.select(range(min(MAX_SAMPLES_PER_SPLIT, len(dev_data))))
    print(f"Using capped split size: {len(test_data)} test, {len(dev_data)} dev")

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

def cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r,h in zip(refs,hyps))
    total = sum(len(r) for r in refs)
    return round(edits/max(total,1)*100, 2)

def wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r,h in zip(refs,hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits/max(total,1)*100, 2)

def normalize_text(t):
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())

# ---- Load Omnilingual ASR pipeline ----
print(f"Loading {MODEL_CARD} (zero-shot)...")
import librosa
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

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

# ---- Inference loop ----
results = {}
for split_name, split_data in [("dev", dev_data), ("test", test_data)]:
    print(f"\n{'='*50}")
    print(f"Split: {split_name} ({len(split_data)} samples)")
    print(f"{'='*50}")

    refs, hyps = [], []
    total_audio_sec = 0
    t0 = time.time()

    for i, sample in enumerate(split_data):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = normalize_text(sample["text"])

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        total_audio_sec += len(audio) / 16000

        # Omnilingual ASR pipeline takes file paths — write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            tmp_path = tmp.name

        try:
            result = pipeline.transcribe([tmp_path], lang=[LANG_CODE], batch_size=1)
            hyp = result[0] if result and len(result) > 0 else ""
            if isinstance(hyp, dict):
                hyp = hyp.get("text", "") or hyp.get("transcription", "") or str(hyp)
            hyp = normalize_text(str(hyp).strip())
        except Exception as e:
            print(f"  [{i}] Inference error: {e}")
            hyp = ""
        finally:
            os.unlink(tmp_path)

        refs.append(text)
        hyps.append(hyp)

        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(split_data)}]")

    elapsed = time.time() - t0
    cer_val = cer(refs, hyps)
    wer_val = wer(refs, hyps)

    results[split_name] = {
        "cer": cer_val, "wer": wer_val,
        "num_samples": len(refs),
        "inference_time_sec": round(elapsed, 1),
        "audio_duration_sec": round(total_audio_sec, 1),
        "rtf": round(elapsed / max(total_audio_sec, 1), 3),
    }

    print(f"\n  {split_name.upper()}: CER={cer_val}%, WER={wer_val}%, time={elapsed:.1f}s, RTF={results[split_name]['rtf']}")

    # Print random decode samples
    print(f"\n  Decode samples:")
    for idx in random.sample(range(len(refs)), min(10, len(refs))):
        print(f"    REF: {refs[idx]}")
        print(f"    HYP: {hyps[idx]}")
        print()

# ---- Push results ----
from huggingface_hub import HfApi
api = HfApi(token=token)
try:
    api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except Exception as e:
    print(f"Repo creation: {e}")

final = {
    "experiment": "Omni_ZS",
    "model": MODEL_CARD,
    "approach": "zero-shot (no training)",
    "language_code_requested": requested_lang_code,
    "language_code_effective": LANG_CODE,
    "results": results,
}
api.upload_file(
    path_or_fileobj=json.dumps(final, indent=2).encode(),
    path_in_repo="Omni_ZS/metrics.json",
    repo_id=RESULTS_REPO, token=token,
)
print(f"\nResults pushed to {RESULTS_REPO}/Omni_ZS/")
print(json.dumps(final, indent=2))
