#!/usr/bin/env python3
"""
Local diagnostic: test whether Mimi codec faithfully reconstructs Adja audio.

Hypothesis to test: Mimi is a waveform-level codec (language-agnostic), so it
should encode and decode Adja tones, ATR vowels, and nasals without phonemic loss.
If reconstruction sounds clean and tonal quality is preserved, the codec is NOT
the bottleneck for CSM failures on Adja (pointing to the Llama tokenizer instead).

Usage (no GPU needed):
  pip install moshi soundfile datasets huggingface_hub
  HF_TOKEN=<your_token> python mimi_reconstruction_test.py

Outputs 5 reconstructed WAV files to ./mimi_recon/ — listen and judge quality.

References:
  - Mimi codec: https://arxiv.org/abs/2410.00037
  - Failure analysis: learnings-from-the-past/training-gotchas.md
"""
import os, sys, unicodedata
from pathlib import Path

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN required (needed to load the Adja dataset)")

print("Installing dependencies...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "moshi", "soundfile", "datasets>=3.4.1", "huggingface_hub", "numpy"])

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset

print("Loading Adja dataset (first 10 samples)...")
ds = load_dataset("JosueG/adja-tts-orpheus", token=token, split="train")
samples = ds.select(range(min(10, len(ds))))

print("Loading Mimi codec...")
from moshi.models import loaders
mimi, _ = loaders.get_mimi("kyutai/mimi", device="cpu")
mimi.eval()
print(f"Mimi loaded. Sample rate: {mimi.sample_rate}Hz  |  Frame rate: {mimi.frame_rate}Hz\n")

OUT_DIR = Path("./mimi_recon")
OUT_DIR.mkdir(exist_ok=True)


def normalize_text(t):
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())


results = []
for i, ex in enumerate(samples):
    text = normalize_text(ex["text"])
    audio_arr = np.array(ex["audio"]["array"], dtype=np.float32)
    orig_sr = ex["audio"]["sampling_rate"]

    # Resample to Mimi's expected 24kHz
    if orig_sr != mimi.sample_rate:
        import torchaudio.transforms as T
        wf = torch.from_numpy(audio_arr).unsqueeze(0)
        audio_arr = T.Resample(orig_freq=orig_sr, new_freq=mimi.sample_rate)(wf).squeeze(0).numpy()

    # Save original
    orig_path = OUT_DIR / f"{i:02d}_original.wav"
    sf.write(str(orig_path), audio_arr, mimi.sample_rate)

    # Encode → decode
    wav_t = torch.from_numpy(audio_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    try:
        with torch.no_grad():
            codes = mimi.encode(wav_t)                 # [B, K, T_frames]
            recon = mimi.decode(codes).squeeze().numpy()  # [T]
        recon_path = OUT_DIR / f"{i:02d}_recon.wav"
        sf.write(str(recon_path), recon.astype(np.float32), mimi.sample_rate)

        orig_dur = len(audio_arr) / mimi.sample_rate
        recon_dur = len(recon) / mimi.sample_rate
        n_frames = codes.shape[-1]
        print(f"[{i:02d}] '{text[:60]}'")
        print(f"       orig={orig_dur:.2f}s  recon={recon_dur:.2f}s  codec_frames={n_frames}")
        results.append({"idx": i, "text": text, "orig_dur": orig_dur,
                        "recon_dur": recon_dur, "codec_frames": n_frames, "status": "ok"})
    except Exception as e:
        print(f"[{i:02d}] FAILED: {e}")
        results.append({"idx": i, "text": text, "status": f"error: {e}"})

print(f"\nOutput: {OUT_DIR.resolve()}")
print("\nListen to each pair (original vs recon) and check:")
print("  - Do tonal contrasts (é/è) survive reconstruction?")
print("  - Are ATR vowels (ɛ vs e, ɔ vs o) distinguishable?")
print("  - Is there any smearing of short consonants or nasals?")
print("\nIf reconstruction is clean → codec is NOT the CSM bottleneck.")
print("If tones or vowels are lost → codec IS a bottleneck (unexpected).")
