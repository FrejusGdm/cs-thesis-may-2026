# Mimi Codec Fine-Tuning Guide

Technical reference for M1 (acoustic-only) and M2 (semantic re-distillation) Mimi fine-tuning
experiments. Read this before writing any M1/M2 training code.

**Run M0 (reconstruction gate) before implementing M1 or M2.** If Mimi reconstruction on Adja
audio already sounds acceptable, the codec is not the binding constraint and this track can be
deprioritized in favor of CF experiments.

---

## Mimi Architecture

Paper: https://arxiv.org/abs/2410.00037 (Défossez et al., Kyutai, 2024)

Mimi is a neural audio codec trained to compress speech at 24 kHz into a compact discrete
representation while preserving both acoustic quality and semantic content.

**Architecture stack:**

```
Input waveform (24 kHz)
    ↓
Convolutional encoder (EnCodec-style strided convolutions)
    ↓
32-level Residual Vector Quantizer (RVQ)
    — only levels 0–7 (8 codebooks) are used in practice in CSM
    — codebook size: 2048 per level
    — total discrete tokens per frame: 8 codes at 12.5 Hz = 8 × 12.5 = 100 codes/sec
    ↓
Transformer encoder + decoder (contextual refinement, Moshi's addition over EnCodec)
    ↓
Convolutional decoder
    ↓
Reconstructed waveform (24 kHz)
```

**Frame rate**: Mimi operates at **12.5 Hz** (one RVQ frame every 80 ms = 1920 samples at 24 kHz).
This is important for alignment with other audio models.

**Training objectives (original):**
1. L1 reconstruction loss on the waveform
2. Multi-scale STFT loss
3. Adversarial (GAN) loss — perceptual quality, needed for scratch training
4. WavLM semantic distillation on codebook 0 — aligns the first codebook with speech semantics

---

## KEY INSIGHT: WavLM Is Not In the Mimi Checkpoint

**This is the most common misunderstanding about Mimi.**

WavLM was used as a **training teacher only** during Mimi's original training to distill semantic
information into codebook 0. The WavLM model is **not** part of the `kyutai/mimi` checkpoint.

The `kyutai/mimi` HuggingFace checkpoint contains:
- Convolutional encoder weights
- RVQ codebook embeddings (all 32 levels)
- Transformer encoder/decoder weights
- Convolutional decoder weights

It does **not** contain WavLM, any other SSL model, or discriminator weights.

Implication for M1: you can fine-tune `MimiModel` for acoustic reconstruction on Adja audio
**without** WavLM. L1 + STFT loss is sufficient for domain adaptation. You are not reproducing
Mimi from scratch; you are adapting an existing checkpoint to a new acoustic domain.

Implication for M2: semantic re-distillation (aligning codebook 0 with an MMS-300M teacher on
Adja speech) requires loading MMS-300M separately and computing the distillation target at training
time. This adds ~300M parameters to the training pipeline (teacher is frozen) and significantly
increases compute cost.

---

## Can We Fine-Tune Mimi?

**Yes.** `MimiModel` is a standard HuggingFace `PreTrainedModel` and supports gradient descent
like any other HF model.

```python
from transformers import MimiModel, AutoFeatureExtractor

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
```

The encoder, RVQ codebooks, and decoder are all trainable. Standard `model.train()` /
`optimizer.step()` applies.

**What `kyutai-labs/moshi-finetune` does NOT do:** that repository fine-tunes the Moshi
language model (the LLM that generates Mimi token sequences), not Mimi's encoder/decoder/RVQ.
Do not confuse the two.

---

## M1: Acoustic-Only Mimi Fine-Tuning

**Hypothesis**: Mimi's convolutional encoder/decoder and RVQ codebooks were trained on
English-heavy data. Fine-tuning the full encoder+decoder+RVQ on Adja audio using reconstruction
loss adapts the codec's acoustic representation to Adja phonemes.

**Loss function** (no GAN, no WavLM teacher needed for domain adaptation):

```python
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

def reconstruction_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor,
                        sample_rate: int = 24000) -> torch.Tensor:
    """L1 + multi-scale STFT loss for Mimi fine-tuning."""
    l1 = F.l1_loss(pred_wav, target_wav)

    stft_loss = torch.tensor(0.0, device=pred_wav.device)
    for fft_size in [512, 1024, 2048]:
        hop = fft_size // 4
        pred_spec = torch.stft(pred_wav.squeeze(1), n_fft=fft_size, hop_length=hop,
                               return_complex=True).abs()
        tgt_spec = torch.stft(target_wav.squeeze(1), n_fft=fft_size, hop_length=hop,
                              return_complex=True).abs()
        stft_loss = stft_loss + F.l1_loss(pred_spec.log1p(), tgt_spec.log1p())

    return l1 + stft_loss / 3
```

**Training loop sketch:**

```python
from transformers import MimiModel, AutoFeatureExtractor
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
model = MimiModel.from_pretrained("kyutai/mimi")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

for batch in dataloader:
    audio = batch["audio"]  # (B, 1, T) at 24 kHz
    inputs = feature_extractor(audio, sampling_rate=24000, return_tensors="pt")

    # Encode to discrete codes, then decode back
    outputs = model(**inputs)
    reconstructed = outputs.audio_values  # (B, 1, T)

    loss = reconstruction_loss(reconstructed, audio)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**What to fine-tune**: encoder + decoder + RVQ codebooks. All parameters. LoRA is not standard
for codec models (codebook embeddings are not linear projections); full fine-tuning on ~1.7h of
Adja audio is feasible given Mimi's size (~150M params).

---

## M2: Semantic Re-Distillation with MMS-300M Teacher

**Hypothesis**: Mimi's codebook 0 was originally distilled to align with WavLM semantic features.
Re-distilling codebook 0 with an MMS-300M teacher (which covers 1000+ languages including several
African ones) adds African phonological coverage to the codec's semantic representation.

**Frame-rate alignment note**: Mimi operates at 12.5 Hz; MMS-300M produces features at ~50 Hz
(20 ms frames). To compute the distillation loss, you must downsample the MMS features to 12.5 Hz
using `torch.nn.functional.interpolate`:

```python
import torch.nn.functional as F

# mms_features: (B, T_mms, D_mms) at 50 Hz
# mimi_codebook0: (B, T_mimi, D_mimi) at 12.5 Hz
# T_mms / T_mimi ≈ 4 (50 / 12.5)

mms_downsampled = F.interpolate(
    mms_features.permute(0, 2, 1),  # (B, D, T_mms)
    size=mimi_codebook0.shape[1],   # T_mimi
    mode="linear",
    align_corners=False
).permute(0, 2, 1)  # (B, T_mimi, D_mms)

distill_loss = F.mse_loss(mimi_codebook0_proj, mms_downsampled.detach())
```

Where `mimi_codebook0_proj` is a learned linear projection from Mimi's codebook-0 embedding
dimension to MMS's feature dimension.

**Total M2 loss**:
```
total_loss = reconstruction_loss(reconstructed, original)
           + lambda_distill * distill_loss
```

Recommended `lambda_distill ≈ 0.1` to start; tune if codebook 0 diverges.

**Compute note**: M2 requires loading both Mimi (~150M) and MMS-300M (~300M) on the same GPU.
On L40S (48 GB), this is feasible if both are in bf16 and batch sizes are small. On A100 80GB,
comfortable for bf16 full models + gradient storage.

---

## M1 vs M2: When to Run Which

| Question | M1 (acoustic) | M2 (semantic) |
|----------|--------------|--------------|
| Does Mimi encode Adja phonemes accurately? | Tests this via reconstruction quality | Tests this + semantic alignment |
| Do you need WavLM or another teacher? | No | No (MMS-300M teacher instead) |
| Compute cost | Lower (1 model, L1+STFT only) | Higher (~2x, MMS teacher + distillation) |
| Risk of codec divergence | Lower | Higher (distillation loss can destabilize RVQ) |
| Run first? | YES — M1 is the baseline | Only if M1 is insufficient |

**Decision rule**: run M1 first. If M1-adapted CSM Stage 2 produces better intelligibility than
original-Mimi CSM Stage 2, report M1. If M1 shows no improvement, run M2 to test whether semantic
re-distillation adds anything beyond acoustic reconstruction.

---

## MimiModel API Reference

```python
from transformers import MimiModel, AutoFeatureExtractor
import torch

# Load
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
model = MimiModel.from_pretrained("kyutai/mimi")

# Encode raw audio to discrete codes
audio = torch.randn(1, 1, 24000)  # 1 second at 24 kHz, (B, C, T)
inputs = feature_extractor(audio.squeeze(0).numpy(), sampling_rate=24000,
                            return_tensors="pt")
with torch.no_grad():
    codes = model.encode(inputs["input_values"])
    # codes.audio_codes: (B, num_codebooks, T_frames) — T_frames = T / 1920 at 24kHz

# Decode codes back to audio
with torch.no_grad():
    reconstructed = model.decode(codes.audio_codes)
    # reconstructed.audio_values: (B, 1, T)

# Full forward pass (encode + decode)
with torch.no_grad():
    outputs = model(**inputs)
    # outputs.audio_values: reconstructed waveform
    # outputs.audio_codes: discrete codes
    # outputs.projected_quantized_states: pre-quantization features (for distillation target)
```

**Key attributes:**
- `model.encoder` — convolutional encoder + Transformer encoder
- `model.decoder` — convolutional decoder + Transformer decoder
- `model.quantizer` — RVQ with 32 codebooks, 2048 entries each

**Frame rate formula**: `T_frames = ceil(T_samples / 1920)` for 24 kHz input.

---

## M0: Reconstruction Gate (Run Before M1/M2)

Before running M1 or M2, verify that Mimi reconstruction quality on Adja is actually degraded.
This is a free local test (~30 minutes).

```python
from transformers import MimiModel, AutoFeatureExtractor
from datasets import load_dataset
import soundfile as sf
import torch
import os

model = MimiModel.from_pretrained("kyutai/mimi").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

ds = load_dataset("JosueG/adja-tts-orpheus", split="train")
os.makedirs("/tmp/mimi_gate", exist_ok=True)

for i, sample in enumerate(ds.select(range(10))):
    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    # Resample to 24 kHz if needed
    if sr != 24000:
        import torchaudio
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        audio_array = torchaudio.functional.resample(audio_tensor, sr, 24000).squeeze(0).numpy()

    inputs = feature_extractor(audio_array, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    reconstructed = outputs.audio_values.squeeze().numpy()
    sf.write(f"/tmp/mimi_gate/original_{i:02d}.wav", audio_array, 24000)
    sf.write(f"/tmp/mimi_gate/reconstructed_{i:02d}.wav", reconstructed, 24000)
    print(f"Sample {i}: original {len(audio_array)/24000:.2f}s, "
          f"reconstructed {len(reconstructed)/24000:.2f}s")

print("Listen to /tmp/mimi_gate/original_*.wav vs reconstructed_*.wav")
print("If reconstructed sounds like distorted/foreign-language audio → M1/M2 needed")
print("If reconstructed sounds like same-language but slightly degraded → codec is fine, focus on CF")
```

**Interpretation:**
- Reconstructed audio sounds like degraded Adja (same phonemes, slightly fuzzy): codec is adequate.
  Move to CF experiments.
- Reconstructed audio sounds like a different language or pure noise: codec is the bottleneck.
  Implement M1 before running any Stage 2 experiments.
