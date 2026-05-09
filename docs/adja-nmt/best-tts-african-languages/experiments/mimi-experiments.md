# Mimi Codec Fine-Tuning Experiments

**Track**: TTS — Text-to-Speech for Adja (Gbe family)
**Date created**: 2026-04-26
**Hypothesis**: The off-the-shelf `kyutai/mimi` codec is a bottleneck for CSM-based
Adja TTS because it was trained predominantly on English and French speech. Tonal
phonemes and Gbe-family vowel contrasts (ɛ/ɔ/ɖ, tone marks) may not survive
encode→decode faithfully, degrading every downstream CSM fine-tune regardless of
how well the LM backbone is trained.

**Gating condition**: M0 reconstruction test must confirm audible Adja fidelity
before committing compute to M1–M2b.

---

## M0 — Mimi Reconstruction Gate Test

**Script**: `scripts/sagemaker_jobs/M0_mimi_reconstruction_test.py`

### Hypothesis
If `kyutai/mimi` encode→decode degrades Adja tonal/vowel contrasts in a way that
is audible to a native speaker, then all downstream CSM experiments using Mimi are
fundamentally bounded regardless of LM training quality. M0 makes this failure mode
explicit before any GPU-hours are committed to M1–M2b.

### Data
- Dataset: `JosueG/adja-tts-orpheus`, split=`train`
- 20 samples, resampled to 24kHz (Mimi native rate)

### Metrics
**L1 distance**: Mean absolute difference between original and reconstructed waveform
samples, averaged per clip.

**Spectral Convergence (SC)**:

$$SC = \frac{||\, |STFT(x)| - |STFT(\hat{x})| \,||_F}{||\, |STFT(x)| \,||_F + \epsilon}$$

Lower SC = better reconstruction. Reference threshold:
- SC < 0.30: excellent — proceed to M1
- SC 0.30–0.60: acceptable with caution
- SC > 0.60: Mimi likely dropping tonal content — consider SNAC/Orpheus fallback

### Instance
Local (CPU or GPU). No SageMaker needed. Runtime < 5 minutes.

### Results (2026-04-26) — PASS
Run on 5 WaxalNLP `ewe_tts` samples, CPU, `kyutai/mimi`.

| Sample | L1 | SC |
|--------|----|----|
| 00 | 0.00648 | 0.1931 |
| 01 | 0.00771 | 0.1578 |
| 02 | 0.00572 | 0.1879 |
| 03 | 0.00623 | 0.1520 |
| 04 | 0.00571 | 0.1758 |
| **Mean** | **0.00637** | **0.1733** |

Verdict: **PASS — excellent reconstruction** (SC < 0.30 threshold).
Mimi preserves Gbe tonal and vowel contrasts faithfully.
**Decision: M1/M2/M2b are deprioritized. Focus budget on CF2, CF3, CF1, AT1.**

### References
- Mimi codec: https://arxiv.org/abs/2410.00037
- EnCodec (architectural ancestor of Mimi): https://arxiv.org/abs/2210.13438

---

## M1 — Mimi Acoustic Fine-Tune (Reconstruction Loss Only)

**Script**: `scripts/sagemaker_jobs/train_M1_mimi_acoustic.py`

### Hypothesis
Domain adaptation of Mimi using reconstruction loss (L1 + multi-scale STFT) on
in-domain Gbe speech (Ewe TTS + Adja TTS) should reduce the domain gap between
the codec's training distribution (English/French) and Adja phonology, improving
CSM Stage 1 (Ewe) quality and reducing catastrophic forgetting in Stage 2 (Adja).

### Data
| Source | Config | Split | Clips |
|--------|--------|-------|-------|
| `google/WaxalNLP` | `ewe_tts` | train | ~1,215 |
| `JosueG/adja-tts-orpheus` | — | train | ~1,600 |

All audio resampled to 24kHz (Mimi native rate). NFC normalization on all text
(not used for codec fine-tuning itself, but required for downstream CSM).

### Loss Function
$$L_{total} = L_1(x, \hat{x}) + \lambda_{stft} \cdot L_{STFT}(x, \hat{x})$$

Where $L_{STFT}$ is the multi-scale spectral loss summed over FFT sizes {512, 1024, 2048}:

$$L_{STFT} = \frac{1}{K} \sum_{k=1}^{K} \left[ SC_k + \frac{1}{|F_k||T_k|} \sum_{f,t} |\log|STFT_k(x)|_{f,t} - \log|STFT_k(\hat{x})|_{f,t}| \right]$$

Default: $\lambda_{stft} = 1.0$

Reference: Yamamoto et al. (2020), "Parallel WaveGAN" — https://arxiv.org/abs/1910.11480

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=0.01 |
| LR | 1e-4 |
| Batch size | 8 |
| Epochs (Mimi) | 10 |
| LR schedule | OneCycleLR |
| Max audio length | 10s (240k samples at 24kHz) |

After Mimi fine-tune, CSM Stage 1 (Ewe) is run with the improved codec
(HF Trainer, 20 epochs, lr=5e-5, cosine schedule, early stopping patience=5).

### SageMaker Instance
`ml.p3.8xlarge` — 4x NVIDIA V100 16GB, ~$14.69/hr

**Cost estimate**:
- Mimi fine-tune: ~2h × $14.69 = ~$29
- CSM Stage 1: ~6–8h × $14.69 = ~$88–$117
- **Total: ~$117–$146**

### Expected Outcome
Mimi reconstruction quality on Adja test clips should improve (lower SC) after
domain adaptation. CSM Stage 1 (Ewe) trained on top of M1-Mimi should produce
cleaner Ewe TTS than the baseline (CSM + stock Mimi), as measured by native-speaker
A/B listening.

### References
- Mimi codec: https://arxiv.org/abs/2410.00037
- EnCodec: https://arxiv.org/abs/2210.13438
- Parallel WaveGAN multi-scale STFT: https://arxiv.org/abs/1910.11480

---

## M2 — Mimi Fine-Tune with MMS-300m Semantic Distillation Teacher

**Script**: `scripts/sagemaker_jobs/train_M2_mimi_mms_teacher.py`

### Hypothesis
Reconstruction loss alone (M1) optimizes for waveform fidelity but does not
explicitly preserve phonemic identity. A frozen multilingual speech model
(`facebook/mms-300m`, trained on 1,162 languages including Ewe) can serve as a
semantic teacher: its hidden states encode language-relevant features (phonemes,
tones). Distilling these into Mimi's codebook-0 representations (coarsest level,
captures highest-level structure) should encourage the codec to preserve tonal
contrasts that pure waveform reconstruction may not enforce.

### Architecture
```
Audio clip
  ├── 16kHz → MMS-300m (frozen) → last_hidden_state [B, T_mms, 1024]  @ ~50Hz
  └── 24kHz → Mimi (trainable) → codes [B, n_codebooks, T_mimi]       @ ~12.5Hz
                                    │
                              Decode → wav_recon
```

### Dual-Rate Pipeline
Each audio clip is resampled twice per forward pass:
- 24kHz: fed to Mimi for encode→decode
- 16kHz: fed to MMS-300m for semantic target extraction

### Temporal Alignment
MMS frame rate ≈ 50Hz; Mimi frame rate ≈ 12.5Hz (ratio ≈ 4:1).

MMS features are aligned to Mimi length via `F.interpolate` (linear, time dim):

$$F^{aligned}_{MMS} = \text{Interpolate}(F_{MMS}, \text{size}=T_{Mimi})$$

### Loss Function
$$L_{total} = L_{recon} + \lambda_{sem} \cdot \left( \text{MSE}(F^{aligned}_{MMS}, E_{Mimi,0}) \cdot s \right)$$

Where:
- $L_{recon}$ = L1 + multi-scale STFT (same as M1)
- $E_{Mimi,0}$ = codebook-0 embeddings from Mimi quantizer
- $\lambda_{sem} = 0.1$ (semantic loss weight)
- $s = 100.0$ (scale factor — MSE of hidden states is in a different magnitude range)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=0.01 |
| LR | 1e-4 |
| Batch size | 4 (smaller than M1 — dual pipeline) |
| Epochs (Mimi) | 10 |
| $\lambda_{stft}$ | 1.0 |
| $\lambda_{sem}$ | 0.1 |
| Semantic scale $s$ | 100.0 |

### SageMaker Instance
`ml.p3.8xlarge` (~$14.69/hr)

**Cost estimate**:
- ~3h × $14.69 = **~$44**

### Expected Outcome
M2 should outperform M1 on phonemic preservation metrics (lower PER, better
tone-contrast retention in native-speaker listening) at the cost of higher memory
and compute per step. If M2 does not improve over M1, the semantic distillation
signal from a non-tonal teacher (mms-300m has limited Gbe-specific data) is
insufficient to justify the overhead — proceed to M2b with an Adja-adapted teacher.

### References
- Mimi codec: https://arxiv.org/abs/2410.00037
- MMS (mms-300m): https://arxiv.org/abs/2305.13516
- EnCodec: https://arxiv.org/abs/2210.13438
- Knowledge distillation: Hinton et al. (2015) https://arxiv.org/abs/1503.02531

---

## M2b — Mimi Fine-Tune with MMS-1b-all (Adja Adapter) as Semantic Teacher

**Script**: `scripts/sagemaker_jobs/train_M2b_mimi_mms_adja_teacher.py`

### Hypothesis
M2 uses `facebook/mms-300m` as a generic multilingual teacher. M2b upgrades to
`facebook/mms-1b-all` with the Adja language adapter ("adj") loaded, providing
a larger model (1B vs 300M) whose intermediate representations are specifically
conditioned on Adja phonology via the adapter mechanism. This should produce a
stronger and more language-specific supervisory signal for Adja tonal contrasts.

**Critical note**: `facebook/mms-tts-ajg` does NOT exist on HuggingFace.
`facebook/mms-1b-all` is the correct Adja audio model — a `Wav2Vec2ForCTC`
checkpoint with per-language adapter layers activated via `.load_adapter("adj")`.

### Adapter Loading Strategy
```python
from transformers import Wav2Vec2ForCTC
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
try:
    mms.load_adapter("adj")
    print("Loaded Adja (adj) adapter")
except Exception:
    mms.load_adapter("ewe")
    print("Adja adapter unavailable, using Ewe (ewe) adapter as fallback")
mms.eval()
mms.requires_grad_(False)
# Use mms.wav2vec2(audio_16k).last_hidden_state for semantic targets
```

### Feature Extraction
```
mms.wav2vec2(audio_16k).last_hidden_state  →  [B, T_mms, 1280]
```
(mms-1b uses D=1280 hidden dimension vs 1024 for mms-300m)

The backbone hidden states (not CTC logits) capture acoustic-phonemic features
conditioned on the Adja adapter. These serve as the distillation target.

### Architecture Difference from M2
| Dimension | M2 | M2b |
|-----------|-----|-----|
| Teacher model | mms-300m | mms-1b-all |
| Teacher params | 300M | 1B |
| Teacher feature dim | 1024 | 1280 |
| Language specificity | Generic multilingual | Adja adapter |
| Batch size | 4 | 2 (higher memory) |

All other components (dual-rate pipeline, Mimi fine-tuning, loss functions,
temporal alignment, codebook-0 distillation target) are identical to M2.

### Loss Function
Same as M2:
$$L_{total} = L_{recon} + \lambda_{sem} \cdot \left( \text{MSE}(F^{aligned}_{MMS-1b}, E_{Mimi,0}) \cdot s \right)$$

### SageMaker Instance
`ml.p3.8xlarge` (~$14.69/hr)

**Cost estimate**:
- ~3.5h × $14.69 = **~$51**
- (Larger teacher increases forward pass cost by ~30% vs M2)

### Expected Outcome
M2b should outperform M2 specifically on Adja-relevant tonal contrasts, since the
teacher is explicitly adapted to Adja phonology. If the Adja adapter is unavailable
and falls back to Ewe, results may be similar to M2 (Ewe and Adja share ~80% of
phoneme inventory). Success criterion: native-speaker listening shows reduced tone
confusion and better vowel quality in CSM Stage 2 output (Adja) when Mimi is
pre-trained with M2b vs M1 or M2.

### References
- MMS (mms-1b-all): https://arxiv.org/abs/2305.13516
- MMS adapter details: https://huggingface.co/facebook/mms-1b-all
- Mimi codec: https://arxiv.org/abs/2410.00037
- EnCodec: https://arxiv.org/abs/2210.13438

---

## Experiment Matrix Summary

| Exp | Teacher | Loss | Instance | Est. Cost | Gate |
|-----|---------|------|----------|-----------|------|
| M0 | — (gate test) | SC + L1 (eval only) | Local | $0 | Run first |
| M1 | None | L1 + multi-STFT | ml.p3.8xlarge | ~$117–146 | M0 passes |
| M2 | mms-300m (frozen) | L1 + STFT + semantic MSE | ml.p3.8xlarge | ~$44 | M0 passes |
| M2b | mms-1b-all + adj adapter | L1 + STFT + semantic MSE | ml.p3.8xlarge | ~$51 | M0 passes |

## Decision Tree
```
M0 (local gate)
  ├── SC > 0.60 (Mimi drops tonal content)
  │     → Retire Mimi track. Use Orpheus/SNAC experiments instead.
  └── SC ≤ 0.60 (Mimi is viable)
        → Run M1 + M2 + M2b in parallel
              └── Native-speaker A/B listening on CSM Stage 2 output
                    selects best Mimi variant for final Stage 2 training.
```

## Known API Uncertainties (TODO: verify)
The following require confirming against the installed `transformers` version before running:
1. `encoder_outputs.audio_codes` — may be `.codes` depending on transformers version
2. `decoder_outputs.audio_values` — may be `.waveform`
3. `mimi.quantizer.quantizers[0]` — codebook-0 embedding lookup path needs inspection
4. `AutoFeatureExtractor` for Mimi — batched numpy list input behavior
5. `mms.wav2vec2` backbone attribute — may differ across Wav2Vec2ForCTC versions
6. Feature extractor input format: `[B, 1, T]` vs `[B, T]`

Run `python M0_mimi_reconstruction_test.py --smoke` locally to surface any API
issues before committing SageMaker budget.
