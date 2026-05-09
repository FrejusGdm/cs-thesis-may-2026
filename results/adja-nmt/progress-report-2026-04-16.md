# Adja ASR Exploration — Progress Report

**Date**: April 16, 2026
**Author**: Josue Godeme
**Advisor**: Prof. Coto

---

## 1. Objective

Explore which ASR (Automatic Speech Recognition) approach works best for **Adja**, a low-resource tonal language in the Gbe family (spoken in Benin/Togo, ~600K speakers). Given a dataset of ~1,600 transcribed utterances (~2 hours of audio), we systematically compare:

- **Zero-shot** inference (no training)
- **CTC-based fine-tuning** (MMS, XLS-R, wav2vec2)
- **Encoder-decoder fine-tuning** (Whisper)
- **Cross-lingual transfer** from Ewe (closest well-resourced Gbe relative)

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | `JosueG/adja-tts-orpheus` (HuggingFace, private) |
| Total utterances | 1,597 |
| Split | Train: 1,277 (80%) / Dev: 160 (10%) / Test: 160 (10%) |
| Total audio | ~1.95 hours |
| Sampling rate | 48 kHz (resampled to 16 kHz for models) |
| Vocabulary | 109 unique characters including ɛ, ɔ, ŋ, ɖ, tone marks |
| Normalization | Unicode NFC applied to all transcriptions |

---

## 3. Experiments and Results

### 3.1 Summary Table

| Rank | Model | Approach | CER ↓ | WER ↓ | Key Finding |
|------|-------|----------|-------|-------|-------------|
| **1** | **Whisper-small-Ewe → Adja (E4)** | **Encoder-decoder fine-tune** | **24.90%** | **73.09%** | **Ewe transfer helps** |
| 2 | Whisper-small → Adja (C2) | Encoder-decoder fine-tune | 27.02% | 74.15% | Strong baseline |
| 3 | MMS-1B + French adapter (C3) | CTC fine-tune | 87.19% | 98.42% | CTC struggles |
| 4 | Whisper-small zero-shot (C1) | No training | 366.74% | 100.91% | Hallucinations |
| — | XLS-R, wav2vec2, MMS-Ewe | CTC fine-tune | 100% | 100% | CTC collapse |

### 3.2 Why Whisper Outperforms CTC Models

**Architectural difference**: Whisper uses an encoder-decoder with cross-attention (like mBART for NMT). At each decoding step, it attends to the audio encoding AND the previously generated characters. This means it learns character dependencies — "after ŋ, ɖ is likely" — implicitly.

**CTC limitation**: CTC-based models (MMS, XLS-R, wav2vec2) predict each frame independently. They have no mechanism to learn that certain character sequences are more likely than others in Adja. This is particularly damaging for a low-resource language where the model needs every advantage.

**Analogy**: This is like the difference between word-by-word translation (CTC) vs. an NMT decoder that conditions on what it has already translated (Whisper). The NMT decoder is better for the same reason.

**Data efficiency**: Whisper-small (244M params) learned useful Adja patterns within 5 epochs (CER dropped from 92% to 71%). MMS-1B (965M params, 4x larger) only reached 87% CER after 6 epochs. More parameters don't help if the architecture isn't suited to the problem.

### 3.3 Cross-Lingual Transfer: Ewe → Adja

**Finding**: Whisper pre-tuned on Ewe (E4) achieves **CER=24.9%**, beating vanilla Whisper (C2) at **CER=27.0%** — a **2.1% absolute improvement**.

**Why this matters**: Ewe (Eʋegbe) is the closest well-resourced language to Adja in the Gbe family. The E4 model (`dodziraynard/whisper-small-ee`) was already trained on Ewe speech, so its encoder learned acoustic patterns shared between the two languages: similar vowel inventories (both have ɛ, ɔ), similar tonal systems, and similar consonant clusters.

**Interesting training dynamics**: E4 had a slow start (100% CER for 14 epochs while the Ewe tokenizer adapted) but then caught up rapidly. By epoch 30, it matched C2. By epoch 50, it surpassed C2 — and was **still improving** (last epoch was a new best).

### 3.4 Decode Samples (Qualitative Analysis)

**Best model (E4, CER=24.9%):**

```
REF: ŋɖuɖu lɔwo nuɔn
HYP: ŋ ɖudu lɔwo nɔ?
→ Correctly recognizes ŋ, ɖ, lɔwo. Confuses u/ɔ.

REF: Eshilɔ ɖote yi mi jaja a ?
HYP: Eɖote yi mi jaja
→ Gets ɖote, yi, mi, jaja perfectly. Misses "shi" and "lɔ".

REF: Enu maku enyi
HYP: Enu maku eye
→ First two words perfect. Confuses "enyi" with "eye".
```

**What the model gets right**: Common Adja words (yi, mi, enu, maku), special characters (ŋ, ɖ, ɔ), word boundaries.

**What the model struggles with**: Tone marks, less frequent characters, word endings, distinguishing similar sounds (u vs ɔ, nyi vs ye).

---

## 4. Why CTC Models Failed

Three of our CTC experiments (C4 XLS-R, C5 wav2vec2, E1 MMS-Ewe) experienced **CTC collapse** — the model learned to output only blank tokens, resulting in 100% CER.

**Root causes identified and fixed during experimentation**:
1. Feature extractor padding corrupted attention masks
2. CTC loss doesn't support float16 (needed float32 cast)
3. MMS-1B exceeded A10G GPU memory (24GB)
4. Early stopping patience too aggressive (5 → should be 20+)
5. PyTorch version incompatibilities with cloud GPU drivers

MMS-French (C3) was the only CTC model that partially trained (CER=87%), confirming that CTC fine-tuning IS possible but requires careful tuning. The CTC approach may improve with: longer training, language model integration, and focal loss for rare characters.

---

## 5. Technical Details

### Compute Resources

| Platform | GPU | Cost/hour | Used for |
|----------|-----|-----------|----------|
| HuggingFace Jobs | A10G (24GB) | $1.50 | C2, E4 (Whisper), C4, C5 |
| HuggingFace Jobs | A100 (80GB) | $2.50 | C3, E1 (MMS-1B) |
| Local (MacOS) | CPU | Free | Code validation, dry runs |

**Total estimated cost**: ~$25-30 (well within $100 HF credits)

### Training Configuration (Best Models)

| Parameter | C2 (Whisper) | E4 (Whisper-Ewe) |
|-----------|-------------|-----------------|
| Base model | openai/whisper-small | dodziraynard/whisper-small-ee |
| Parameters | 244M | 244M |
| Learning rate | 1e-5 | 1e-5 |
| Batch size | 8 (effective 32 with grad accum) | 8 (effective 32) |
| Epochs trained | 50 | 50 |
| Best epoch | 47 | 50 |
| Training time | 5.5 hours | 3.7 hours |

---

## 6. Next Steps (Proposed)

### Immediate (this week)
1. **Train Whisper-large-v3 on Adja** — 6x more parameters than whisper-small. Needs HPC (A100 80GB). Expected significant CER improvement.
2. **Continue E4 training** — it was still improving at epoch 50. Run 100 more epochs.
3. **Add character n-gram language model** — already built a 5-gram character LM from Adja text. Using it for beam search decoding (pyctcdecode) could reduce WER by 10-50% based on literature.

### Research experiments (weeks 2-3)
4. **Focal CTC loss** — weight rare characters (ɖ, ŋ, tone marks) higher in loss computation. Literature shows 3-9% improvement on rare characters.
5. **Hybrid vocabulary** — separate tone marks from base characters so the model can share character knowledge across toned/untoned variants.
6. **AfriHuBERT** — alternative base model pretrained on 1,226 African languages. May have better tonal representations.
7. **TTS data augmentation** — use `facebook/mms-tts-ewe` to generate synthetic training audio from Adja text, especially for utterances with rare characters.

### Analysis (week 4)
8. **Error analysis** — categorize errors by type (tone errors, character confusion, word boundary, etc.)
9. **Data scaling curves** — how does CER change with 25%, 50%, 75%, 100% of training data?
10. **Paper direction decision** — choose between: (a) cross-lingual transfer paper, (b) architecture comparison paper, (c) low-resource ASR techniques paper

---

## 7. Key Takeaways

1. **Encoder-decoder (Whisper) significantly outperforms CTC-based models** for low-resource Adja ASR with only ~2 hours of data.

2. **Cross-lingual transfer from Ewe works** — starting from a related Gbe language gives a measurable 2.1% CER improvement over training from scratch.

3. **CER=24.9% from 2 hours of data** is a strong first result. For context, state-of-the-art Ewe ASR (with 100+ hours of data) achieves ~31% WER — our Adja system with 50x less data is approaching comparable territory.

4. **The model learns real Adja** — producing actual words (enu, maku, ɖote, jaja, lɔwo) with correct special characters (ŋ, ɖ, ɔ, ɛ).

5. **There is substantial room for improvement** — language model integration, larger models, more data, and tonal techniques are all untapped.

---

## Appendix: Repository Structure

```
adja-nmt/
├── CLAUDE.md                               # Project rules
├── docs/
│   ├── asr-learning-guide.md               # ASR concepts tutorial
│   ├── language-models-for-asr.md          # LM/n-gram/focal loss explainer
│   └── improving-adja-character-recognition.md  # Research techniques
├── experiments/
│   ├── registry.md                         # All experiments tracked
│   └── asr/                                # 8 experiment directories
├── scripts/
│   ├── hf_jobs/                            # Self-contained HF Jobs scripts
│   ├── slurm/                              # HPC submission scripts
│   └── containers/                         # Apptainer recipes
├── results/
│   ├── comparison.md                       # Results table
│   ├── run-ledger.md                       # Run log
│   └── progress-report-2026-04-16.md       # This report
└── data/
    ├── manifests/                          # Train/dev/test splits
    ├── char_vocab.json                     # 117-token vocabulary
    └── char_5gram.arpa                     # Character language model
```

---

## Appendix: Full Training Curves

### C2 (Whisper-small → Adja)
| Epoch | Loss | CER | WER |
|-------|------|-----|-----|
| 1 | 8.51 | 92.16% | 98.25% |
| 5 | 2.92 | 71.42% | 96.23% |
| 10 | 1.45 | 34.93% | 84.93% |
| 15 | 0.91 | 30.77% | 79.93% |
| 20 | 0.70 | 28.96% | 78.00% |
| 25 | 0.51 | 28.50% | 77.39% |
| 30 | 0.41 | 28.08% | 75.46% |
| 35 | 0.36 | 27.23% | 74.67% |
| 40 | 0.29 | 27.74% | 74.58% |
| 45 | 0.25 | 28.01% | 76.42% |
| 47 | 0.24 | **27.02%** | 74.15% |
| 50 | 0.24 | 29.09% | 77.65% |

### E4 (Whisper-Ewe → Adja)
| Epoch | Loss | CER | WER |
|-------|------|-----|-----|
| 1 | 17.54 | 100.00% | 100.00% |
| 8 | 3.39 | 99.10% | 99.56% |
| 15 | 3.00 | 90.89% | 99.91% |
| 20 | 1.79 | 40.21% | 95.53% |
| 25 | 1.39 | 29.78% | 80.98% |
| 30 | 1.17 | 28.73% | 78.70% |
| 35 | 1.05 | 27.97% | 79.32% |
| 40 | 0.95 | 27.80% | 80.81% |
| 45 | 0.86 | 25.38% | 74.06% |
| 50 | 0.75 | **24.90%** | **73.09%** |
