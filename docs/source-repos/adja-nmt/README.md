# Adja ASR: Systematic Speech Recognition for a Low-Resource Tonal Language

Exploration repo for building automatic speech recognition (ASR) systems for **Adja** — a tonal language in the Gbe family spoken in Benin and Togo. Adja has very limited digital resources, making it a compelling testbed for low-resource speech technology.

---

## 📚 THE DOCS ARE IMPORTANT — READ THEM BEFORE TOUCHING ANYTHING

If you're picking this repo up again after a break (future-me, collaborators, contributors from open source): **everything non-obvious about this project is in `docs/`**. Do not rediscover bugs we already solved. Do not retrain models without re-reading the gotchas. Every day I didn't re-read these I wasted hours.

**Start here** (in order, based on what you need):

| If you want to… | Read this first |
|---|---|
| Understand the project | this README, then `results/comparison.md` |
| Fine-tune Whisper | **`docs/whisper-training-gotchas.md`** ← real bugs + fixes, especially §0 (EOS masking) |
| Fine-tune wav2vec2 / XLS-R / MMS with CTC | `scripts/hf_jobs/ctc_finetune.py` + `docs/training-hyperparameters-guide.md` |
| Build or use the character LM | `docs/language-models-for-asr.md` + `docs/why-lm-barely-helped.md` |
| Understand CER vs WER + normalization | `docs/cer-wer-metrics-deep-dive.md` + `docs/normalization-and-lm-explained.md` |
| Plan the next experiment | `ideas/asr-improvements-roadmap.md` |
| Write the paper | `results/comparison.md` + `results/progress-report-*.md` + the YAMLs under `experiments/asr/*/config.yaml` |

**The five most painful bugs from this session (reread before touching the trainers):**

1. **Whisper EOS masking** — `pad_token_id == eos_token_id` in Whisper's tokenizer. If you mask by token equality you erase the real EOS and the model never learns to stop → infinite hallucination loops at inference. Use the `attention_mask` instead. (`docs/whisper-training-gotchas.md` §0)
2. **CTC collapse** — manual CTC loss computation produced loss=0 and all-blank output. Switch to `Wav2Vec2ForCTC.forward(labels=...)` which does it correctly internally.
3. **Whisper 3000-frame requirement** — `transformers >= 4.49` requires exactly 3000 mel frames. `padding=True` no longer auto-pads. Pad manually with `torch.nn.functional.pad`.
4. **Save the full processor** — not just `model.save_pretrained()`. Without `feature_extractor.save_pretrained()` and `ctc_vocab.json`, the checkpoint is useless for LM decoding later.
5. **Patience=5 is too aggressive** for low-resource fine-tuning. Use 20. Original C3 run died at epoch 6 while loss was still plummeting.

**Current best result**: XLS-R 300M + Optuna-tuned character n-gram LM → **normalized CER=22.67%, normalized WER=70.76%** on ~2 hours of Adja training data.

---

## Why This Matters

Most speech technology is built for high-resource languages (English, Mandarin, Spanish). For the ~600,000+ Adja speakers, there are no existing ASR systems, voice assistants, or speech-to-text tools. This project systematically explores what works — from classical approaches to state-of-the-art pre-trained models — to find the best path forward.

Adja is particularly interesting because it is **tonal**: pitch changes word meaning, and most ASR systems weren't designed with this in mind.

## What We're Comparing

We compare ASR approaches across baseline, fine-tuning, transfer, and omnilingual settings, all evaluated on the same Adja speech dataset:

### Group B: From-Scratch Models
| ID | Model | What It Tests |
|----|-------|--------------|
| B1 | BiLSTM + CTC | Simplest neural baseline. Is the data clean enough to learn from scratch? |
| B2 | Conformer + CTC | Does attention + convolution help over pure RNNs? |
| B3 | Transformer + CTC | Is it the convolution in Conformer that helps, or just attention? |
| B4 | Listen Attend Spell | Attention-based decoding vs CTC — which paradigm wins? |

### Group C: Pre-Trained Fine-Tuning
| ID | Model | What It Tests |
|----|-------|--------------|
| C1 | Whisper (zero-shot) | What does a massive model know about Adja with no adaptation? |
| C2 | Whisper (fine-tuned) | How much does Adja-specific data improve Whisper? |
| C3 | MMS (French adapter) | Meta's 1100-language model — best under 1h of data per benchmarks |
| C4 | XLS-R | 128-language self-supervised model with CTC head |
| C5 | wav2vec 2.0 | English-only SSL — does multilingual pretraining actually help? |

### Group D: Newer Models
| ID | Model | What It Tests |
|----|-------|--------------|
| D1 | W2v-BERT 2.0 | Data-efficient model from the SeamlessM4T stack |
| D2 | Moonshine | Tiny model (27M params) — can small models work for Adja? |
| D3 | Distil-Whisper | Efficiency vs quality tradeoff |
| D4 | Whisper + n-gram LM | Does a simple language model on top help? |

### Group E: Cross-Lingual Transfer (Ewe → Adja)
| ID | Model | What It Tests |
|----|-------|--------------|
| E1 | MMS + Ewe adapter | Does starting from the closest Gbe language beat French? |
| E4 | Whisper-Ewe → Adja | Does an Ewe-tuned Whisper transfer to Adja? |

### Group F: Omnilingual ASR (Meta OmniASR)
| ID | Model | What It Tests |
|----|-------|--------------|
| Omni_ZS | omniASR_LLM_300M (zero-shot) | Can OmniASR transcribe Adja without any fine-tuning? |
| Omni_ICL | omniASR_LLM_7B_ZS (in-context) | Does few-shot prompting/context improve zero-shot decoding? |

Latest Omni zero-shot result (`Omni_ZS`, HF Job `69e37788`):
- Dev: CER `145.22%`, WER `129.32%`
- Test: CER `139.49%`, WER `133.87%`
- Output path: `JosueG/adja-asr-results/Omni_ZS/`
- Interpretation: Adja remains strongly out-of-distribution in zero-shot OmniASR.

## Dataset

~1,600 utterances of Adja speech with transcriptions. Private dataset on HuggingFace (`JosueG/adja-tts-orpheus`).

The orthography uses special characters: ɛ, ɔ, ŋ, ɖ, and tone marks (é, è, etc.). All processing preserves these — we never normalize them away.

## Project Structure

```
├── CLAUDE.md                           # Project rules and context
├── docs/
│   └── asr-learning-guide.md           # ASR concepts explained for NMT researchers
├── experiments/
│   ├── registry.md                     # Master experiment tracker
│   ├── asr/
│   │   ├── shared/                     # Data prep, metrics, normalization
│   │   ├── B1_bilstm_ctc/             # From-scratch BiLSTM + CTC
│   │   ├── C1_whisper_zeroshot/        # Whisper inference (no training)
│   │   ├── C2_whisper_finetune/        # Whisper fine-tuning
│   │   ├── C3_mms_finetune/           # MMS with French adapter
│   │   ├── C4_xlsr_finetune/          # XLS-R fine-tuning
│   │   ├── C5_wav2vec2_finetune/      # wav2vec 2.0 (English-only control)
│   │   ├── E1_mms_ewe_finetune/       # MMS with Ewe adapter
│   │   └── E4_whisper_ewe_finetune/   # Whisper-Ewe → Adja transfer
│   └── tts/
│       ├── T1_sesame_csm_finetune/    # Working vanilla HF baseline
│       ├── T2_orpheus_finetune/       # Orpheus 3B vanilla PEFT path
│       ├── T6_mms_tts_ewe_finetune/   # Ewe -> Adja VITS transfer
│       ├── T7_voxcpm_finetune/        # VoxCPM2 multilingual tokenizer-free track
│       ├── T8_ims_toucan_finetune/    # IMS-Toucan low-resource multilingual track
│       ├── T9_xtts_v2_finetune/       # XTTS-v2 practical small-data baseline
│       └── T10_f5_e2_tts_finetune/    # F5/E2 flow-matching track
├── scripts/
│   ├── containers/                     # Apptainer recipes for HPC
│   ├── slurm/                          # SLURM job templates
│   ├── download_models.sh              # Pre-download model weights
│   └── validate_before_submit.sh       # Pre-flight checks
├── results/
│   ├── run-ledger.md                   # Chronological run log
│   └── comparison.md                   # Results comparison table
├── data/
│   └── manifests/                      # Train/dev/test splits (TSV)
└── learnings-from-the-past/            # HPC and training gotchas
```

## Running Experiments

### 1. Prepare Data

```bash
python experiments/asr/shared/data_prep.py \
    --output-dir data --hf-token $HF_TOKEN --dry-run
```

### 2. Build Containers (on HPC)

```bash
cd scripts/containers
bash build_container.sh
```

### 3. Download Model Weights (on HPC)

```bash
bash scripts/download_models.sh
```

### 4. Validate Locally

```bash
bash scripts/validate_before_submit.sh B1
```

### 5. Submit to HPC

```bash
bash scripts/slurm/submit.sh B1
```

Every experiment has a `--dry-run` flag that runs 2 training steps on 5 samples, so you can validate locally before waiting in the HPC queue.

## TTS expansion map

The repo now includes explicit runbooks and HF Jobs launchers for the requested
multilingual TTS families:

- `experiments/tts/T7_voxcpm_finetune/README.md`
- `experiments/tts/T8_ims_toucan_finetune/README.md`
- `experiments/tts/T9_xtts_v2_finetune/README.md`
- `experiments/tts/T10_f5_e2_tts_finetune/README.md`
- shared prioritization doc: `docs/multilingual-speech-strategy-2026-04-18.md`

Canonical HF launcher scripts:

- `scripts/hf_jobs/T7_voxcpm_finetune.py`
- `scripts/hf_jobs/T8_ims_toucan_finetune.py`
- `scripts/hf_jobs/T9_xtts_v2_finetune.py`
- `scripts/hf_jobs/T10_f5_e2_tts_finetune.py`

## Compute

- **Primary**: Dartmouth Discovery HPC (SLURM, A100 80GB GPUs, Apptainer containers)
- **Secondary**: HuggingFace Jobs for lighter experiments

## Key References

- [CTC (Graves et al. 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — The alignment method used by most of our models
- [Whisper (Radford et al. 2022)](https://cdn.openai.com/papers/whisper.pdf) — 680K hours weakly supervised ASR
- [MMS (Pratap et al. 2023)](https://jmlr.org/papers/v25/23-1318.html) — 1100+ language ASR/TTS
- [wav2vec 2.0 (Baevski et al. 2020)](https://arxiv.org/abs/2006.11477) — Self-supervised speech representations
- [XLS-R (Babu et al. 2022)](https://arxiv.org/abs/2111.09296) — Cross-lingual speech representations
- [Conformer (Gulati et al. 2020)](https://arxiv.org/abs/2005.08100) — Convolution-augmented transformer
- [SpecAugment (Park et al. 2019)](https://arxiv.org/abs/1904.08779) — Data augmentation for speech
- [PazaBench (2025)](https://arxiv.org/html/2512.10968v1) — African language ASR benchmark
- [Ewe ASR Dataset (ACL 2025)](https://aclanthology.org/2025.icnlsp-1.32.pdf) — Closest Gbe relative

## Related Work

This builds on my previous NMT research for Adja, where I showed that ~4K systematically-structured sentences + 10K random sentences achieve BLEU ~20, while 10K random sentences alone achieve BLEU 2-3. The structured curriculum approach drove the improvement. Now I'm exploring whether similar principles apply to speech.

## License

Research use. Dataset is private.
