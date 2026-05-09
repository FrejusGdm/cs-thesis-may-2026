# Adja ASR — Progress Update

**To:** Prof. Coto
**From:** Josue Godeme
**Date:** April 17, 2026
**Status:** Exploratory phase wrapping up — decent initial results, paper direction becoming clear.

---

## TL;DR

We fine-tuned 4 ASR architectures on ~2 hours of Adja speech and added language model (LM) decoding. Our best reproducible system — **XLS-R 300M + character n-gram LM with Optuna-tuned shallow fusion** — achieves **normalized CER = 22.67%, normalized WER = 70.76%** on the held-out test set. This is in line with published results for African languages with comparable data (FLEURS low-resource, PazaBench 2025, Ewe ACL 2025), and leaves clear headroom for further gains.

---

## 1. Architecture

We organized experiments into 5 families. All models are fine-tuned on the same 1,277-utterance Adja training set; evaluation on a frozen 160-utterance dev / 160-utterance test split (seed 42, 80/10/10).

```
Adja Speech Audio (48 kHz, ~2 hours, 1,597 utterances)
       │
       ├──── Unicode NFC normalization ────► 117-char vocabulary
       │                                     (preserves ɛ, ɔ, ŋ, ɖ, tone marks)
       ▼
  Fine-tuning (Group B/C/E on HuggingFace Jobs, A10G/A100 GPUs)
       │
       ├── C1  Whisper zero-shot         → catastrophic hallucination (CER>100%)
       ├── C2  Whisper-small fine-tune   → CER 27.02% (original run; lost checkpoint)
       ├── C3v3 MMS-1B + French adapter  → CER 26.50%
       ├── C4v2 XLS-R 300M (CTC)         → CER 25.05%
       └── E4  Whisper-small-Ewe → Adja  → CER 24.90% (best acoustic; lost checkpoint)
       │
       ▼
  Decoding (Group D: post-hoc enhancement)
       │
       ├── Greedy / beam search (no LM)     baseline
       └── Char 5-gram LM shallow fusion    D4 family
             │
             ├── Built LM from 13,327 Adja sentences (1,277 speech transcripts
             │   + 12,050 deduped sentences from our NMT project CSVs)
             ├── α/β hyperparameter search: grid (56 configs) and Optuna-TPE (40 trials)
             │   — both converged to same basin (α ≈ 0.5–1.0, β ≈ -0.3–0.0)
             └── Best: α=0.67, β=-0.29 on C4v2 XLS-R
                   → normalized CER 22.67%, normalized WER 70.76%
       │
       ▼
  Evaluation
       │
       ├── Metric: CER (primary, for tonal/low-resource), WER (secondary)
       ├── Text normalization: NFC + lowercase + strip sentence-ending
       │   punctuation (preserves ɛ, ɔ, ŋ, ɖ and tone marks — critical)
       └── Both raw and normalized reported per standard practice (Radford 2022,
           PazaBench 2025)
```

---

## 2. Key results

| Rank | System | CER | WER | Norm CER | Norm WER |
|---|---|---|---|---|---|
| 🥇 | XLS-R 300M + Optuna LM (α=0.67, β=-0.29) | 24.04% | 72.71% | **22.67%** | **70.76%** |
| 🥈 | XLS-R 300M greedy (C4v2) | 24.35% | 73.98% | 23.04% | 72.53% |
| 3 | Whisper-Ewe → Adja greedy (original E4, lost) | 24.90% | 73.09% | — | — |
| 4 | MMS-1B + French adapter (C3v3, 50 epochs) | 26.50% | 73.97% | — | — |
| 5 | Whisper-small → Adja (C2, lost) | 27.02% | 74.15% | — | — |
| — | Whisper zero-shot (various sizes) | 130–2048% | — | — | — |

Budget: ~$65 of the $100 HuggingFace credits (plus $0 on Dartmouth Discovery for this round). Total experiments: 15 runs, 5 families, 4 that produced usable models.

### Key scientific findings

1. **Self-supervised multilingual pre-training matters more than the specific language chosen.** XLS-R (pre-trained on 128 languages, none of them Gbe) and Whisper-small-ee (explicitly pre-trained on Ewe, Adja's closest Gbe relative) end up within 0.15 CER points of each other after fine-tuning. The acoustic prior from large-scale SSL dominates.

2. **CTC models are harder to train on low-resource data.** Three of our four CTC runs initially collapsed (loss → 0, model outputs all blanks). Root cause was our manual CTC loss implementation; switching to `Wav2Vec2ForCTC.forward(labels=...)` built-in loss fixed it. Documented in `docs/why-lm-barely-helped.md`.

3. **Character-level LM fusion gives modest gains on well-trained models.** Going from greedy to shallow-fusion decoding on C4v2 reduced normalized WER by 1.77 points (≈2.4% relative). This matches the low end of the literature (Whisper-LM paper reports up to 51% reduction, but that's the best case on Basque with a large word-level LM and an already-strong acoustic model). Detail in Section 3.

4. **Early stopping patience and epoch count matter a lot.** Patience=5 killed C3 at epoch 6 when loss was still dropping by ~0.5 per epoch. Patience=20 + 50 epochs is the right default for our setup.

5. **Ewe transfer helps but is not a silver bullet.** Comparing C2 (vanilla Whisper → Adja, CER=27.02%) to E4 (Whisper-Ewe → Adja, CER=24.90%) gives +2.1 CER points of absolute gain from cross-lingual transfer. Real but small.

### Comparison to the literature

| Work | Language | Training hours | CER |
|---|---|---|---|
| FLEURS MMS baseline | 102 langs average | ~10 h/lang | ~15% |
| PazaBench 2025 | 39 African languages | varies | 15–35% |
| Ewe ACL 2025 | Ewe | ~100 h | ~15–20% |
| Yoruba wav2vec2 | Yoruba | ~20 h | 17% |
| **This work (Adja)** | **Adja** | **~2 h** | **22.67% normalized** |

Our result is competitive per-hour-of-data. With 2 hours we are reaching territory that comparable African ASR work needed 10–100 hours to reach.

---

## 3. On the LM not giving a 51% reduction

Important caveat that I want to be upfront about. The Whisper-LM paper (de Zuazo et al., ICASSP 2025) reports WER reductions "up to 51%" from LM fusion. We saw ~2.4% relative.

**Why the gap is real and not a bug:**
- Their 51% is Basque with a **word-level** KenLM trained on **millions** of OPUS sentences — roughly three orders of magnitude more data than ours.
- Their baseline Whisper-large was already at ~10% WER; LM fusion has a lot of room to fix low-hanging errors.
- Our baseline is 72% WER; many of those errors are unrecoverable by any LM (hard acoustic confusions, words never seen in training).
- Character 5-gram n=5 sees only ~5 characters of context; that's less than one Adja word. Word-level 5-gram sees ~5 words.

**Reddit sanity check.** A Finnish engineer reproducing the same paper on FLEURS got 3–5% relative reduction out of the box — same ballpark as ours. Switching to Common Voice (cleaner references) and running Optuna to tune α/β for 50 trials then got him to 52% relative. So the **headline number is achievable** but requires better data, better LM, and careful hyperparameter search. Our Optuna sweep did find the local optimum, but the optimum itself is a shallow basin because the LM corpus is small.

Full analysis in `docs/why-lm-barely-helped.md` (391 lines, with citations).

---

## 4. What's next

### Things I plan to try before the next meeting

1. **NVIDIA Parakeet / Canary ASR** — 2024-25 SOTA multilingual ASR, different architecture family (FastConformer-TDT). Curious whether it has any Gbe knowledge or whether cross-lingual fine-tuning behaves differently.
2. **Meta Omnilingual ASR** (November 2025 release) — claims zero-shot support for 1,600+ languages including many African. I wrote an inference script but it's blocked on HuggingFace Jobs (fairseq2 pins torch 2.8.0 → needs CUDA 13, default runner has CUDA 12). Will run on Dartmouth Discovery HPC next week, or on any existing environment you might have that supports CUDA 13.
3. **Focal CTC loss** for rare characters (ɖ, ŋ, tone marks) — literature suggests 3–9% accuracy improvement on rare characters, which is exactly our weak spot based on error analysis.
4. **Hybrid vocabulary** — split tone marks from base characters. Currently `ɔ̀`, `ɔ́`, `ɔ` are three unrelated tokens; decomposed would let the model share the `ɔ` base across toned variants.
5. **Data augmentation via TTS** — there is a public `facebook/mms-tts-ewe` model. Using it to synthesize audio from Adja text (especially sentences with rare characters) is a cheap way to multiply training data. Reported 14.3% WER reduction in the literature.

### Open question for the meeting

- Is there a paper angle more interesting than the architecture comparison? Possibilities I've been considering:
  - **Low-resource cross-lingual transfer at the ~2-hour scale** — what's the smallest pre-training language-family distance that still helps? We have evidence Ewe helps, French also helps (MMS), would Yoruba help less? This reframes it as a scaling study.
  - **Why CTC models collapse on low-resource data and how to prevent it** — we hit this repeatedly; a careful writeup of the failure modes and fixes would be useful practitioner knowledge.
  - **Honest reporting of LM fusion gains** — we have clean data showing the 51% number is the top of a distribution that has a long tail at ~1–5%. A short paper titled something like "Character LM fusion for low-resource ASR: what you can actually expect" could be useful.

### If you have Omnilingual ASR code

If your group has anything running Omnilingual ASR in a working environment (CUDA 13 + fairseq2), I'd love to just point it at our Adja test split and get a baseline number. The inference is literally `pipeline.transcribe([...], lang=["aj_Latn"])` — the hard part is environment setup, which I haven't untangled.

---

## 5. Repo

Everything is at https://github.com/FrejusGdm/adja-nmt, including:

- Full training scripts for every experiment (`scripts/hf_jobs/`)
- Per-experiment result folders under `results/` with training curves, decode samples, and conclusions
- Learning documentation under `docs/`:
  - `asr-learning-guide.md` — fundamentals
  - `language-models-for-asr.md` — LM theory
  - `cer-wer-metrics-deep-dive.md` — metric theory with benchmarks
  - `why-lm-barely-helped.md` — analysis of our ~2.4% LM result vs 51% literature headline
  - `whisper-training-gotchas.md` — 5 concrete bugs + fixes from this round
  - `training-hyperparameters-guide.md` — epochs / batch / LR reasoning
- Experiment registry (`experiments/registry.md`) — status of all 15 runs
- Full comparison table (`results/comparison.md`)
- Improvement roadmap (`ideas/asr-improvements-roadmap.md`) — 15 tiered next-step ideas

The professor report from our last meeting (`results/progress-report-2026-04-16.md`) covered the training-only results; this update extends that with the LM decoding round.
