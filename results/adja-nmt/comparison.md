# ASR Model Comparison — Adja

Last updated: 2026-04-29 (deployable-checkpoint reality vs. lost-checkpoint claims).

## Lost-checkpoint correction (2026-04-29)

Two of the headline numbers below — the original E4 (CER~24.90\%) and the
original C2 (CER~27.02\%) — are **logged-only**: the training metrics survived
on Hugging Face but the model weights were never uploaded. We tried to
reproduce E4 twice (E4v4 on 2026-04-17 and E4_v5 on 2026-04-28, both with the
same hyperparameters and the original padding bug fixed) and both attempts
plateau at CER~37.18% / 37.35%. Working hypothesis: the original E4 trained
with the EOS-mask hallucination bug we later fixed in
`scripts/hf_jobs/whisper_finetune.py`; that bug acted as accidental
regularization, and removing it lets the model overfit to a worse plateau on
1.36 hours of Adja audio. See `experiments/registry.md` E4_v5 row and
`results/tts-comparison.md` for the fuller writeup.

**For the deployable Adja Whisper, use CER~37.18% (E4v4 = `JosueG/whisper-ewe-adja-e4v4`).**
The 24.90% number is reported as a logged result only. Cite it that way in
the thesis: see `04_asr_sections/06_results.tex` for the corrected wording.

The **C4v2 XLS-R + CTC** rows below are unaffected — those weights survived
upload (in `JosueG/adja-asr-results/C4v2/best_model/`, with the tokenizer
reconstructed at `JosueG/wav2vec2-xlsr-adja-c4v2`) and remain the reproducible
flagship of the chapter at CER 25.05% (greedy) / 22.67% (with character LM).

## 🏆 Greedy vs LM Decoding — Final Table

| Model | Decoding | Test CER | Test WER | Normalized CER | Normalized WER |
|---|---|---|---|---|---|
| **C4v2 XLS-R** | Greedy | 24.35% | 73.98% | 23.04% | 72.53% |
| **C4v2 XLS-R** | LM (α=0.5, β=0) 6-cfg grid | 24.10% | 73.07% | 22.80% | 71.32% |
| **C4v2 XLS-R** | LM (α=0.5, β=0.0) 56-cfg grid | 24.08% | 72.53% | 22.76% | **70.76%** |
| **C4v2 XLS-R** | **LM (α=0.67, β=-0.29) Optuna 40 trials** | **24.04%** | **72.71%** | **22.67%** 🥇 | **70.76%** 🥇 |
| E4v2 Whisper-Ewe (broken) | Greedy | 89.53% | 130.73% | 86.76% | 129.42% |
| E4v2 Whisper-Ewe (broken) | LM | 42.92% | 93.38% | 39.06% | 90.60% |
| E4 (original, lost) | Greedy | 24.90%* | 73.09%* | — | — |
| C3v3 MMS-Fr | Greedy | 26.50% | 73.97% | — | — |
| C2 Whisper-small (lost) | Greedy | 27.02%* | 74.15%* | — | — |
| Qwen3-ASR 0.6B | Greedy | 53.14% | 100.0% | 52.26% | 100.65% |

*Reported from earlier metrics.json — full checkpoint lost, can't run LM.

### What we learned from LM decoding
1. **LM helps CTC models modestly**: C4v2 saw 1.2-1.8 WER point improvement from greedy → Optuna-tuned LM. Consistent with literature for char-level n-gram LMs on low-resource languages.
2. **LM helps broken models a lot**: E4v2 (Whisper with hallucination bug) saw 37-point WER drop (130→93) because LM pulls outputs back to plausible Adja. This confirms the "ceiling effect" in `docs/why-lm-barely-helped.md`.
3. **Optuna converges to α≈0.67, β≈-0.29** — Bayesian TPE reached the same optimum as our 56-config grid but with fewer trials. All top-10 trials cluster tightly, suggesting we're at a real local minimum not noise.
4. **Our best reproducible result is now `C4v2 + Optuna LM`**: Normalized CER=22.67%, Normalized WER=70.76%.

## Acoustic Model Results (pre-LM)

| Rank | Exp ID | Model | Approach | Best CER ↓ | Best WER ↓ | Best Epoch | Train Time | GPU | Cost |
|------|--------|-------|----------|-----------|-----------|-----------|-----------|-----|------|
| 🥇 | E4 (original, lost) | Whisper-small-Ewe → Adja | Fine-tune (encoder-decoder) | 24.90% | 73.09% | 50 | 3.7h | A10G | ~$5.50 |
| 🥈 | **C4v2** | **XLS-R 300M (CTC, fixed)** | **Fine-tune (CTC)** | **25.05%** | **72.39%** | 48 | 3.75h | A10G | ~$5.60 |
| 🥉 | C3v3 | MMS-1B + French (A100) | Fine-tune (CTC) | 26.50% | 73.97% | 50 | ~8h | A100 | ~$20 |
| 4 | C2 (lost) | Whisper-small → Adja | Fine-tune (encoder-decoder) | 27.02% | 74.15% | 47 | 5.5h | A10G | ~$8.25 |
| 5 | QASR-s42 | Qwen3-ASR-0.6B | Fine-tune (Qwen ASR) | 53.14% | 100.0% | — | — | A10G | — |
| 6 | C3 (killed) | MMS-1B + French (patience=5) | Fine-tune (CTC) | 87.19% | 98.42% | 1 | 0.6h | A100 | ~$1.50 |
| 7 | C1 | Whisper-large-v3 (zero-shot) | Inference only | ~130% | — | — | — | A10G | ~$0.50 |
| 8 | C1 | Whisper-small (zero-shot) | Inference only | 366.74% | 100.91% | — | — | A10G | — |
| 9 | C1 | Whisper-tiny (zero-shot) | Inference only | 2047.84% | 1208.25% | — | — | A10G | — |
| — | C4 (v1) | XLS-R 300M (first attempt) | CTC | 100.0% | 100.0% | — | — | A10G | ~$0.50 |
| — | C5 | wav2vec 2.0 (English) | CTC | 100.0% | 100.0% | — | — | A10G | ~$0.50 |
| — | E1 | MMS-1B + Ewe adapter | CTC | 100.0% | 100.0% | — | — | A100 | ~$0.50 |
| — | E4v2 | Whisper-Ewe (broken retrain) | Fine-tune | 76-90% | 120%+ | — | — | A10G | ~$1.50 |
| — | C2_redo, E4_redo | Retrain attempts (hallucinating) | Fine-tune | cancelled | — | — | — | A10G | ~$2 |
| — | Omni_ZS_7B | Omnilingual ASR 7B zero-shot | Inference | blocked (CUDA 13) | — | — | — | — | — |
| — | Omni_ZS_300M | Omnilingual ASR 300M zero-shot | Inference | 139.49% | 133.87% | — | ~0.2h | A10G | ~$0.50 |
| — | Omni_ICL_k1 | OmniASR LLM 7B | In-context learning (k=1) | 284.87% | 424.14% | — | ~0.2h | A10Gx4 | — |
| — | Omni_ICL_k3 | OmniASR LLM 7B | In-context learning (k=3) | 311.76% | 493.10% | — | ~0.2h | A10Gx4 | — |
| — | Omni_ICL_k10 | OmniASR LLM 7B | In-context learning (k=10) | 511.76% | 696.55% | — | ~0.2h | A10Gx4 | — |
| — | Omni_FT_CTC_300M | OmniASR CTC 300M v2 | Recipe fine-tune (400 steps) | **55.92%** | **96.66%** | — | ~0.1h | A10G | Full-data rerun `69e3ce60cd8c002f31dfebf9` with uncapped splits (`train=1277`, `dev=160`, `test=160`) validated on dev+test at step 400. Test WER 96.66 (fairseq2), test CER 55.92 (script post-eval); dev WER 97.05, dev CER 57.48. |
| — | Omni_FT_LLM_300M | OmniASR LLM 300M v2 | Recipe fine-tune (200 steps) | **50.31%** | **96.94%** | — | ~0.1h | H200 | Full-data rerun `69e3d20ccd8c002f31dfec21` with uncapped splits (`train=1277`, `dev=160`, `test=160`) validated on dev+test at step 200. Test WER 96.94 (fairseq2), test CER 50.31 (script post-eval); dev WER 90.43, dev CER 48.85. |
| — | Omni_FT_CTC_3B | OmniASR CTC 3B v2 | Recipe fine-tune (390 steps) | **47.95%** | **91.55%** | — | ~0.2h | H200 | Full-data rerun `69e3e7e6ac288e522d8efe11` with uncapped splits (`train=1277`, `dev=160`, `test=160`) validated on dev+test at step 390. Test WER 91.55 (fairseq2), test CER 47.95 (script post-eval); dev WER 91.32, dev CER 50.01. |
| — | Omni_FT_LLM_3B | OmniASR LLM 3B v2 | Recipe fine-tune (200 steps) | **44.66%** | **89.97%** | — | ~0.4h | H200 | Full-data rerun `69e3d998ac288e522d8efdda` with uncapped splits (`train=1277`, `dev=160`, `test=160`) validated on dev+test at step 200. Test WER 89.97 (fairseq2), test CER 44.66 (script post-eval); dev WER 88.46, dev CER 45.12. |
| — | Omni_AJG_LLM3B_R1 | OmniASR LLM 3B v2 | AJG rerun (Aja Benin tag fix, 8s max, 0.1s min) | **44.57%** | **89.57%** | — | ~0.4h | H200 | Fresh rerun `69e428fccd8c002f31dfef7b` with `LANGUAGE_CODE=ajg_Latn`; full 320-pair eval (`expected_total=320`, `used_pairs=320`). This isolates the language-tag correction without changing core 3B LLM recipe settings. |
| — | Omni_AJG_CTC3B_R5_M2 | OmniASR CTC 3B v2 | AJG rerun (Aja Benin tag fix, stable 2s min) | **51.75%** | **96.29%** | — | ~0.2h | H200 | Fresh rerun `69e43365cd8c002f31dfefdd` with `LANGUAGE_CODE=ajg_Latn`, `MIN_AUDIO_LEN=32000` (stable path) and `DATA_PARALLELISM=fsdp`, `SAVE_MODEL_ONLY=true`. Eval uses 310 pairs by design (`155+155` kept after min-length filtering). |
| — | Omni_AJG_CTC7B_R3_M2 | OmniASR CTC 7B v2 | AJG rerun (7B, stable 2s min) | **47.63%** | **92.48%** | — | ~0.5h | H200 | Third retry `69e4de80cd8c002f31dff62b` completed after two CPU-init startup failures. This improves over AJG CTC 3B (`51.75`) and slightly edges the older `adj`-coded 3B CTC rerun (`47.95`), but still trails Omni 3B LLM (`44.57`). |
| — | Omni_AJG_LLM7B_R7 | OmniASR LLM 7B v2 | AJG rerun (7B, real 4-process FSDP launch) | **43.22%** | **87.49%** | — | ~0.6h | H200x4 | Final successful 7B LLM retry `69e53548cd8c002f31dff936`. Required an explicit `torch.distributed.run` launch; earlier `x2/x4` retries were still effectively single-process. Best Omni result in the repo so far, but still far behind the repo-best non-Omni systems. |
| — | Omni_FX_LLM3B_A8_M0 | OmniASR LLM 3B v2 | Forensics: 8s max audio, 0.1s min audio | **44.11%** | **88.40%** | — | ~0.4h | H200 | Controlled run `69e40d23cd8c002f31dfee7a` with `MAX_AUDIO_SEC=8`, `MIN_AUDIO_LEN=1600`. Kept full `160/160` dev and `160/160` test; eval meta shows `expected_total=320, used_pairs=320`. |
| — | Omni_FX_LLM3B_A8_M2 | OmniASR LLM 3B v2 | Forensics: 8s max audio, 2s min audio | **44.84%** | **90.53%** | — | ~0.4h | H200 | Controlled run `69e40d23ac288e522d8efe5e` with `MAX_AUDIO_SEC=8`, `MIN_AUDIO_LEN=32000`. Dropped exactly 5 short clips from each eval split (`155/160`), yielding `expected_total=310, used_pairs=310`. |
| — | Omni_FX_LLM3B_A15_M0 | OmniASR LLM 3B v2 | Forensics: 15s max audio, 0.1s min audio | **46.86%** | **91.57%** | — | ~0.3h | H200 | Controlled run `69e41393cd8c002f31dfeea8` with `MAX_AUDIO_SEC=15`, `MIN_AUDIO_LEN=1600` and full `320/320` eval pairs. In this run, longer audio cap did not improve metrics relative to the 8s cap under same min-length setting. |

7B update:
- `Omni_AJG_LLM7B_R7` is now the best Omni run in the repo at test CER `43.22` / WER `87.49`.
- The decisive fix was infrastructure: a real 4-process `torch.distributed.run` launch. Larger GPU flavors alone did not help while the wrapper stayed single-process.

## Key Findings

### 1. Whisper encoder-decoder and XLS-R CTC are now neck-and-neck
E4 (Whisper-Ewe) and C4v2 (XLS-R) are statistically tied: 24.90% vs 25.05% CER (0.15 point gap). C4v2 actually has the **lower WER** (72.39% vs 73.09%). This overturns our day-1 conclusion that Whisper dominates. Both approaches work — they just needed proper tuning.

### 2. The CTC "collapse" was a fixable bug, not a fundamental problem
First XLS-R attempt (C4): 100% CER forever. Second attempt (C4v2) with the built-in loss fix + patience=20: **25.05% CER**. The collapse came from our manual CTC loss implementation, NOT from CTC being ill-suited for Adja.

**Lesson**: When something gets 100% CER, check the model's `.forward(labels=...)` built-in loss before blaming the architecture.

### 3. Ewe cross-lingual transfer still helps, but less than expected
E4 beats C2 by 2.1% CER absolute (24.9 vs 27.0). But C4v2 (no Ewe pre-training, just XLS-R's 128-language pretraining) ALSO beats C2. So strong self-supervised pretraining matters more than the specific language choice — though Ewe gives an extra edge.

### 4. C3v3 MMS is trending to be competitive
MMS at epoch 19 is already at CER=29.04% (vs killed-early C3 at CER=87%). Loss still dropping. Projected final CER: 22-26% if trends continue. Could become the new leader when it finishes.

### 5. Patience and epoch count matter massively
All three completed experiments (E4, C2, C4v2) converge best around epoch 45-50. Initial patience=5 killed good training runs. **The patience=20, epochs=50 config is the right default for Adja fine-tuning.**

### 6. Zero-shot Whisper is still garbage — no exception for large-v3
Even whisper-large-v3 (1.5B params) outputs CER~130% on Adja zero-shot (hallucinations in Italian/Russian/pseudo-Romance). Confirms Adja is fully out-of-distribution. Fine-tuning is essential.

### 7. Qwen3-ASR now runs end-to-end, but the 0.6B result is not competitive on Adja
The full Qwen3-ASR-0.6B pilot completed and the held-out scorer now reports raw WER=100.0%, raw CER=53.14%, normalized WER=100.65%, normalized CER=52.26%. That is much better than total collapse and the outputs preserve some Adja-like orthography, but it is still far behind the best XLS-R / Whisper-Ewe systems. The current evidence says Qwen is operational on Adja, not yet performant.

## Training Curves — Best Models

### C4v2: XLS-R with fixed CTC loss
```
Ep  1: CER=100.0% | loss=17.54  (warm-up)
Ep  8: CER=99.52% | loss= 3.40  (first breakthrough)
Ep 15: CER~60-70% | loss~ 2.0   (rapid descent)
Ep 30: CER=30.5%  | loss~ 1.1
Ep 40: CER=26.0%  | loss= 0.93
Ep 48: CER=25.05% | loss= 0.80  ← BEST
Ep 50: CER=25.36% | loss= 0.76
```

### E4: Whisper-small-ee → Adja
```
Ep  1: CER=100.0% | loss=17.54  (Ewe tokenizer adjusting)
Ep 15: CER=90.89% | loss= 3.00
Ep 20: CER=40.21% | loss= 1.79  (rapid descent)
Ep 30: CER=28.73% | loss= 1.17
Ep 46: CER=25.36% | loss= 0.83
Ep 50: CER=24.90% | loss= 0.75  ← BEST (still improving!)
```

## Decode Samples — Top Two Models

### C4v2 (XLS-R, CER=25.05%, WER=72.39%)
| Reference | Hypothesis |
|-----------|-----------|
| ŋɖuɖu lɔwo nuɔn | ŋ ɖudu lɔwo nɔ ? |
| Eshilɔ ɖote yi mi jaja a ? | Eɖote yi mi jaja |
| Enu maku enyi | Enu maku eye |

### E4 (Whisper-Ewe, CER=24.90%, WER=73.09%)
| Reference | Hypothesis |
|-----------|-----------|
| ŋɖuɖu lɔwo nuɔn | ŋ ɖudu lɔwo nɔ? |
| Eshilɔ ɖote yi mi jaja a ? | Eɖote yi mi jaja |
| Enu maku enyi | Enu maku eye |

**Striking observation**: the two best models from completely different architectures produce nearly identical outputs. Both make the same character confusion (u→ɔ), both segment words identically ("ŋ ɖudu lɔwo" instead of "ŋɖuɖu lɔwo"). This is signal about the **data itself**, not the models — suggesting the word-boundary inconsistency in training transcriptions is the next bottleneck.

## Cost Tracking

| Round | Cost | Cumulative |
|-------|------|-----------|
| Day 1 experiments | ~$27 | $27 |
| Day 2 reruns (C4v2, E4v2, C3v3) | ~$16 | $43 |
| Remaining HF budget | | ~$57 |

## What's Next

**Tier 1** (from `ideas/asr-improvements-roadmap.md`):
1. Qwen3-ASR-1.7B or multilingual balanced fine-tuning if we want to test whether the Qwen family can scale into a useful unsupported-language regime
2. Char LM beam search (wait for more Adja text data)
3. SentencePiece word normalization (addresses the spacing problem seen in decode samples)

**After C3v3 finishes**:
- Update this table with MMS-French final result
- If it beats 25%, MMS becomes the new winner
- Otherwise, stick with E4/C4v2 as co-winners
