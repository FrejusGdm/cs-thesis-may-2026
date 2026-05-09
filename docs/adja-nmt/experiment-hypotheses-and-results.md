# Experiment Hypotheses, Mechanisms & Results
## Adja TTS/ASR — Living Research Document

**Purpose:** Pre-registered hypotheses for every experiment, updated with actual results.
Feeds directly into thesis Chapter 4 (TTS) and Chapter 5 (ASR) and the paper ablation table.

Last updated: 2026-04-26

---

## The Central Research Question

> **Which component of LLM-based TTS is the binding constraint for intelligible speech
> synthesis in Adja (Gbe family, ~1.6h training data)?**

The candidate bottlenecks, ranked by our prior confidence:

| Bottleneck | Prior confidence | Key experiment |
|------------|-----------------|----------------|
| Catastrophic forgetting (Ewe priors lost in Stage 2) | **High** — confirmed symptom | CF1, CF2, CF3 |
| LLM backbone has no Gbe acoustic prior | **High** | AT1, S6 |
| Codec semantic tokens are English-biased | **Medium** | M0, (implicit in Spark vs CSM comparison) |
| Llama BPE tokenizer fragments Adja characters | **Low** — T1-tokfix weakly refuted | TK1 |
| Insufficient data volume | **Medium** | All experiments (indirect) |

---

## TTS Experiments

---

### CF1 — Elastic Weight Consolidation (EWC) Stage 2
**Status:** ✅ Completed 2026-04-26 | **Instance:** ml.g5.2xlarge (71.5 min)

**Hypothesis:**
Penalizing changes to Stage-1 weights in proportion to their Fisher information (importance
for Ewe generation) should prevent catastrophic forgetting while allowing Adja adaptation.
Weights critical for Ewe phoneme generation are "anchored" near their Stage-1 values.

**Mechanism (Kirkpatrick et al. 2017, PNAS):**
```
L_total = L_adja + (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
```
F_i = Fisher diagonal (importance of weight i for Ewe). θ*_i = Stage-1 anchor.
λ=1000 (from paper). Fisher estimated over 500 Ewe samples.

**Prediction:**
- Lower eval loss than naive Stage 2 baseline (6.5115) ✓
- Partially intelligible Adja output

**Result:**
- best_eval_loss = **6.4941** — BEATS baseline (6.5115) ✓
- Audio verdict (Josue 2026-04-26): **ALL NOISE** — "crispy noise, could hear something in background but not intelligible" ✗

**Interpretation for paper:**
*EWC optimizes the loss metric without improving perceptual quality.* The regularization
forces weights to stay near their Ewe values, but the functional pathways for audio
generation are not preserved — only their mathematical proximity. This decouples loss
and intelligibility: **Stage 2 eval loss is not a reliable predictor of speech quality.**
This is a standalone publishable observation.

EWC may require a much higher λ to be effective (the model found a loss-reducing path
that satisfies the penalty while still producing noise), or the anchor weights need to
be in the audio heads specifically, not the full LoRA parameter set.

---

### CF2 — Curriculum Annealing Stage 2
**Status:** ✅ Completed 2026-04-26 | **Instance:** ml.g5.2xlarge (49.9 min training)

**Hypothesis:**
Gradually shifting the training data ratio from Ewe-heavy (5:1) to Adja-heavy (1:5)
over 20 epochs prevents catastrophic forgetting by keeping the model actively practicing
Gbe phonology generation throughout Stage 2.

**Mechanism (Bengio et al. 2009, ICML — Curriculum Learning):**
Each epoch t uses ratio: Ewe:Adja = (5−0.4t) : (1+0.4t), linearly from 5:1 to 1:5.
The model retains Ewe generation fluency in early epochs while Adja patterns accumulate.

**Data:** Ewe = WaxalNLP ewe_tts (1,215 clips). Adja = adja-tts-orpheus 80/10/10
(~987 train / ~123 dev / ~124 test). NFC normalization on all text.

**Prediction:**
- Worse loss than EWC (Ewe data dilutes Adja gradient signal) ← confirmed
- Better perceptual quality than EWC (active Ewe practice maintains generation)← confirmed

**Result:**
- best_eval_loss = **6.6203** (worse than CF1 6.4941 and baseline 6.5115)
- Still improving at epoch 19 — early stopping never fired
- Audio verdict (Josue 2026-04-26): **1/5 INTELLIGIBLE ADJA** 🎉
  - `CF2_adja_sent02.wav` — "Ŋ nyanyɔ mɔ wo anu ahán !" — confirmed intelligible
  - 4/5 noise
- Natural durations (2.3–7.8s) — contrast with naive Stage 2 where ALL 5 hit 10s cap

**Interpretation for paper:**
*Data-based curriculum beats regularization-based anti-forgetting on perceptual quality,
even when it produces a worse loss metric.* The model needs to PRACTICE Gbe speech
generation, not just have Gbe-adjacent weights. This is the first intelligible Adja
output in any Stage 2 experiment. Key claim: **active rehearsal > passive regularization**
for cross-lingual TTS transfer. The result also motivates longer training (40 epochs —
model was still improving at epoch 19).

**Open questions:**
- Would 40 epochs produce 2/5 or 3/5 intelligible?
- Would a steeper annealing schedule (start more Ewe-heavy) do better?
- Would curriculum + EWC combined beat either alone?

---

### CF3 — Frozen Backbone Stage 2
**Status:** 🔄 Running | **Instance:** ml.g5.2xlarge (~6h)

**Hypothesis:**
Catastrophic forgetting in Stage 2 originates in the Llama backbone's language
representations, not in the audio projection heads. Freezing the backbone entirely
and only training the audio interface layers should preserve Ewe-learned phoneme
representations while adapting to Adja's audio distribution.

**Mechanism:**
All Llama backbone weights frozen (no gradient). Only audio projection layers
(input_projection, codebook_head, the layers mapping between text representations
and Mimi code space) updated during Stage 2.

**Prediction:**
- If intelligible: forgetting is in the backbone → audio heads alone can adapt
- If noise: backbone must be updated → the backbone IS the bottleneck

**Result:**
- **Listening verdict (Josue 2026-04-27): ALL NOISE** — 0/5 intelligible.

**Interpretation for paper:**
*Catastrophic forgetting in Stage 2 lives in the Llama backbone, not the audio interface.*
Freezing the backbone and training only audio projection heads is insufficient — the backbone
must actively update during Stage 2. This directly informs the CF2 result: CF2 works because
it keeps the backbone updating with Ewe data (active practice), not because of anything about
the audio heads.

Combined picture: backbone needs Ewe rehearsal (CF2) > weight proximity (CF1) > frozen (CF3).

**Why this matters for paper:**
This is the architectural ablation that locates the forgetting. If CF3 = intelligible,
the fix is cheap (freeze backbone, fine-tune ~2% of parameters). If CF3 = noise,
the backbone must update and CF2's curriculum approach (or LAPT) becomes the primary recommendation.

---

### TK1 — Expanded Tokenizer Ablation (Ewe Stage 1)
**Status:** ✅ Completed 2026-04-26 | **Instance:** ml.g5.2xlarge

**Hypothesis:**
The T1-tokfix experiment was confounded: it tested tokenizer expansion with direct
Adja training on ~2h of data (insufficient to isolate the variable). TK1 tests
the expanded tokenizer on Ewe Stage 1 with full 1,215 clips — controlling for data
volume. If tokenizer expansion helps, Ewe Stage 1 eval loss should be lower than
standard T1-csm-ewe-stage1 (eval_loss=4.326).

**Mechanism:**
70 Adja/Gbe characters added to Llama vocabulary via `add_tokens()`.
`model.resize_token_embeddings()` expands the embedding matrix.
`config.vocab_size` restored to preserve CSM's audio codebook offset (2051).

**Prediction:**
- If Stage 1 Ewe loss LOWER than 4.326: tokenizer fragmentation IS a meaningful bottleneck
- If Stage 1 Ewe loss SIMILAR to 4.326: tokenizer is NOT the bottleneck (replicate T1-tokfix finding with proper controls)

**Result:**
- WER = **100%** (all word sequences wrong)
- CER = **58.52%** over 160 test samples
- best_eval_loss = 1.8038
- Training: 114.7 min, 30 epochs, vocab=110 Adja chars

**Interpretation:**
Worse than C-small baseline (CER 27.02%). The Ewe adapter provides Gbe-family acoustic
representations, but swapping the lm_head to an Adja vocabulary creates a mismatch: the
adapter's outputs are optimized for Ewe phoneme distributions, not Adja. The model trained
for 30 epochs but CER of 58.52% suggests it learned the character inventory but not the
phoneme-to-character mapping cleanly. WER=100% suggests systematic errors in word boundaries
or substitutions.

Possible fix: freeze the Ewe adapter (don't update acoustic features) and only train lm_head
— test whether the Ewe acoustic priors are correct and only the output layer needs adaptation.

---

### AT1 — CSM Audio-LM Pretraining on Ewe → Stage 1 → Stage 2
**Status:** 🔄 Running ~10h | **Instance:** ml.g5.12xlarge (~28h total)

**Hypothesis:**
Pretraining the Llama backbone on Mimi code sequences from 183k unlabeled Ewe clips
(before any text supervision) gives it a Gbe-family acoustic prior in codec-token space.
When Stage 1 (Ewe TTS) and Stage 2 (Adja TTS) then run on this backbone, the model
already "knows what Gbe speech sounds like" and catastrophic forgetting is reduced.

**Mechanism (AudioLM — Borsos et al. 2022, arXiv:2209.03143):**
Stage A: next-token prediction over Mimi code streams (pure audio-LM, no text).
183k × ~30s clips ≈ 100-200h of Gbe audio.
Stage B: Ewe TTS supervised (same as T1-csm-ewe-stage1).
Stage C: Adja TTS (same as T1-csm-ewe-adja-stage2).

**Prediction:**
- Stage A should converge to loss ~2-4 nats over Mimi codes (well below 8.3 random)
- Stage C (Adja) should produce more intelligible output than CF2 (which starts from text-pretrained Llama)
- The model should show less catastrophic forgetting because the audio token distribution is familiar

**Pending verdict.**

**Why this matters:**
AT1 tests the "acoustic prior" hypothesis — the single most theoretically motivated
intervention. If AT1 > CF2 on intelligibility, the paper's main recommendation is
audio-LM pretraining on related-language data before cross-lingual TTS.

---

### S6 — Orpheus Audio-LM on Ewe SNAC Codes → Ewe+Adja TTS
**Status:** 🔄 Running (resubmitted after snac package fix) | **Instance:** ml.g5.12xlarge (~26h)

**Hypothesis:**
Same as AT1 but with Orpheus (SNAC codec, tonal ZH base) instead of CSM (Mimi).
Tests whether audio-LM pretraining benefit generalizes across codec families.

**Prediction:**
- Similar improvement over naive Stage 2 baseline as AT1
- Confirms pretraining hypothesis is architecture-agnostic

**Pending verdict.**

---

### T3A — Spark Ewe Stage 1 → Adja Stage 2 (Direct)
**Status:** 🔄 Submitted 2026-04-26 | **Instance:** ml.g5.2xlarge (~5h)

**Hypothesis:**
Spark's BiCodec uses XLSR-53 semantic tokens (53 languages, including African).
The Ewe→Adja gap in BiCodec's token space is much smaller than in Mimi (English-biased).
Direct Stage 2 (no curriculum, no EWC) may suffice for Spark where it failed for CSM.

**Mechanism:**
Spark Ewe Stage 1 checkpoint → LoRA r=128 fine-tune on Adja TTS (987 clips, 20 epochs).
No Ewe data in Stage 2 — pure Adja. Compare to T1-csm-ewe-adja-stage2 (noise) and CF2 (1/5 intelligible).

**Prediction:**
- At least 1/5 intelligible (Spark's multilingual prior should match or beat CF2)
- Possibly multiple intelligible sentences (Spark proved it can do direct Adja from 120 steps)
- If this is noise: the Ewe Stage 1 actually HURTS Spark (overwrites direct-Adja priors)

**This experiment answers:** Is Spark's success with direct Adja transferable through
a Gbe-family bridge, or does the Ewe stage interfere with the existing multilingual priors?

---

### T3C — Spark Ewe Stage 1 → Adja Stage 2 (Curriculum Mixed)
**Status:** 🔄 Submitted 2026-04-26 | **Instance:** ml.g5.2xlarge (~6h)

**Hypothesis:**
Applying the CF2 curriculum recipe (3:1→1:3 Ewe:Adja over 20 epochs) to Spark
provides additional anti-forgetting on top of the codec's multilingual prior.
Should produce equal or better results than T3A (direct).

**Prediction:**
- If T3A > T3C: Spark's codec prior already handles the Ewe→Adja transition; curriculum adds overhead
- If T3C > T3A: curriculum is universally beneficial regardless of codec architecture
- Likely: both produce intelligible output; T3C may produce more

---

## ASR Experiments

---

### S7 — Whisper-tiny Direct Adja Fine-tune
**Status:** ✅ Completed 2026-04-26 (14 min) | **Instance:** ml.g5.xlarge

**Hypothesis:**
Whisper-tiny (39M params), despite no cross-lingual transfer and limited capacity,
can learn some Adja recognition from 987 training clips.

**Prediction:**
- CER > 27.02% (worse than C-small baseline) — tiny has less capacity
- Establishes size-ablation baseline (tiny / small / large)

**Result: METRICS PENDING (download from S3)**

---

### S4 — MMS-1B Ewe Adapter → Adja CTC Fine-tune
**Status:** 🔄 Running ~5h | **Instance:** ml.g5.2xlarge

**Hypothesis:**
Meta's MMS-1B includes a pre-trained Ewe language adapter. Loading with
`target_lang="ewe"` gives a model that already produces Gbe-family phoneme
representations. Swapping only the `lm_head` for Adja CTC and fine-tuning
should require minimal steps — the Gbe acoustic model is already built.

**Mechanism:**
MMS architecture: shared encoder + per-language adapter + per-language lm_head.
"Ewe adapter" ≈ 3M parameters encoding Ewe-specific acoustic patterns.
Replace lm_head with Adja character vocab (built from training data).

**Prediction:**
- CER < 27.02% with very few training steps (fast convergence due to Ewe adapter)
- Potentially the most compute-efficient path to good Adja ASR

---

### S1 — Whisper Large-v3 LoRA, Ewe → Adja
**Status:** 🔄 Running ~20h | **Instance:** ml.g5.2xlarge

**Hypothesis:**
Fine-tuning Whisper large-v3 on WaxalNLP Ewe ASR (Stage A) then Adja (Stage B)
with LoRA gives the model Gbe-family phonological patterns before Adja fine-tuning.
Large-v3 has 1.5B params and saw multilingual data in pretraining — a better
starting acoustic model than tiny.

**Prediction:**
- CER < 15% (significantly better than C-small 27.02% and S7)
- Ewe transfer meaningful: Stage B Adja convergence faster and to lower CER than
  direct fine-tune from English-only Whisper large-v3

---

### S2 — XLS-R 1B SSL on Ewe Unlabeled → Adja CTC
**Status:** 🔄 Running ~36h | **Instance:** ml.g5.12xlarge

**Hypothesis:**
Self-supervised pretraining (wav2vec 2.0 contrastive learning) on 183k unlabeled
Ewe clips builds robust Gbe acoustic representations in XLS-R 1B. The CTC fine-tune
on Adja then only needs to learn phoneme-to-character mapping, not acoustic features.

**Mechanism (wav2vec 2.0 — Baevski et al. 2020, arXiv:2006.11477):**
Contrastive objective: predict correct quantized representation among distractors
using context transformer. Stage A: 183k Ewe clips. Stage B: Adja CTC fine-tune.

**Prediction:**
- Better than direct XLS-R fine-tune (no SSL pretraining)
- CER < 27.02% (C-small baseline)
- Confirms SSL on related-language unlabeled data is beneficial for low-resource ASR

---

### S3 — MMS-1B SSL on Ewe Unlabeled → Adja CTC
**Status:** 🔄 Running ~36h | **Instance:** ml.g5.12xlarge

**Hypothesis:**
Same as S2 but MMS-1B was pre-trained on 1,000+ languages (Pratap et al. 2023,
arXiv:2305.13516) including Ewe-family languages. SSL on additional Ewe data
further specializes a model that already has African phonology priors.

**Prediction:**
- Better than S2 (stronger starting prior from 1000-language pretraining)
- Potentially best ASR result — the highest-quality Gbe acoustic model in the experiment suite

---

## Result Summary Table (updated as experiments complete)

| Exp | Track | Hypothesis | Result | Verdict |
|-----|-------|-----------|--------|---------|
| CF1 | TTS | EWC regularization prevents forgetting | loss↑ but perceptual quality ✗ | ❌ Hypothesis partially confirmed (loss improved, intelligibility did not) |
| CF2 | TTS | Curriculum annealing prevents forgetting | 1/5 intelligible Adja 🎉 | ✅ **First breakthrough** |
| CF3 | TTS | Forgetting is in backbone, not audio heads | ✅ Completed | ❌ **ALL NOISE** — backbone must update, frozen heads insufficient |
| TK1 | TTS | Expanded tokenizer helps with Ewe Stage 1 | PENDING | — |
| AT1 | TTS | Audio-LM pretraining on Ewe reduces forgetting | RUNNING | — |
| S6 | TTS | Audio-LM pretraining generalizes to Orpheus | RUNNING | — |
| T3A | TTS | Spark Ewe→Adja direct (multilingual codec) | RUNNING | — |
| T3C | TTS | Spark Ewe→Adja curriculum | RUNNING | — |
| S7 | ASR | Whisper-tiny baseline | COMPLETED (CER pending) | — |
| S4 | ASR | MMS Ewe adapter → cheap Adja ASR | ✅ Completed | ⚠️ CER=58.52%, WER=100% — worse than C-small (27.02%) |
| S1 | ASR | Whisper large-v3 LoRA Ewe→Adja | RUNNING | — |
| S2 | ASR | XLS-R 1B SSL Ewe → Adja CTC | RUNNING | — |
| S3 | ASR | MMS-1B SSL Ewe → Adja CTC | RUNNING | — |

---

## Key Emerging Findings

1. **Loss ≠ intelligibility in Stage 2.** CF1 has lower loss than CF2 but worse audio quality. The eval loss for cross-lingual TTS transfer does not predict perceptual quality. Use native-speaker listening tests AND (once S1 completes) automated CER via the Adja ASR model.

2. **Active rehearsal > passive regularization.** CF2 (keep practicing Ewe) beats CF1 (penalize weight drift). The model needs to generate Gbe speech during Stage 2 training, not just have Gbe-adjacent weights.

3. **Natural duration is a necessary but not sufficient condition.** All 8 completed experiments with natural sample durations still produced mostly noise. Natural duration (not hitting 10s cap) is a prerequisite for intelligibility, not a guarantee.

4. **The codec prior matters most.** Spark (XLSR-53 BiCodec) produced intelligible Adja from 120 steps of DIRECT training. CSM (Mimi, English-heavy) needed a Gbe bridge AND curriculum and still gets 1/5. The T3A/T3C experiments will quantify this difference.

---

## Automated Evaluation Plan (once S1 completes)

Use Whisper large-v3 Adja ASR (S1 output) as automatic intelligibility scorer:
```
TTS_output.wav → S1 ASR → transcription → CER vs source_text
```
This gives a continuous intelligibility metric replacing binary listening verdicts.
Run on ALL generated samples from CF1/CF2/CF3/AT1/T3A/T3C/S6.
Expected correlation with listening verdicts: high if S1 CER < 30%.

---

## Thesis Chapter Mapping

| Chapter | Experiments | Key claims |
|---------|-------------|-----------|
| Ch4.1 Codec analysis | M0 (pending), Spark vs CSM comparison | Codec semantic token bias explains intelligibility gap |
| Ch4.2 Catastrophic forgetting | CF1, CF2, CF3 | Curriculum > EWC; active rehearsal > regularization |
| Ch4.3 Acoustic pretraining | AT1, S6 | AudioLM pretraining on related language reduces Stage 2 forgetting |
| Ch4.4 Tokenizer | TK1 | Tokenizer is not the primary bottleneck (confirm/refute) |
| Ch4.5 Architecture comparison | T3A/T3C vs CF2 | Codec choice matters more than training recipe |
| Ch5 ASR | S1, S2, S3, S4, S7 | Cross-lingual transfer via Gbe bridge for Adja ASR |
