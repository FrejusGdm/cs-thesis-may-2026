# Adja Speech & NMT — Master Experiment Plan

Everything worth exploring, across all tracks, for the thesis and any papers that emerge from it.
This is a living document. Update status after every run. Add new ideas freely.

---

## Epistemology Policy — Read This First

**Failure explanations in this repo are hypotheses, not facts.**

When an experiment produces bad audio or high WER, we often write down a reason. That reason is
our best current guess, based on what we know at the time. It may be wrong. Example: we initially
blamed the Mimi and SNAC audio codecs for CSM and Orpheus failures on Adja. We later found that
SNAC handles Mandarin (a tonal language) fine in the Orpheus `3b-zh` variant — suggesting the real
bottleneck may have been the Llama 3.2 BPE text tokenizer fragmenting Adja diacritics. But we
haven't proven the tokenizer is the cause either. Both are hypotheses.

**Rules for documenting failures:**
- Prefix causal claims with "Hypothesis:" or "Possible cause:"
- If you have experimental evidence (e.g., ablation, reconstruction test), say so explicitly
- If you have only a theory, say so explicitly
- The word "because" in a failure note implies confirmation — use "may be because" unless confirmed

This matters for the thesis: reviewers will ask "how do you know?" — make sure the log can answer.

---

## Track 1: TTS — Text-to-Speech for Adja

### 1A — Completed Experiments

| ID | Model | Status | Result | Hypothesis for outcome |
|----|-------|--------|--------|------------------------|
| T1-csm-vanilla | CSM 1B (direct Adja FT, LoRA r=32) | completed | Unintelligible audio | Hypothesis: Llama 3.2 BPE tokenizer byte-fragments Adja diacritics; LM cannot learn phoneme→audio mapping with 1.7h of data |
| T1-csm-r128 | CSM 1B (direct Adja FT, LoRA r=128) | completed | Unintelligible; dev_loss ≈ r=32 | Hypothesis: capacity is not the bottleneck; same tokenizer issue |
| T1-csm-fullft | CSM 1B (direct Adja full FT) | completed | Dev_loss 6.496 ≈ LoRA r=32 (6.488) | Hypothesis: confirms capacity is not bottleneck; structural issue with tokenizer or codec representation |
| T2-orpheus-r32/64/128 | Orpheus 3B EN (LoRA, 3 ranks) | completed | Unintelligible audio | Same tokenizer hypothesis; English SNAC base has no Gbe-family prior |
| T2-orpheus-fullft | Orpheus 3B EN (full FT) | completed | Unintelligible; best dev_loss 5.42 | Hypothesis: same as above; French base (T2-fr) lost to EN base, suggesting it's phonological not geographic proximity that matters |
| T2-orpheus-fr-r64 | Orpheus 3B FR base (LoRA r=64) | completed | Unintelligible | Hypothesis: French proximity is sociolinguistic; Adja's Gbe family is phonologically distant from both EN and FR |
| T3-spark | Spark TTS 0.5B (full FT, direct Adja) | completed ✓ | **INTELLIGIBLE Adja — native speaker confirmed (2026-04-18)** | Working hypothesis: Qwen2 tokenizer handles Adja chars natively + XLSR-53 BiCodec covers African phonology |
| T5-qwen3-1p7b | Qwen3-TTS 1.7B (direct Adja FT) | completed | Loss 9.09→4.74 over 630 steps; audio verdict pending | — |
| T6-mms-ewe | MMS-TTS-Ewe (VITS) → Adja | running | — | Hypothesis: Ewe is Adja's closest relative with a public TTS model; character-level tokenizer has no OOV issue |
| T7-voxcpm | VoxCPM2 | running | — | — |
| T8-ims-toucan | IMS-Toucan multilingual | running/blocked | Repeatedly blocked by runtime asset fetches | Hypothesis: Transphone G2P fetch race; prefetch now patched |
| T9-xtts-v2 | XTTS-v2 GPT fine-tune | failed/retry | BeamSearchScorer API drift | Hypothesis: Transformers version mismatch in Coqui fork |
| T10-f5 (full-FT) | F5-TTS flow-matching (full FT, direct Adja 1234 clips) | completed ✓ | **Pure noise (listening test 2026-04-22)** | Direct-Adja data volume is insufficient regardless of architecture |
| T11-e2 (full-FT) | E2-TTS flow-matching (full FT, direct Adja 1234 clips) | completed ✓ | **Pure noise (listening test 2026-04-22)** | Same as T10 |
| T1-tokfix | CSM 1B + expanded Llama tokenizer (35 Adja chars) → direct Adja | completed ✓ | **Speech-like noise, rare intelligible fragments (listening test 2026-04-22)** | **Retires the primary-tokenizer-bottleneck hypothesis.** Fixing the tokenizer without adding Gbe-family data only marginally improves noise quality. Data volume dominates. |
| T2-tokfix | Orpheus 3B + expanded tokenizer → direct Adja | completed, no audio | Run completed but 0 wav files in output | Generation path didn't execute — needs investigation |
| T1-ewe-stage1 | CSM 1B + LoRA r=32, WaxalNLP Ewe TTS (1215 clips) | completed ✓ | **INTELLIGIBLE Ewe — native speaker confirmed (2026-04-22)** | **Confirms Gbe-family transfer works for CSM.** Architecture is capable; prior direct-Adja failures were data-bound. |
| T2-en-ewe-stage1 | Orpheus 3B EN + LoRA r=64, WaxalNLP Ewe TTS | completed ✓ | eval_loss ~0.16 @ ep 14; audio deferred to Stage 2 | Checkpoint available for Stage 2 |
| T2-fr-ewe-stage1 | Orpheus 3B FR + LoRA r=64, WaxalNLP Ewe TTS | completed ✓ | eval_loss ~0.16 @ ep 14; audio deferred to Stage 2 | Checkpoint available for Stage 2 |
| T3-spark-ewe-stage1 | Spark TTS 0.5B + LoRA r=128, WaxalNLP Ewe TTS | failing (infra) | Hit CVE + triton JIT + Python.h + libcuda in succession | Latest retry `69e8dd0bd2fd2eb837d76959` pending |

---

### 1B — Simple Fine-tunes (Just Try It)

These are the "just run it and see" experiments. No architectural changes, no fancy pipelines.
Do them first — they're cheap and set the baseline for everything more complex.

#### T1-ewe — CSM fine-tuned on Ewe, then Adja

**Hypothesis**: CSM has never seen a Gbe-family language. Fine-tuning on 1.5k Ewe TTS pairs
(WaxalNLP `ewe_tts` split) before Adja may give the Llama backbone a Gbe-family acoustic prior.
Not enough to fix the tokenizer issue, but may reveal whether more data helps at all.

- **Stage 1**: Fine-tune CSM 1B on WaxalNLP `ewe_tts` (train: 1,215 rows, val: 152, test: 152)
- **Stage 2**: Fine-tune the Stage 1 checkpoint on Adja (1,234 train)
- **Data needed**: `google/WaxalNLP`, config `ewe_tts`, resample to 24kHz
- **Compute**: ~30 min × 2 stages on L40S
- **Baselines**: T1-csm-vanilla (direct Adja, no Ewe)
- **What to compare**: dev_loss and audio quality at each stage
- **Status (2026-04-22)**: **Stage 1 completed and validated — produces intelligible Ewe**. Stage 2 ready to launch via `scripts/hf_jobs/T1_csm_ewe_adja_stage2.py`. Stage 1 checkpoint at `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1`. Hypothesis partially confirmed: Gbe-family data unblocks CSM's ability to produce coherent Gbe speech. Stage 2 tests whether that transfers to Adja.

#### T2-zh-direct — Orpheus `3b-zh` base → direct Adja fine-tune

**Hypothesis**: The Chinese pretrained checkpoint (tonal language, same SNAC codec) gives the LM
a tonal acoustic prior. May help even without Ewe bridge. The tokenizer issue remains — Llama 3.2
BPE may still fragment Adja chars — but tonal priors could help the model map tone diacritics to
correct F0 once it figures out the character mapping.

- **Base model**: `canopylabs/3b-zh-ft-research_release`
- **Fine-tune**: Adja (1,234 train), same config as T2-orpheus-r64
- **Compare against**: T2-orpheus-en (same rank, EN base)
- **Status**: planned

#### T2-zh-ewe — Orpheus `3b-zh` → Ewe fine-tune → Adja fine-tune

**Hypothesis**: Tonal LM prior (zh) + Gbe-family acoustic data (Ewe) + Adja adaptation. Three-stage
cascade. Most promising Orpheus path.

- **Stage 1**: Start from `3b-zh`, fine-tune on WaxalNLP `ewe_tts` (1,215 rows)
- **Stage 2**: Fine-tune Stage 1 checkpoint on Adja (1,234 train)
- **Status**: planned

#### T2-ewe-only — Orpheus EN → Ewe fine-tune → Adja fine-tune

**Hypothesis**: Same as T2-zh-ewe but from EN base. Lets us isolate the contribution of the zh
tonal prior by comparing against T2-zh-ewe.

- **Stage 1**: `orpheus-3b-0.1-pretrained` (EN), fine-tune on WaxalNLP `ewe_tts`
- **Stage 2**: Fine-tune on Adja
- **Status**: planned

#### T3-ewe — Spark TTS → Ewe fine-tune → Adja fine-tune

**Hypothesis**: Spark already works on Adja directly (T3). Fine-tuning on Ewe first (related Gbe
language with 8× more TTS data) should further improve Adja quality. Most grounded cascade because
all three layers (tokenizer, LM, codec) are already multilingual in Spark.

- **Stage 1**: Spark 0.5B → WaxalNLP `ewe_tts` (1,215 rows)
- **Stage 2**: Stage 1 checkpoint → Adja (1,234 train)
- **Compare against**: T3-spark (direct Adja, no Ewe)
- **Status**: planned
- **Priority**: HIGH — Spark is our only working model

---

### 1C — Tokenizer Surgery

These experiments test the tokenizer hypothesis directly. If adding Adja/Ewe characters to the
Llama vocab fixes the failure, that confirms the tokenizer was the bottleneck.

#### T1-tokfix — CSM + tokenizer expansion → direct Adja

**Status (2026-04-22): COMPLETED — hypothesis refuted.** Audio is speech-like noise with rare intelligible fragments; marginally better than T1-vanilla but still not intelligible. Tokenizer expansion alone is insufficient; data volume / Gbe-family prior is the dominant bottleneck. Results: [T1_csm_tokfix](https://huggingface.co/JosueG/adja-tts-results/tree/main/T1_csm_tokfix).

**Hypothesis**: Llama 3.2 BPE fragments Adja diacritics (ɛ, ɔ, ŋ, ɖ, è, é, ɔ̀, ɛ́, etc.) into
byte fallback tokens. Adding these as native tokens + resizing embeddings should unblock the LM's
ability to learn text→audio mapping.

**Diagnostic first** (do this locally before training):
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(tok.tokenize("ɛnyi wɛ ajɔ è ɔ ŋ ɖ"))
# If you see multi-byte fragments (▁Ã, ©, etc.) → tokenizer expansion needed
# If you see whole tokens → tokenizer is fine, codec or data volume is the issue
```

**Approach**:
```python
new_chars = ['ɛ', 'ɔ', 'ŋ', 'ɖ', 'ɛ̀', 'ɛ́', 'ɔ̀', 'ɔ́', 'ɛ̃', 'ɔ̃']  # enumerate all OOVs
tok.add_tokens(new_chars)
model.resize_token_embeddings(len(tok))
# Initialize new embeddings from mean of existing embeddings (better than random)
```

- **Status**: planned
- **Priority**: HIGH — cheapest way to test the tokenizer hypothesis

#### T2-tokfix — Orpheus EN + tokenizer expansion → direct Adja

Same idea as T1-tokfix but for Orpheus EN base.

- **Status**: planned

#### T2-zh-tokfix — Orpheus `3b-zh` + tokenizer expansion → direct Adja

Combines tonal prior + tokenizer fix. Should be strongest Orpheus variant if both hypotheses hold.

- **Status**: planned

---

### 1D — Audio LM Pretraining on Unlabeled Ewe

These experiments test whether pretraining the LM backbone on unlabeled Ewe audio tokens (no text
needed) improves TTS quality. 183k unlabeled Ewe rows = potentially hundreds of hours of audio.

**Key architecture insight**: the audio codec (Mimi, SNAC) is frozen during all TTS training. The
LM backbone learns to predict audio tokens. Training on unlabeled audio means: encode audio → codec
tokens, then train LM on next-token prediction over those tokens. No text required.

#### T_audiolm_csm — Mimi token pretraining on unlabeled Ewe → CSM fine-tune on Adja

**Hypothesis**: 183k rows of Ewe audio (possibly 100–300h) encoded as Mimi tokens teaches the CSM
backbone Gbe-family acoustic patterns (prosody, tone F0 contours, phonotactics) before it sees
Adja text-audio pairs.

**Prerequisite**: Run Mimi reconstruction test on Adja audio locally (encode → decode → listen).
If reconstruction quality is poor (tonal distinctions lost), this track is deprioritized.

```python
from moshi.models import loaders
mimi = loaders.get_mimi(mimi_weight, device='cpu')
codes = mimi.encode(adja_wav)   # [B, 32, T]
recon = mimi.decode(codes)      # listen to recon — does it preserve tones?
```

**Training approach**:
```python
model.train()
model.codec_model.eval()  # codec stays frozen
# Feed Mimi token sequences as labels, predict next token
# Use CsmForConditionalGeneration.forward() with labels= set to audio tokens
```

- **Data**: WaxalNLP `ewe_asr` unlabeled split (183,920 rows)
- **Status**: planned — do reconstruction test first

#### T_audiolm_orpheus — SNAC token pretraining on unlabeled Ewe → Orpheus fine-tune on Adja

Same idea via Orpheus/SNAC. SNAC confirmed language-agnostic (Mandarin `3b-zh` works).

```python
from snac import SNAC
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
codes = snac.encode(ewe_wav)  # list of 3 tensors: [B, T/8], [B, T/4], [B, T/2]
# Apply 7-token interleaved offset encoding, build input_ids sequences
# Train Orpheus LM on audio-only sequences
```

- **Data**: WaxalNLP `ewe_asr` unlabeled split (183,920 rows)
- **Status**: planned

---

### 1E — Other Models (Unexplored)

#### T-llasa-1b and T-llasa-3b — Llasa TTS on Adja

Llama-based TTS (HKUSTAudio). Uses EnCodec. May have different tokenizer properties.
- **HuggingFace**: `HKUSTAudio/Llasa-1B`, `HKUSTAudio/Llasa-3B`
- **Status**: planned

#### T-oute-1b — OuteTTS 1B on Adja

Voice cloning focused model. Interesting for zero-shot inference without training.
- **HuggingFace**: `OuteAI/OuteTTS-1.0-1B`
- **Status**: planned

#### T-inworld — Inworld TTS on Adja

48kHz output (matches our raw data). 11 languages, emotion control, training code included.
- **Repo**: https://github.com/inworld-ai/tts
- **Status**: planned (docs research done in T4)

#### T-nllb-tokenizer — CSM/Orpheus with NLLB SentencePiece tokenizer

**Hypothesis**: NLLB SentencePiece is trained on 200 languages including African ones. Swapping the
Llama BPE tokenizer for NLLB SP could fully resolve the tokenizer bottleneck without adding tokens
individually. More radical than `add_tokens()` but may work better.

- **Reference**: `ideas/nllb-tokenizer-for-tts.md`, `ideas/nllb-tokenizer-for-tts-reading-list.md`
- **Status**: planned (research track)

---

### 1F — Validation Experiments

These are not new models — they validate whether our best checkpoints actually produce good Adja.

#### T-mimi-reconstruction — Does Mimi faithfully encode Adja phonology?

30-minute local test. Encode 20 Adja utterances → decode → listen. Compare reconstructed audio to
original. Pay attention to tone preservation (é vs è on the same vowel) and special consonants (ɖ).

- **Expected outcome**: If reconstruction is clean → codec is fine, tokenizer was the bottleneck.
  If reconstruction smears tones → codec needs evaluation.
- **Status**: planned — do this before T_audiolm_csm

#### T-ewe-quality — Does Ewe fine-tuning produce intelligible Ewe speech?

After T1-ewe and T2-zh-ewe, have someone assess Ewe audio quality before proceeding to Adja.
Ewe has more speakers; easier to evaluate. If the model can't produce intelligible Ewe, it won't
help with Adja.

---

## Track 2: ASR — Automatic Speech Recognition for Adja

### 2A — Completed Experiments (Summary)

| Best results so far | WER | CER | Deployable? |
|---|---|---|---|
| E4 Whisper-Ewe → Adja (logged-only) | 73.09% | 24.90% | **No** — weights never uploaded |
| **E4v4** Whisper-Ewe → Adja (reproduction, EOS-mask fix) | 83.61% | **37.18%** | Yes — `JosueG/whisper-ewe-adja-e4v4` |
| D4_C4v2_lm_optuna (XLS-R + LM) | 70.76% | 22.67% | Greedy CTC only on serverless: see `JosueG/wav2vec2-xlsr-adja-c4v2` |
| C2 Whisper-small direct Adja (logged-only) | 74.15% | 27.02% | **No** — weights never uploaded |
| C3v3 MMS + French adapter | 73.97% | 26.50% | (weights present but no published artifact) |
| Omni 7B LLM (best Omni) | 87.49% | 43.22% | — |

> **Deployable correction (2026-04-29):** rows marked logged-only have
> training metrics but no checkpoint on the Hub. Two reproduction attempts
> for E4 plateau at CER ~37%. See `results/comparison.md` for the full
> writeup.

Full table in `experiments/registry.md`. Note: E4 is still improving at ep 50 — best model
may not be final.

### 2B — Active / Running

| ID | Model | Status |
|----|-------|--------|
| D4-ctc-lm | C4v2 + char LM (pyctcdecode, α/β sweep) | queued |
| D4-whisper-lm | Whisper-Ewe (E4) + char LM shallow fusion | queued |

### 2C — Simple Fine-tunes Not Yet Run

#### E5 — MMS-1B + Ewe adapter → Adja

MMS (Meta Massively Multilingual Speech) was trained on 1000+ languages. E1 crashed due to Ewe
adapter vocab mismatch. This is a retry with the mismatch fixed.

**Hypothesis**: MMS Ewe adapter is the closest pretrained acoustic model to Adja we have. Adapting
from Ewe instead of French (C3) should give lower WER.

- **Fix needed**: align Ewe adapter vocab with Adja character set before fine-tuning
- **Status**: planned

#### E6 — Whisper large-v3 → direct Adja

Current best Whisper experiments (C2, E4) used `small`. Large-v3 has 4× more parameters and was
trained on 680k hours including many low-resource languages.

**Hypothesis**: More parameters + better pretraining data → lower WER, especially for tonal features.

- **HuggingFace**: `openai/whisper-large-v3`
- **Compute**: ~3× more than C2; likely fits on A100 80GB
- **Status**: planned

#### E7 — Whisper large-v3 → Ewe → Adja

Same as E4 (Whisper-small → Ewe → Adja) but with large-v3. Ewe has 15k labeled ASR rows in
WaxalNLP — substantial training data for a 3-stage curriculum.

- **Stage 1**: Whisper large-v3 → WaxalNLP `ewe_asr` labeled (15k rows)
- **Stage 2**: Stage 1 checkpoint → Adja (1,234 train)
- **Status**: planned
- **Priority**: HIGH — current best model (E4 with Whisper-small) may have headroom with larger model

#### E8 — SeamlessM4T v2 ASR → Adja

SeamlessM4T v2 is a unified model covering ASR, ST, MT, and TTS for 200+ languages. Adja is
not in its language list but it includes Ewe and Fon (Gbe relatives).

**Hypothesis**: SeamlessM4T's multilingual audio encoder has broader coverage than Whisper small.
Fine-tuning the ASR component on Adja may benefit from its Ewe/Fon priors.

- **HuggingFace**: `facebook/seamless-m4t-v2-large`
- **Status**: planned (C6 is in registry as "planned" — this is the same)

---

### 2D — Pretraining Track (SSL on Unlabeled Ewe)

These experiments test whether self-supervised pretraining on 183k unlabeled Ewe rows (potentially
100–300h) helps ASR on Adja. Well-grounded in the SSL ASR literature (wav2vec 2.0, MMS, XLS-R).

#### P1 — MMS wav2vec2 SSL continue-pretraining on unlabeled Ewe → fine-tune on labeled Ewe → Adja

**Hypothesis (from MMS paper)**: the 3-stage curriculum (unlabeled pretraining → labeled target
fine-tune → low-resource adaptation) is the established recipe for low-resource ASR. Ewe and Adja
share Gbe-family phonology. 183k unlabeled rows is substantial.

**Pipeline**:
1. Start from MMS wav2vec2 checkpoint (already multilingual)
2. Continue self-supervised masked frame prediction on 183k unlabeled Ewe audio
3. Fine-tune with CTC on 15k labeled Ewe ASR pairs
4. Adapt with CTC on 1,234 Adja pairs

**References**:
- wav2vec 2.0: https://arxiv.org/abs/2006.11477 (Baevski et al., 2020) — establishes SSL→FT recipe
- MMS: https://jmlr.org/papers/v25/23-1318.html (Pratap et al., 2023) — demonstrates exactly this
  3-stage cascade for low-resource languages

- **Status**: planned
- **Priority**: HIGH — strongest scientific grounding in the literature

#### P2 — XLS-R 300M continue-pretraining on unlabeled Ewe → fine-tune on Adja

Same as P1 but using XLS-R 300M (which is what C4/D4 already fine-tuned). Lets us directly compare
the SSL-pretrained checkpoint against the standard C4v2 result.

**Reference**: XLS-R: https://arxiv.org/abs/2111.09296 (Babu et al., 2022)

- **Status**: planned

#### P3 — Whisper continue-pretraining / knowledge distillation on Ewe audio

Whisper is encoder-decoder, so standard SSL (mask+predict) is harder to apply directly. Options:
- **Pseudo-labeling**: run existing Whisper-Ewe (E4) on 183k unlabeled Ewe → noisy transcripts →
  treat as additional supervised data for Stage 2 of E7
- **Encoder-only SSL**: extract Whisper encoder, apply masked feature prediction, then reattach decoder

**Hypothesis**: Pseudo-labeled Ewe data (noisy but abundant) helps if combined with clean data.
This is knowledge distillation / noisy student training.

- **Reference**: Noisy student training for ASR: Park et al. (2020)
- **Status**: planned (lower priority than P1/P2 — more complex for uncertain gain)

---

### 2E — Data Augmentation and Post-processing

#### A1 — Speed perturbation + SpecAugment for Adja ASR

Standard augmentation for low-resource ASR. Not yet applied to any Adja experiments.

- **Speed perturbation**: rates 0.9, 1.0, 1.1 → 3× data
- **SpecAugment**: mask frequency + time bands
- **Apply to**: C2, C4v2, E4 (retrain with augmentation)
- **Status**: planned

#### A2 — Language Model rescoring with larger Adja LM

D4 experiments used a 13k-sentence Adja LM. Explore bigger LMs built from web-scraped Adja text
(if any exists) or text generated by LLMs conditioned on known Adja patterns.

- **Status**: planned

---

## Track 3: Speech Translation (S2TT)

### 3A — Cascade (ASR → MT)

#### ST1 — Best ASR model → NLLB MT → French

Use E4 or E7 (once available) as the ASR component. Feed Adja transcripts into NLLB for
French translation. Baseline cascade system.

- **Status**: planned

#### ST2 — ASR + LLM MT (in-context)

Use Gemini / GPT-4o with few-shot Adja→French examples to translate ASR output. Compare against
NLLB fine-tune.

- **Status**: planned

### 3B — Direct Speech Translation

#### ST3 — SeamlessM4T v2 S2TT → French (zero-shot)

SeamlessM4T supports direct speech→text translation for many languages. Adja is not in the
official list but zero-shot inference on Ewe (which is supported) as a proxy is interesting.

- **Status**: planned

#### ST4 — SeamlessM4T v2 S2TT fine-tune on Adja→French

Fine-tune SeamlessM4T's speech translation component on Adja audio paired with French translations
(if we can get gold translations for the ASR dataset).

- **Status**: planned (data preparation needed)

---

## Track 4: MT / In-Context Learning

Prior work from neurosymbolic paper established NLLB fine-tune + structured data as the main
approach. See `learnings-from-the-past/key-findings.md` for full results.

### Remaining MT Experiments

#### MT1 — NLLB + WaxalNLP Ewe-French pairs as auxiliary data

WaxalNLP Ewe ASR data has Ewe text. If we can find or generate Ewe→French translations, we can
use them as auxiliary data for Ewe→Adja transfer in the MT model (pivot translation).

- **Status**: planned (data availability unclear)

#### MT2 — Gemini / GPT-4o in-context learning for Adja→French

Feed Adja→French example pairs as context. Test how much context is needed for reliable translation.
Compare against fine-tuned NLLB.

- **Status**: planned

#### MT3 — Linguistic pattern extraction via LLM prompting

Feed Adja sentences to Gemini/GPT-4o and ask it to infer grammar rules, tonal patterns, verb
conjugation. Build a structured description of Adja grammar from the ~1.6k training sentences.

- **Status**: planned (interesting for the linguistics track in the thesis)

---

## Track 5: Full Pipeline

End-to-end spoken language understanding: speech → ASR → MT → (optionally TTS)

#### PIPE1 — Full Adja→French pipeline

- ASR: best model from Track 2
- MT: best model from Track 4
- Evaluate end-to-end BLEU on Adja speech with gold French translations

#### PIPE2 — Adja→French→Adja TTS pipeline (round-trip)

Translate speech to French text, then back-synthesize to Adja using TTS. Useful for:
- Data augmentation (generate more Adja audio from translated text)
- Language learning tool

---

## Priority Matrix

**Revised 2026-04-22 after Gbe-family Stage 1 results landed.** Gbe-bridge confirmed for CSM (Stage 1 Ewe is intelligible); direct-Adja with tokenizer expansion refuted as sufficient. Stage 2 is the clear P0.

| Priority | Experiment | Rationale |
|---|---|---|
| P0 (do now) | **T1-ewe Stage 2 (CSM-Ewe → Adja)** | **Stage 1 produces intelligible Ewe. Direct test of whether Gbe prior transfers to Adja.** Script ready: `T1_csm_ewe_adja_stage2.py`. |
| P0 (do now) | T2-en-ewe Stage 2 (Orpheus-EN-Ewe → Adja) | Stage 1 checkpoint available; tests the same Gbe-bridge hypothesis on a different architecture + codec family (SNAC vs Mimi). |
| P0 (do now) | T2-fr-ewe Stage 2 (Orpheus-FR-Ewe → Adja) | Paired comparison with T2-en-ewe Stage 2; tests whether FR sociolinguistic proximity helps on top of Ewe. |
| P1 (this week) | T3-ewe Stage 2 (Spark-Ewe → Adja) | Pending Stage 1 (infra blocker 2026-04-22). Highest-probability improvement over direct-Adja Spark baseline. |
| P1 (this week) | T2-zh-ewe Stage 1+2 (Orpheus ZH → Ewe → Adja) | Needs resubmit with the PEFT grad fix. Tonal LM prior + Gbe bridge is the most theoretically grounded Orpheus path. |
| P1 (this week) | E7 (Whisper large-v3 → Ewe → Adja) | Current best ASR is Whisper-small; headroom likely exists. |
| P1 (this week) | P1 (MMS SSL → Ewe → Adja) | Best-grounded ASR pretraining experiment. |
| P2 (next week) | P2 (XLS-R SSL pretraining on Ewe) | Ablation against P1. |
| P2 (next week) | T-mimi-reconstruction | Now lower priority — CSM-Ewe already producing intelligible Ewe strongly implies Mimi preserves enough Gbe phonology. Only run if Stage 2 Adja output regresses unexpectedly. |
| P3 (whenever) | T_audiolm_csm / T_audiolm_orpheus | Most experimental; 183k unlabeled Ewe rows for SSL audio LM. |
| P3 (whenever) | T-llasa, T-oute, T-inworld | Breadth sweep; low-cost to try. |
| P3 (whenever) | ST3, ST4 (speech translation) | Need good ASR first. |
| ❌ retired | T1-tokfix / T2-tokfix | Completed 2026-04-22; hypothesis refuted by listening test. Tokenizer expansion alone does not unblock Adja TTS. |
| ❌ retired | F5 / E2 direct Adja full-FT | Completed 2026-04-22; produced pure noise. Direct-Adja single-architecture path is data-bound regardless of architecture. |

---

## Dependency Map

```
Mimi reconstruction test → T_audiolm_csm
Tokenizer diagnostic  → T1-tokfix → T2-tokfix → T2-zh-tokfix

T3-spark (done) → T3-ewe

T2-zh-direct → T2-zh-ewe (compare: does Ewe stage help?)
T2-ewe-only → T2-zh-ewe (compare: does zh prior help?)

P1 (MMS SSL on Ewe) → E5 continuation on Adja
WaxalNLP ewe_tts download → T1-ewe, T2-zh-ewe, T3-ewe
WaxalNLP ewe_asr labeled → E7, P1, P2
WaxalNLP ewe_asr unlabeled → T_audiolm_csm, T_audiolm_orpheus, P1, P2, P3

E7 (Whisper large-v3 Ewe→Adja) → ST1, ST2 (best ASR needed for pipeline)
Best TTS model → PIPE2 (round-trip pipeline)
```

---

## Data Inventory

| Dataset | Location | Size | Used for |
|---------|----------|------|----------|
| Adja TTS/ASR | `JosueG/adja-tts-orpheus` (HF, private) | ~1.6k utterances, ~1.7h | All Adja experiments |
| WaxalNLP Ewe ASR labeled | `google/WaxalNLP`, config `ewe_asr` | 15,054 train + 1,916 val + 1,891 test | E5, E7, P1, P2 |
| WaxalNLP Ewe ASR unlabeled | `google/WaxalNLP`, config `ewe_asr` | 183,920 rows | Audio LM pretraining, SSL pretraining |
| WaxalNLP Ewe TTS | `google/WaxalNLP`, config `ewe_tts` | 1,215 train + 152 val + 152 test | T1-ewe, T2-zh-ewe, T3-ewe |

**Note on WaxalNLP audio**: stored as HF Audio features (array + sampling_rate). Resample:
- To 24kHz for CSM (Mimi), Orpheus (SNAC), Spark (BiCodec)
- To 16kHz for Whisper, MMS, XLS-R
- Original 48kHz may be needed for Inworld TTS

---

## Open Questions

1. **Does Mimi faithfully reconstruct Adja tones?** → T-mimi-reconstruction
2. **Is Llama BPE fragmentation the primary bottleneck for CSM/Orpheus?** → T1-tokfix
3. **Does Ewe pretraining help Adja TTS, or does only the architecture matter?** → T1-ewe vs T3-ewe
4. **Does tonal LM prior (zh base) help Adja, or is Gbe-family data more important?**
   → T2-zh-direct vs T2-ewe-only vs T2-zh-ewe
5. **Does SSL pretraining on 183k unlabeled Ewe improve Adja ASR beyond labeled Ewe alone?**
   → P1 ablation (with and without Stage 1)
6. **What is the practical data threshold for CSM/Orpheus on a genuinely new language?**
   Orpheus docs say "50 examples" for a new *speaker* — a new *language* likely needs orders of
   magnitude more. The Mandarin variant used 100k+ hours equivalent. We have 1.7h.
