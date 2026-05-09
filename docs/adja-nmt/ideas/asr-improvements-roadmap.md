# ASR Improvements Roadmap for Adja

Ideas to try next, ordered by effort vs impact. Not yet implemented.

---

## Tier 1 — Low Effort, High Impact (Do First)

### 1. Normalized WER Computation
**What**: Apply Whisper-style text normalization to both references and hypotheses before computing WER. Strips punctuation, lowercases, collapses spaces — but preserves Adja special chars (ɛ, ɔ, ŋ, ɖ) and tone marks.

**Why**: Adja orthography has inconsistent word boundaries. The model often gets characters right but splits words differently. Current WER penalizes this as total error. Normalized WER would drop 10-20 points without retraining.

**Reference**: [Whisper's BasicTextNormalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py)

**How**: Add a `normalize_for_wer()` function in `experiments/asr/shared/metrics.py`, apply before computing WER. Recompute metrics on existing decode outputs pushed to `JosueG/adja-asr-results`.

**Effort**: 1-2 hours. No retraining needed.

---

### 2. Character Language Model Beam Search (D4)
**What**: Use the 5-gram character LM we already built (`data/char_5gram.arpa`) with `pyctcdecode` for CTC beam search decoding of C3/C4 models, OR shallow fusion with Whisper.

**Why**: The Whisper-LM paper (2025) showed up to **51% WER reduction** with LM integration. Our LM is trained and sitting there unused.

**Reference**: [Whisper-LM (de Zuazo et al., 2025)](https://arxiv.org/abs/2503.23542), [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)

**How**: Add a `decode_with_lm.py` script that loads a trained model + the ARPA LM, runs beam search with alpha/beta hyperparameter tuning on dev set.

**Effort**: 4-6 hours. No retraining.

**Status**: Josue has more Adja text coming — retrain LM first when that arrives.

---

### 3. Word Boundary Normalization in Training Data
**What**: Use SentencePiece or a simple rule-based tokenizer to enforce consistent word boundaries in both training transcriptions and eval references before computing metrics.

**Why**: Looking at `data/manifests/train.tsv` — Adja text has inconsistent spacing:
- `kpanŋkɔ` as one word
- `ŋ ɖuɖu` split into two
- `pleŋupleŋu` vs `pipan` (reduplication sometimes split, sometimes not)

This is linguistic variation in the data, not model error. Fix once, apply everywhere.

**How**:
1. Train SentencePiece word model on all Adja text: `spm.SentencePieceTrainer.train(input='adja_transcripts.txt', model_type='word', vocab_size=200)`
2. Re-segment all train/dev/test transcripts consistently
3. Retrain and re-evaluate — both CER and WER should improve

**Effort**: 1 day. Requires retraining.

---

## Tier 2 — Medium Effort, Proven Wins

### 4. Continue E4 from Checkpoint (Not From Scratch)
**What**: E4 was still improving at epoch 50 (last epoch = new best). Load `JosueG/adja-asr-results/E4/best_model` and continue for 50 more epochs.

**Why**: Original E4: CER=24.9% at ep 50, trajectory was downward. Extrapolation suggests ~20% CER possible.

**Risk**: Our E4v2 attempt (from scratch restart with new transformers) hallucinated loops. Loading the saved checkpoint avoids this failure mode entirely.

**How**: Modify `whisper_finetune.py` to accept `-e RESUME_FROM=JosueG/adja-asr-results/E4/best_model`, load weights before training.

**Effort**: 2-3 hours script work + 4-6 hours training on a10g-large (~$9).

---

### 5. Whisper-large-v3 Fine-Tune
**What**: Fine-tune 1.5B-parameter Whisper-large-v3 (6x bigger than Whisper-small used in C2/E4).

**Why**: Larger models consistently outperform smaller ones on low-resource languages once data scales past 1-2 hours. Expected CER drop: 5-8 points.

**Constraints**: Needs A100 80GB. HF Jobs `a100-large` at $2.50/hr. Expect 6-8 hour run → ~$20.

**How**: Same `whisper_finetune.py` with `-e MODEL_NAME=openai/whisper-large-v3 -e BATCH_SIZE=2 -e GRAD_ACCUM=16`.

**Effort**: Just submit. Same script.

---

### 6. Whisper-large-v3 Pre-tuned on Ewe
**What**: If a Whisper-large-ee model exists on HF Hub (or we train one first), fine-tune that on Adja instead of vanilla Whisper-large.

**Why**: E4 > C2 by 2.1 points because of Ewe pre-training. Same advantage at larger scale should give even bigger wins.

**Check**: Search HuggingFace for `whisper-large` + `ewe` models. If none exist, this becomes a 2-step experiment (train Whisper-large on Ewe first — Ewe has 1,130h of data available in WAXAL).

**Effort**: TBD based on whether a pre-trained model exists.

---

## Tier 3 — Research-Grade Techniques

### 7. Focal CTC Loss for Rare Characters
**What**: Weight CTC loss inversely by character frequency. `ɖ`, `ŋ`, and tone marks get higher gradient signal than common chars like `e`, `a`.

**Why**: Standard CTC treats `e` (6% freq) and `ɖ` (1.5% freq) equally. Focal loss: `(1-p)^γ × CE`. Confident predictions contribute less, rare chars get more attention.

**Expected impact**: 3-9% accuracy improvement on rare chars per [Feng et al. 2019](https://www.hindawi.com/journals/complexity/2019/9345861/).

**How**: Modify `ctc_finetune.py` to compute class weights from training vocab, apply in custom CTC loss wrapper.

**Effort**: 1-2 days. Needs retraining C3/C4/C5.

---

### 8. Multi-Task Tone Classification
**What**: Add a second head to the Whisper/Wav2Vec2 encoder that predicts tone labels (high/mid/low/rising/falling) per frame alongside character prediction.

**Why**: Adja is tonal. Tone marks are character changes in our vocab. A dedicated tone head forces the encoder to explicitly model pitch.

**Reference**: [wav2vec 2.0 for Yoruba Tone Recognition (ACM TALLIP 2024)](https://dl.acm.org/doi/10.1145/3690384) — achieved Tone Error Rate = 17.72%.

**How**: Derive tone labels from existing tone marks (é → high, è → low, no mark → mid). Train with `loss = ctc_loss + λ × tone_ce_loss`.

**Effort**: 2-3 days. Needs retraining.

---

### 9. Hybrid Vocabulary (Split Tone Marks)
**What**: Change vocabulary from composed (ɔ̀ as 1 token) to decomposed (ɔ + ̀ as 2 tokens). Model shares knowledge of `ɔ` across toned/untoned variants.

**Why**: Currently `ɔ̀`, `ɔ́`, `ɔ` are unrelated tokens in the model's eyes. With split vocab, the model learns `ɔ` from all occurrences, then separately predicts the tone.

**Reference**: See `docs/improving-adja-character-recognition.md` section 1.

**How**: Change `char2idx` construction in `ctc_finetune.py` to use NFD normalization and treat combining marks separately.

**Effort**: 1 day. Needs retraining.

---

### 10. TTS Data Augmentation (with MMS-TTS-Ewe)
**What**: Generate synthetic Adja audio from Adja text using `facebook/mms-tts-ewe` (Ewe TTS — closest available Gbe TTS). Add to training data.

**Why**: 14.3% WER reduction per [Frustratingly Easy Data Augmentation (2025)](https://arxiv.org/abs/2509.15373). Text diversity matters more than speaker diversity.

**Caveat**: Ewe TTS won't perfectly produce Adja phonemes. Useful as a weak signal for rare characters but don't expect miracles.

**Effort**: 2-3 days. Synthesize 10x more data, retrain.

---

### 11. AfriHuBERT as Alternative Base Model
**What**: Use `AfriHuBERT` (pretrained on 10K+ hours from 1,226 African languages) as encoder instead of MMS/XLS-R.

**Why**: Better tonal representations since it actually saw African tonal languages during pretraining. -2.1% WER vs mHuBERT-147 on African ASR.

**Reference**: [Alabi et al. 2025](https://arxiv.org/abs/2409.20201)

**How**: Swap `MODEL_NAME` env var, may need minor adapter tweaks.

**Effort**: 1-2 days including code adaptation.

---

### 12. Omnilingual ASR Zero-Shot (HPC-only)
**What**: Run Meta's Nov 2025 omnilingual-ASR-LLM-7B-ZS on Adja audio in zero-shot mode (no training).

**Why**: Claims support for 1,600+ languages including unseen ones via IETF codes. Perfect test case for Adja (`aj_Latn`).

**Constraint**: Requires torch 2.8 + CUDA 13. Not available on HF Jobs default runners. **Must run on HPC or local GPU.**

**How**: Script already written at `scripts/hf_jobs/omni_asr_zeroshot.py`. Adapt to HPC SLURM job with CUDA 13 module loaded.

**Effort**: 2-4 hours once on HPC.

---

## Tier 4 — Questions to Answer Before Doing

### 13. How much data are we actually using?
Check: do ALL 1,277 training utterances contribute, or are some skipped due to length/quality filters in the current scripts? Could be a hidden "free" data increase.

### 14. Is the dev set the right size?
160 dev utterances = small but not tiny. A 1% CER difference = ~16 character errors. Consider reporting 95% confidence intervals for all CER numbers in the paper.

### 15. Error analysis by character
For the best model (E4), compute per-character error rate. Which characters get confused most? That tells us where focal loss / tone head would help most.

---

## Done / Completed (for reference)

- ✅ Unicode NFC normalization applied to all data
- ✅ SpecAugment / speed perturbation (in B1 BiLSTM-CTC only — could add to fine-tuning pipelines)
- ✅ Whisper zero-shot baseline established
- ✅ Whisper + Whisper-Ewe fine-tuning showing positive results
- ✅ MMS + French adapter partially working (C3), rerun queued
- ✅ 5-gram character LM built (`data/char_5gram.arpa`) — ready for integration
- ✅ Research guides written in `docs/`:
  - `asr-learning-guide.md` — ASR fundamentals
  - `language-models-for-asr.md` — n-grams, focal loss, hybrid vocab theory
  - `improving-adja-character-recognition.md` — techniques survey
  - `training-hyperparameters-guide.md` — epochs, batch size, LR theory
