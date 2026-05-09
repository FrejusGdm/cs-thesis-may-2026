# C1: Whisper Zero-Shot ASR Baseline for Adja

## Overview

This experiment runs OpenAI's Whisper model on Adja speech data **with no fine-tuning**.
The goal is to establish a zero-shot baseline: what does a large multilingual ASR model
produce when it encounters Adja, a low-resource Gbe language it almost certainly never
saw during training?

We expect poor results -- and that is the point. The WER/CER numbers from this
experiment quantify the gap that subsequent fine-tuning experiments (C2, C3, ...) need
to close. If Whisper already performed well zero-shot, fine-tuning research would be
less motivated.

## Model

**Whisper** (Radford et al., 2022) is a sequence-to-sequence Transformer trained on
680,000 hours of weakly-supervised multilingual audio from the web. It supports
transcription and translation across 99 languages.

- Paper: <https://cdn.openai.com/papers/whisper.pdf>
- HuggingFace model: `openai/whisper-large-v3`

Whisper's training data is dominated by English and other high-resource languages.
Adja (ISO 639-3: `ajg`), spoken primarily in Togo and Benin, is not listed among
Whisper's supported languages. The closest related language with any representation
might be French (due to colonial influence in the region) or Ewe/Fon (related Gbe
languages), though neither is a good proxy for Adja phonology and tone.

## What This Script Does

1. Loads the Adja test set (and optionally dev set) from the TSV manifest produced by
   `experiments/asr/shared/data_prep.py`.
2. Runs Whisper inference under multiple language settings:
   - `None` -- let Whisper auto-detect the language
   - `"fr"` -- force French (geographically plausible fallback)
   - Forced transcription mode (no language token, just transcribe)
3. Computes WER and CER for each setting using `experiments/asr/shared/metrics.py`.
4. Saves results: `metrics.json`, `decode_samples.txt`, and `timing.json`.

## Usage

```bash
# Full run on test set
python experiments/asr/C1_whisper_zeroshot/infer.py \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C1_whisper_zeroshot/outputs

# Dry run (5 utterances only)
python experiments/asr/C1_whisper_zeroshot/infer.py \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C1_whisper_zeroshot/outputs \
    --dry-run

# Use a smaller model
python experiments/asr/C1_whisper_zeroshot/infer.py \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C1_whisper_zeroshot/outputs \
    --model-size small

# With local model cache
python experiments/asr/C1_whisper_zeroshot/infer.py \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C1_whisper_zeroshot/outputs \
    --model-cache /scratch/models/whisper
```

## Expected Results

We expect WER close to 100% (or above, since insertions can push WER over 100%).
Whisper will likely:
- Detect the language as French, Ewe, or something else entirely
- Produce French or English text that bears no relation to the Adja reference
- Occasionally capture phonetically similar fragments

These numbers serve as the upper bound of error that fine-tuning must improve upon.

## References

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).
  *Robust Speech Recognition via Large-Scale Weak Supervision*.
  <https://cdn.openai.com/papers/whisper.pdf>
