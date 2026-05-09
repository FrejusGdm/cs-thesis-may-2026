# C2: Whisper Fine-Tuned ASR for Adja

## Overview

This experiment fine-tunes OpenAI's Whisper on Adja speech data. Starting from the
`openai/whisper-small` checkpoint, we adapt the model to produce Adja transcriptions
by training on the Adja ASR dataset with a custom training loop.

This is the natural follow-up to C1 (zero-shot Whisper), which established that
Whisper has essentially no ability to transcribe Adja out of the box. Fine-tuning
should dramatically close that gap, even with limited data (~1.5 hours).

## Model

**Whisper** (Radford et al., 2022) is a sequence-to-sequence Transformer trained on
680,000 hours of weakly-supervised multilingual audio from the web. It uses an
encoder-decoder architecture with log-mel spectrogram input and autoregressive
text token output.

- Paper: <https://cdn.openai.com/papers/whisper.pdf>
- HuggingFace model: `openai/whisper-small` (244M parameters)

We start with `whisper-small` to keep training time manageable and iterate on
hyperparameters. If results are promising, scaling to `whisper-medium` or
`whisper-large-v3` is straightforward (just change `model_name` in config).

## Approach

- **Architecture**: Encoder-decoder Transformer (Whisper)
- **Loss**: Cross-entropy on decoder token predictions
- **Decoding**: Greedy (no beam search during evaluation for speed)
- **Optimizer**: AdamW with linear warmup
- **Regularization**: Optional encoder freezing to prevent catastrophic forgetting
- **Early stopping**: Based on dev-set CER

## What This Script Does

1. Loads the Adja train/dev sets from TSV manifests.
2. Initializes WhisperForConditionalGeneration and WhisperProcessor.
3. Optionally freezes the encoder.
4. Runs a custom training loop with gradient accumulation.
5. Evaluates on dev set each epoch (greedy decode, compute CER/WER).
6. Applies early stopping on dev CER.
7. Saves best checkpoint, metrics.json, decode_samples.txt, timing.json.

## Usage

```bash
# Full training run
python experiments/asr/C2_whisper_finetune/train.py \
    --config experiments/asr/C2_whisper_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C2_whisper_finetune/outputs

# Dry run (2 steps, 5 samples)
python experiments/asr/C2_whisper_finetune/train.py \
    --config experiments/asr/C2_whisper_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C2_whisper_finetune/outputs \
    --dry-run
```

## References

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).
  *Robust Speech Recognition via Large-Scale Weak Supervision*.
  <https://cdn.openai.com/papers/whisper.pdf>
