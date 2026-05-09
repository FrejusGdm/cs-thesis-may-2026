# C5: wav2vec 2.0 Fine-Tuned ASR for Adja

## Overview

This experiment fine-tunes the original wav2vec 2.0 model (English-pretrained) on
Adja ASR data. This serves as a **control experiment** to measure whether multilingual
pretraining (C3: MMS, C4: XLS-R) actually helps compared to a strong English-only
self-supervised model.

If C4 (XLS-R, 128 languages) significantly outperforms C5 (wav2vec 2.0, English only),
it confirms that cross-lingual pretraining transfers useful acoustic knowledge for Adja.
If the gap is small, it suggests that the self-supervised learning objective itself
matters more than the language diversity of pretraining data.

## Model

**wav2vec 2.0** (Baevski et al., 2020) learns speech representations through
self-supervised contrastive learning on unlabeled audio. The model masks portions
of the latent speech representations and learns to identify the correct quantized
representations from distractors.

- Paper: <https://arxiv.org/abs/2006.11477>
- HuggingFace model: `facebook/wav2vec2-large-960h` (315M parameters)
- Pretrained on: 960 hours of LibriSpeech (English only)

This model was pretrained exclusively on English audiobook data, making it a useful
baseline to isolate the effect of multilingual pretraining. The acoustic features
learned from English may still transfer partially to Adja (both have similar
fundamental frequencies, consonant-vowel structures), but the model has never heard
tonal language patterns during pretraining.

## Approach

- **Architecture**: wav2vec 2.0 encoder + CTC linear head
- **Loss**: CTC loss
- **Decoding**: Greedy CTC decoding (argmax + blank removal)
- **Vocabulary**: Custom character-level vocabulary from Adja transcriptions
- **Feature encoder**: Frozen (CNN layers)
- **Optimizer**: AdamW with linear warmup

## Usage

```bash
# Full training run
python experiments/asr/C5_wav2vec2_finetune/train.py \
    --config experiments/asr/C5_wav2vec2_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C5_wav2vec2_finetune/outputs

# Dry run
python experiments/asr/C5_wav2vec2_finetune/train.py \
    --config experiments/asr/C5_wav2vec2_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C5_wav2vec2_finetune/outputs \
    --dry-run
```

## References

- Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).
  *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*.
  NeurIPS 2020. <https://arxiv.org/abs/2006.11477>
