# C4: XLS-R Fine-Tuned ASR for Adja

## Overview

This experiment fine-tunes Meta's XLS-R (Cross-Lingual Speech Representations) model
on Adja ASR data. XLS-R is a self-supervised speech model pretrained on 128 languages,
providing cross-lingual representations that transfer well to unseen languages.

Compared to C3 (MMS), XLS-R covers fewer languages (128 vs 1100+) but uses a larger
self-supervised pretraining objective. Comparing C3 and C4 tells us whether broader
language coverage (MMS) or deeper self-supervised learning (XLS-R) matters more for
Adja ASR.

## Model

**XLS-R** (Babu et al., 2022) extends wav2vec 2.0 to the cross-lingual setting with
pretraining on 436K hours of speech from 128 languages. Available in 300M and 1B
parameter variants.

- Paper: <https://arxiv.org/abs/2111.09296>
- HuggingFace model: `facebook/wav2vec2-xls-r-300m` (300M parameters)

We use the 300M variant to balance training cost against representation quality.
The 1B variant can be tested by changing `model_name` in config.

## Approach

- **Architecture**: wav2vec 2.0 encoder + CTC linear head
- **Loss**: CTC loss
- **Decoding**: Greedy CTC decoding (argmax + blank removal)
- **Vocabulary**: Custom character-level vocabulary from Adja transcriptions
- **Feature encoder**: Frozen (CNN layers) to prevent catastrophic forgetting
- **Optimizer**: AdamW with linear warmup

## Usage

```bash
# Full training run
python experiments/asr/C4_xlsr_finetune/train.py \
    --config experiments/asr/C4_xlsr_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C4_xlsr_finetune/outputs

# Dry run
python experiments/asr/C4_xlsr_finetune/train.py \
    --config experiments/asr/C4_xlsr_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C4_xlsr_finetune/outputs \
    --dry-run
```

## References

- Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., ... & Auli, M. (2022).
  *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale*.
  <https://arxiv.org/abs/2111.09296>
