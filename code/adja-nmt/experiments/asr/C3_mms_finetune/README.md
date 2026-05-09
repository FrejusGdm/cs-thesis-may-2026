# C3: MMS Fine-Tuned ASR for Adja

## Overview

This experiment fine-tunes Meta's Massively Multilingual Speech (MMS) model on Adja
ASR data. MMS was trained on 1,100+ languages and is specifically designed for
low-resource speech recognition, making it a strong candidate for Adja.

According to the 2025 PazaBench benchmark, MMS achieves the best ASR results among
publicly available models when training data is under 1 hour -- exactly our scenario.

## Model

**MMS** (Pratap et al., 2023) extends wav2vec 2.0 with massive multilingual pretraining
across 1,100+ languages. It uses a CTC (Connectionist Temporal Classification) head
for decoding, which is simpler and more robust than attention-based decoding for
low-resource settings.

- Paper: <https://jmlr.org/papers/v25/23-1318.html>
- HuggingFace model: `facebook/mms-1b-all` (1B parameters)

MMS covers many African languages in its pretraining data, though Adja (ajg) may not
be directly included. The key advantage is that MMS has seen acoustic patterns from
hundreds of tonal African languages, giving it a strong prior for Adja phonology.

## Approach

- **Architecture**: wav2vec 2.0 encoder + CTC linear head
- **Loss**: CTC loss
- **Decoding**: Greedy CTC decoding (argmax + blank removal)
- **Vocabulary**: Custom character-level vocabulary built from Adja transcriptions
- **Optimizer**: AdamW with linear warmup

The CTC vocabulary is rebuilt from the Adja character set (char_vocab.json produced
by data_prep.py). This includes Adja-specific characters like open-mid vowels and
nasal consonants that are critical for accurate transcription.

## Usage

```bash
# Full training run
python experiments/asr/C3_mms_finetune/train.py \
    --config experiments/asr/C3_mms_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C3_mms_finetune/outputs

# Dry run
python experiments/asr/C3_mms_finetune/train.py \
    --config experiments/asr/C3_mms_finetune/config.yaml \
    --data-dir data/adja_asr \
    --output-dir experiments/asr/C3_mms_finetune/outputs \
    --dry-run
```

## References

- Pratap, V., Tjandra, A., Shi, B., Tober, P., Babu, A., Kunber, S., ... & Auli, M. (2023).
  *Scaling Speech Technology to 1,000+ Languages*. JMLR, 25, 1-46.
  <https://jmlr.org/papers/v25/23-1318.html>
