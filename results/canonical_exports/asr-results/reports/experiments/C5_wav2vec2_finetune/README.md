# C5: wav2vec 2.0 (English-only) Fine-Tune (CTC)

## What was tested
Fine-tune `facebook/wav2vec2-large-960h` on Adja with a freshly-initialized CTC head (115-token vocab). This is the English-only wav2vec 2.0 trained on LibriSpeech 960h — included as a deliberate contrast to multilingual XLS-R 300M, to see whether English-only pretraining generalizes at all to Adja.

## Model
- **facebook/wav2vec2-large-960h** — https://huggingface.co/facebook/wav2vec2-large-960h

## Script
`experiments/asr/C5_wav2vec2_finetune/ctc_finetune.py` (same CTC loop as C3/C4).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens — `lm_head` reinit (ckpt 32 chars → model 115)
- Max epochs: 30
- Early stopping: patience=20 on dev CER

## HF Jobs Run
- Job ID: `69e03d15cd8c002f31dfc3f6`
- Logs: `hf jobs logs 69e03d15cd8c002f31dfc3f6`
- URL: https://huggingface.co/jobs/JosueG/69e03d15cd8c002f31dfc3f6
- Wall time: ~28 minutes
- Results: not pushed
