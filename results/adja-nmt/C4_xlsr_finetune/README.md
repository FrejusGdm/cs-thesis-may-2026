# C4: XLS-R 300M Fine-Tune (CTC)

## What was tested
Fine-tune `facebook/wav2vec2-xls-r-300m` on Adja with a freshly-initialized CTC head (115-token vocab). XLS-R 300M is the multilingual wav2vec 2.0 trained on 128 languages, chosen as a smaller and faster alternative to MMS for CTC. This was the first run that surfaced our CTC-collapse failure mode.

## Model
- **facebook/wav2vec2-xls-r-300m** — https://huggingface.co/facebook/wav2vec2-xls-r-300m
- No language adapter.

## Script
`experiments/asr/C4_xlsr_finetune/ctc_finetune.py` (same CTC loop as C3).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens — `lm_head` freshly initialized
- Trainable params: 311,346,419 / 315,556,595 total (gradient checkpointing ON)
- Max epochs: 30
- Early stopping: patience=20 on dev CER
- Evaluation: dev CER/WER every epoch

## HF Jobs Run
- Job ID: `69e03d0ecd8c002f31dfc3f4`
- Logs: `hf jobs logs 69e03d0ecd8c002f31dfc3f4`
- URL: https://huggingface.co/jobs/JosueG/69e03d0ecd8c002f31dfc3f4
- Wall time: ~28 minutes (ran 6 epochs; stopped when patience exhausted against a 100% CER ceiling)
- Results: not pushed (no meaningful checkpoint to save)
