# E1: MMS-1B + Ewe Adapter Fine-Tune (CTC)

## What was tested
Fine-tune `facebook/mms-1b-all` with the Ewe (`ewe`) adapter loaded, to leverage Ewe's closeness to Adja in the Gbe language family. Hypothesis: Ewe is typologically the nearest MMS-supported language to Adja, so its adapter + encoder states should transfer better than the French (`fra`) adapter used in C3.

## Model
- **facebook/mms-1b-all** — https://huggingface.co/facebook/mms-1b-all
- Adapter: `ewe`

## Script
`experiments/asr/E1_mms_ewe_finetune/ctc_finetune.py` (CTC loop).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens (fixed after running into Ewe-adapter vocab mismatch — see conclusion)
- Adapter: ewe
- Trainable params: 960,508,855 / 964,719,031 total
- Max epochs: 30

## HF Jobs Run
- Job ID: `69e03bb9ac288e522d8eedae`
- Logs: `hf jobs logs 69e03bb9ac288e522d8eedae`
- URL: https://huggingface.co/jobs/JosueG/69e03bb9ac288e522d8eedae
- Wall time: minutes (crashed before completing epoch 1)
- Results: not pushed (job errored)
