# C3: MMS-1B + French Adapter Fine-Tune

## What was tested
Fine-tune `facebook/mms-1b-all` on Adja with the pretrained French adapter loaded, and a freshly-initialized CTC head sized for the 115-token Adja vocab. Motivation: MMS-1B was trained on 1100+ languages and its `fra` adapter is the closest Gbe-family analog available out of the box (French has extensive shared loanword coverage with Adja's local lexicon). This was the first CTC model to learn anything at all — it is the partial success of the CTC track.

## Model
- **facebook/mms-1b-all** — https://huggingface.co/facebook/mms-1b-all
- Adapter: `fra`

## Script
`experiments/asr/C3_mms_finetune/ctc_finetune.py` (CTC loss, custom loop).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens — triggered `lm_head` reinit (ckpt 154 → model 115)
- Adapter: fra (French)
- Trainable params: 960,840,634 / 965,050,810 total (gradient checkpointing ON)
- Max epochs: 30
- Early stopping: patience=5 on dev CER
- Evaluation: dev CER/WER every epoch

## HF Jobs Run
- Job ID: `69e037decd8c002f31dfc376`
- Logs: `hf jobs logs 69e037decd8c002f31dfc376`
- URL: https://huggingface.co/jobs/JosueG/69e037decd8c002f31dfc376
- Wall time: ~36 minutes (stopped at ep 6)
- Results pushed to: `JosueG/adja-asr-results/C3/`
