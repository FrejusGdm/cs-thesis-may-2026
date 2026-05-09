# C1: Whisper Zero-Shot Baselines

## What was tested
Zero-shot ASR performance of pretrained Whisper models on Adja audio — no fine-tuning. Three model sizes evaluated back-to-back in one HF Jobs run: `whisper-tiny`, `whisper-small`, and `whisper-large-v3`. Goal: establish a baseline to show why fine-tuning is required for a low-resource language like Adja.

## Models
- **openai/whisper-tiny** — https://huggingface.co/openai/whisper-tiny
- **openai/whisper-small** — https://huggingface.co/openai/whisper-small
- **openai/whisper-large-v3** — https://huggingface.co/openai/whisper-large-v3

## Script
`experiments/asr/C1_whisper_zeroshot/run_zeroshot.py` (evaluation-only; no gradient steps).

## Config
- Dataset: `JosueG/adja-tts-orpheus`, test + dev splits (160 each, seed=42)
- Task: transcription (forced_decoder_ids default; language auto-detected per audio)
- Device: cuda (HF Jobs A100)
- No language hint passed to the decoder

## HF Jobs Run
- Job ID: `69e02cb9ac288e522d8eed44`
- Logs: `hf jobs logs 69e02cb9ac288e522d8eed44`
- URL: https://huggingface.co/jobs/JosueG/69e02cb9ac288e522d8eed44
- Results pushed to: `JosueG/adja-asr-results` (top-level)
