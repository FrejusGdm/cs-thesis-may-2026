# C2: Whisper-Small Fine-Tune on Adja

## What was tested
Full fine-tuning of `openai/whisper-small` on the Adja train split (1277 utterances) with a custom training loop. Primary target: move the Whisper decoder from hallucinating zero-shot output (see `C1_whisper_zeroshot/`) to producing recognizable Adja transcripts.

## Model
- **openai/whisper-small** — https://huggingface.co/openai/whisper-small

## Script
`experiments/asr/C2_whisper_finetune/whisper_finetune.py` (custom loop; NOT `Seq2SeqTrainer` — see `learnings-from-the-past/` for why).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens (after `fix_tokenizer()` + `resize_token_embeddings()`)
- Max epochs: 50
- Early stopping: patience=20 on dev CER
- Evaluation: dev CER/WER every epoch, with decode samples
- Best-model selection: lowest dev CER across all epochs

## HF Jobs Run
- Job ID: `69e04dbfac288e522d8eee2f`
- Logs: `hf jobs logs 69e04dbfac288e522d8eee2f`
- URL: https://huggingface.co/jobs/JosueG/69e04dbfac288e522d8eee2f
- Total wall time: ~332 minutes
- Results pushed to: `JosueG/adja-asr-results/C2/`

## Caveat on log label
The HF Jobs log header for this job reads `C3v2: facebook/mms-1b-all`. The numeric curve matches the C2 metrics reported in `experiments/registry.md` (best dev CER=27.02% at ep 47), so it is recorded here as C2 per the registry's mapping. Worth double-checking the experiment script that was dispatched under this job ID.
