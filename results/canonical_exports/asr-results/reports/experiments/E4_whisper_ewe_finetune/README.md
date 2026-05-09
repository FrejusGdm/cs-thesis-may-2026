# E4: Whisper-Ewe → Adja Transfer Fine-Tune (BEST MODEL)

## What was tested
Start from a Whisper checkpoint already fine-tuned on Ewe, then fine-tune onto Adja. Hypothesis: because Ewe and Adja are closely related Gbe-family languages, the encoder/decoder should already have the right acoustic-phonetic biases (front-mid vowels ɛ/ɔ, velar nasal ŋ, implosive ɖ, tones), dramatically reducing what has to be learned. This is currently the best model on the project.

## Model
- **Whisper-small-ee** (Whisper-small previously fine-tuned on Ewe) → further fine-tuned on Adja

## Script
`experiments/asr/E4_whisper_ewe_finetune/whisper_finetune.py` (same custom loop as C2).

## Config
- Dataset: `JosueG/adja-tts-orpheus` (1277 train / 160 dev / 160 test, seed=42)
- Vocab: 115 tokens (Ewe tokenizer adapted via `fix_tokenizer()`)
- Max epochs: 50
- Early stopping: patience=20 on dev CER (not triggered)
- Evaluation: dev CER/WER every epoch

## HF Jobs Run
- Job ID: `69e04dc7ac288e522d8eee31`
- Logs: `hf jobs logs 69e04dc7ac288e522d8eee31`
- URL: https://huggingface.co/jobs/JosueG/69e04dc7ac288e522d8eee31
- Wall time: ~223 minutes
- Results pushed to: `JosueG/adja-asr-results/E4/`

## Caveat on log label
The HF Jobs log header for this job reads `C4v2: facebook/wav2vec2-xls-r-300m`. The dev-CER curve captured in the log, however, exactly matches the E4 story described in `experiments/registry.md` (slow 14-epoch plateau at 100% while the Ewe tokenizer adapts, then rapid improvement to 24.9% by epoch 50). The numbers are filed here per the registry mapping; confirm which script actually ran under this job ID before citing.
