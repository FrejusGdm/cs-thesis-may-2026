# D4 — Whisper + Char LM Shallow Fusion (E4 Whisper-Ewe)

## What This Tests

Post-hoc decoding improvement for our best model (E4 Whisper-Ewe → Adja,
CER=24.90%, WER=73.09% with greedy decoding). Generate N beam candidates
with Whisper's decoder, then **rescore each candidate** using the
character n-gram LM. Pick the highest combined score.

No retraining. Just shallow LM fusion at decoding time.

## How It Works

1. Load the trained E4 model from `JosueG/adja-asr-results/E4/best_model`.
2. Load the bigger 5-gram LM (`data/char_5gram.arpa`).
3. For each audio clip:
   - `model.generate(num_beams=5, num_return_sequences=5, output_scores=True)`
     → 5 beam candidates with AM (acoustic model) scores.
   - For each candidate, compute KenLM log-prob over character sequence.
   - Pick `argmax_i (AM_score_i + α · LM_score_i + β · word_count_i)`.

This is **shallow fusion** (rescoring). Real deep fusion would modify the
beam search itself step-by-step, which is not straightforward for Whisper.

## Script

`scripts/hf_jobs/decode_with_lm.py` with `DECODER=whisper`.

## Expected Impact

Whisper's decoder already encodes language-model-like behavior through
its autoregressive training. So LM fusion helps less than it does for CTC
models — expected WER drop is maybe 3-8 points vs CTC's 10-20 points.

Still worth measuring: if it gives a clean signal, we can scale it up.

## References

- `docs/language-models-for-asr.md` — beam search + shallow fusion explained
- [KenLM docs](https://kheafield.com/code/kenlm/)
- [Whisper-LM (2025)](https://arxiv.org/abs/2503.23542) — up to 51% WER reduction on low-resource
