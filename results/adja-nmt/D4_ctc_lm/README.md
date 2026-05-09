# D4 — CTC + Char LM Beam Search (C4v2 XLS-R)

## What This Tests

Post-hoc decoding improvement: take our already-trained C4v2 XLS-R model
(CER=25.05%, WER=72.39% with greedy decoding) and decode its CTC logits
with **beam search + character n-gram language model** instead of greedy
argmax.

No retraining. Just a smarter decoder that knows which character sequences
are statistically likely in Adja.

## How It Works

1. Load the trained C4v2 model from `JosueG/adja-asr-results/C4v2/best_model`.
2. Load the new bigger char 5-gram LM (`data/char_5gram.arpa`, 2MB, built
   from 13,327 Adja sentences).
3. For each audio clip:
   - Run encoder to get per-frame logits (same as training).
   - Run pyctcdecode beam search over logits with the LM:
     `score = acoustic + α · LM_logprob + β · word_count`
4. Try several (α, β) pairs on dev; pick best; report test metrics.

## Script

`scripts/hf_jobs/decode_with_lm.py` with `DECODER=ctc`.

## Alpha / Beta Grid

- α ∈ {0.5, 1.0}: how much to trust the LM
- β ∈ {0.0, 1.0, 2.0}: word insertion bonus (prevents LM from shortening output)

## References

- [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) — beam search + KenLM
- [Whisper-LM (2025)](https://arxiv.org/abs/2503.23542) — up to 51% WER reduction from LM
- `docs/language-models-for-asr.md` — full theory in our docs
