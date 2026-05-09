# D4 Whisper + LM — Conclusion

## What we ran

LM shallow fusion on the E4v2 Whisper-Ewe checkpoint (the only Whisper
checkpoint we have fully saved on HF Hub — the original E4 was never
uploaded with full processor files).

**⚠️ Important caveat**: E4v2 is a BROKEN training run. Its greedy
output is CER=89.53% — the model learned a hallucination loop during
training (overfit to 30-second padded mel inputs). This is NOT
representative of what a correctly-trained Whisper-Ewe would produce.

## Numbers

| Decoding | Test CER | Test WER | Normalized CER | Normalized WER |
|---|---|---|---|---|
| Greedy (baseline) | 89.53% | 130.73% | 86.76% | 129.42% |
| **LM (α=0.5, β=0)** | **42.92%** | **93.38%** | **39.06%** | **90.60%** |
| **Δ** | **-46.61 pts** | **-37.35 pts** | **-47.70 pts** | **-38.82 pts** |

## Verdict

**Huge apparent LM gain, but from a bad baseline.** Don't interpret
this as "LM fusion is 51% effective for Whisper". The real story:

- The acoustic model was hallucinating long sequences unrelated to the
  audio
- LM scoring penalized these obviously-implausible-Adja sequences
- Forced the model back to shorter, more realistic outputs
- Still not GOOD (43% CER is objectively bad) — just less bad

This is the **ceiling-effect flipped**: when AM quality is low, there's
enormous room for LM to improve things. When AM is already good (like
C4v2 at 25% CER), LM has little left to fix.

## What this tells us scientifically

1. **LM + weak AM > LM + strong AM** in terms of absolute improvement
2. **But strong AM without LM > weak AM with LM** — the baseline matters
3. The C4v2 comparison is the *real* signal: modest 1-2 point improvement
   is what a properly-trained model gets from a char n-gram LM
4. If we could fix the Whisper training (retrain to CER=25%), adding an
   LM would likely give another ~1-2 points, not 37

## What would a "proper" Whisper+LM result look like?

Extrapolating from the Whisper-LM paper and our C4v2 results, a
correctly-trained Whisper-Ewe (~24-25% CER) with char LM would give:
- CER: 25% → 23-24%
- WER: 73% → 69-71%
- **Not the 47-point swing we see here.**

## Follow-up

- Fix Whisper training to reproduce original E4 (CER=24.9%) — blocked
  on transformers version drift; tracked in `docs/whisper-training-gotchas.md`
- Re-run LM fusion on the corrected Whisper checkpoint
- Use results to write the honest paper story (not the misleading 47-pt
  number)
