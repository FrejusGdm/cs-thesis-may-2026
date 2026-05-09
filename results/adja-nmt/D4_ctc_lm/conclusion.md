# D4 CTC + LM — Conclusion

## Summary

Character 5-gram LM fusion (shallow fusion with α, β hyperparameters)
on our C4v2 XLS-R 300M model produced **modest but reproducible WER
improvements**:

| Decoding | Test CER | Test WER | Normalized CER | Normalized WER |
|---|---|---|---|---|
| Greedy (baseline) | 24.35% | 73.98% | 23.04% | 72.53% |
| LM 6-cfg grid (α=0.5, β=0) | 24.10% | 73.07% | 22.80% | 71.32% |
| LM 56-cfg grid (α=0.5, β=0.0) | 24.08% | 72.53% | 22.76% | **70.76%** |
| **LM Optuna 40-trial** (α=0.67, β=-0.29) | **24.04%** | **72.71%** | **22.67%** | **70.76%** |

**Absolute improvement from best LM config**: -0.31 CER, -1.27 WER (raw).
**Normalized WER improvement**: -1.77 points, about 2.4% relative.

## Grid vs Optuna

Both search methods converged to essentially the same optimum:
- **Grid's best**: α=0.5, β=0.0 → norm WER=70.76%
- **Optuna's best**: α=0.67, β=-0.29 → norm WER=70.76% (norm CER 0.09 lower)

Differences are at noise level. Both searches confirm the optimum is
in a flat basin around (α≈0.5-1.3, β≈-0.5-0.1). Optuna is slightly
more sample-efficient when you don't know grid bounds; grid gives
more confidence in the landscape shape.

## Verdict

✅ **Converged to a local optimum.** Optuna's TPE sampler found α≈0.67,
β≈-0.29 as the best configuration. The top-10 trials cluster tightly
around (α∈[0.7, 1.4], β∈[-0.5, 0.1]), suggesting we're at a real
optimum — not noise. Grid search independently landed near the same
point (α=0.5, β=0.0 → dev CER=24.52%).

## Why the LM gain is small

Our char 5-gram LM (13K Adja sentences) fixes **low-hanging errors**
— punctuation, spacing, common character confusions. It cannot fix:
- Words never seen in the LM corpus
- Hard acoustic confusions (u↔ɔ, nasal vowels)
- Segmentation errors that cross word boundaries

The acoustic model (CER=25%) is already doing most of the work. Full
analysis in `docs/why-lm-barely-helped.md`.

## Compared to literature

| Source | Language | AM WER | LM WER | Relative |
|---|---|---|---|---|
| Our work | Adja | 72.5% | 70.8% | 2.4% |
| Finnish Reddit | Finnish | 7.0% | 3.3% | 53% (after Optuna!) |
| Whisper-LM paper | Basque | 10.5% | 5.2% | 51% |
| MMS paper avg | Various | ~20% | ~15% | 25% |
| Yi et al. 2023 | Low-resource | varies | varies | 5-15% typical |

We're in the low-end range for low-resource. Larger gains require:
- Word-level LM (not char) → longer context, better modeling
- Much larger LM corpus (100K+ sentences)
- Neural LM (LSTM/Transformer) → capture long-range dependencies
- KenLM with Kneser-Ney smoothing (vs our pure-Python Katz backoff)

## Follow-up

See `ideas/asr-improvements-roadmap.md` item #2 (Char LM beam search
with pyctcdecode) for the proper CTC prefix beam search approach.
Our current decode uses simple top-k per frame; pyctcdecode's real
prefix beam would likely give another 1-3 WER points.
