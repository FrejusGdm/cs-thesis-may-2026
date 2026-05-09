# E4v4 — Whisper-Ewe retrain with EOS masking fix

## Summary

Retrain of `dodziraynard/whisper-small-ee` on Adja after fixing the
critical EOS-masking bug discovered in `docs/whisper-training-gotchas.md`
Section 0. The bug (masking every occurrence of `pad_token_id` in
training labels, which equals `eos_token_id` in Whisper) was the root
cause of both E4v2 (CER=76-90%) and the first E4v3 attempt
(CER=400%+, loss→0.01).

## Numbers

- **Best dev CER: 37.18%** at epoch 20
- Best dev WER: 83.61% at epoch 20
- Total training time: 226 minutes (40 epochs on A10G-large)
- Final epoch loss: 0.0005 (severely overfit)
- Final epoch dev CER: 51.27% (plateaued from epoch 28 onward)

## Verdict

**EOS fix worked — hallucinations gone. But overfitting replaced them.**

The training loss shows classic memorization:
- Epoch 1: loss=6.64, dev CER=459%
- Epoch 10: loss=0.39, dev CER=57% — breakthrough
- Epoch 20: loss=0.011, dev CER=37% — **best**, sweet spot
- Epoch 40: loss=0.0005, dev CER=51% — severe overfit

With only 1,277 training utterances and a 240M-parameter decoder, the
model memorizes training data by epoch 20-25. Early stopping on dev CER
(patience=20) didn't trigger because dev CER oscillated chaotically
between 40-60% in the overfit regime, occasionally dipping slightly
and resetting the patience counter.

## Why it's still worse than the original E4 (CER=24.90%)

The lost original E4 had:
- `transformers` version that auto-padded to 3000 mel (no explicit pad
  in collate)
- Same EOS-masking bug (we now think it was PARTIALLY broken but
  worked well enough to reach CER=24.9%)
- Possibly different `generation_config` defaults

Our E4v4 has the EOS fix applied correctly AND the explicit 3000-mel
pad. Both are theoretically better than the original. The worse result
suggests that:
1. The original E4 benefited from the partially-broken EOS training
   (early stop on imperfect EOS prevented overfitting)
2. OR the 3000-mel pad + silence padding IS still a subtle harm
3. OR the original had something else different we haven't identified

## What to try next

### Immediate regularization tweaks (cheap, likely to help):

1. **Lower LR** from 1e-5 to 5e-6 — slow memorization
2. **Higher weight decay** (0.05 instead of 0.01)
3. **Stronger dropout** on the decoder (currently relies on Whisper's default 0.0)
4. **Much more aggressive early stopping** — patience=5 on dev CER
5. **Gradient clipping** at 0.5 instead of 1.0

### Structural fixes:

6. **Freeze encoder for first 10 epochs** — lets the decoder stabilize
   around the good pre-trained features before overfitting starts
7. **Use `label_smoothing=0.1`** in cross-entropy loss
8. **Mixed-precision training** (bf16) — adds stochastic noise that
   helps generalization

### Data fixes:

9. **Speed perturbation** (±10%) for data augmentation — effectively
   3x training data
10. **SpecAugment** on mel features — time/frequency masking

## Follow-up action

I suggest submitting **one more E4v5 run** with the minimal regularization
changes (LR=5e-6, weight_decay=0.05, label_smoothing=0.1, patience=5).
That's ~$6 of HF credits. Realistic target: CER < 30%.

If E4v5 still overfits, the issue is structural and we should accept
that C4v2 + LM (CER=22.67%) is our best reproducible result, and move
on to Tier 2 experiments (AfriHuBERT, TTS augmentation, Parakeet).

## Training curve

```
ep  1: loss=6.64  dev_CER=459%  (random init)
ep  5: loss=2.55  dev_CER=209%  (first sanity)
ep 10: loss=0.39  dev_CER= 57%  (breakthrough)
ep 16: loss=0.026 dev_CER= 43%  (converging)
ep 20: loss=0.011 dev_CER= 37%  ← BEST
ep 25: loss=0.003 dev_CER= 45%  (overfit starts)
ep 30: loss=0.001 dev_CER= 46%
ep 40: loss=0.0005 dev_CER= 51% (locked in overfit)
```

## Files

- Model: `JosueG/adja-asr-results/E4v4/best_model/` (full processor + config)
- Metrics: `JosueG/adja-asr-results/E4v4/metrics.json`
- Gotchas documented: `docs/whisper-training-gotchas.md` Section 0
- Fixed training script: `scripts/hf_jobs/whisper_finetune.py`
