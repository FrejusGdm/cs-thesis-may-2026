# C3: Conclusion

## Verdict
**killed early**

## What happened
MMS-1B with the French adapter showed real, promising CTC learning for the first 6 epochs — loss dropped from 12.47 → 3.54 (monotonic), and dev WER improved from 99.91% to 98.42%. However, because dev CER was luckiest at epoch 1 (87.19% — probably a random-initialization artifact where the CTC head emitted a few near-correct characters by chance), the `patience=5` early-stopping criterion never saw an "improvement" and killed training at epoch 6 with `Best CER=87.19%` at ep 1. This is the only CTC-based model in the C-track that learned *anything* rather than collapsing to all-blanks; the stopping rule cut it off before we could see how far the MMS encoder + French adapter could actually go.

## Why
Two compounding issues:

1. **Patience too aggressive (5 epochs)**. CTC models with freshly-initialized heads typically spend 10-20 epochs in the "blank-dominant" regime before the character distribution sharpens. Our patience should have been at least 20 for these models.
2. **Best-CER initialization luck**. Epoch 1's CER of 87.19% was produced by a model that had barely trained — it emitted 1-3 character outputs that happened to be present in many references. Once the model started producing longer, more confident outputs at epochs 2-6, CER rose before it could fall again.

The dev loss and dev WER trends both suggest the model was genuinely improving at the time of the kill.

## What to do next
Investigated. Root cause: `patience=5` early stopping fired on a noisy metric before real learning could register. **Rerun queued as `C3v3-s42` (in registry: `experiments/registry.md` line 29)** with `patience=20`, `max_epochs=50`, on A100. Expect dev CER to drop meaningfully below the current 87%+ ceiling; this is the most likely CTC baseline to produce usable Adja transcriptions in the 40-60% CER range.

## Known follow-ups already queued
- **C3v3-s42**: MMS-1B + French adapter, `patience=20`, `epochs=50`, seed 42. Already in the registry as `queued` — waiting on HPC slot.
