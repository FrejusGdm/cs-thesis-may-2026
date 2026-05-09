# C5: Conclusion

## Verdict
**collapsed** (CTC blank collapse)

## What happened
wav2vec 2.0-large-960h fine-tuned with our CTC loss collapsed to blank-emission immediately. Training loss fell from 16.44 to 3.51 in 6 epochs; dev CER/WER stayed pinned at 100% throughout. The model produced empty strings on every evaluation utterance on every epoch. Identical failure profile to C4.

## Why
Two independent causes both pushed the same direction:

1. **Same manual CTC loss bug as C4** — the hand-written `ctc_loss` in our training script rewards blank emission; once the model hits the all-blank basin in epoch 1 it has no gradient signal to leave. This is the primary root cause.
2. **English-only pretraining is a poor fit for Adja**. Unlike XLS-R 300M (128 languages) or MMS (1100+), wav2vec 2.0-large-960h was pretrained only on English LibriSpeech. Its acoustic representations have likely never seen Adja's front-mid vowels (ɛ, ɔ), velar nasal (ŋ), implosive (ɖ), or tonal contour inventory. Even with a working CTC loss, transfer would be weak.

## What to do next
Investigated. Root causes: (1) buggy CTC loss (shared with C4); (2) mismatched pretraining language. Decision: **skip the rerun for now**. Priority goes to C4v2 (XLS-R 300M with fixed CTC loss, multilingual pretraining) because if that works, English-only wav2vec 2.0 is unlikely to match it and is of lower research interest. Revisit C5 only if:

- C4v2 clearly benefits from the fixed CTC loss, AND
- We want a data point specifically measuring "English-pretrained multilingual transfer" as a negative control in the paper.

## Known follow-ups already queued
- **Not re-submitting C5 yet**. Lower priority than C4v2 (XLS-R with fixed CTC loss) and C3v3 (MMS with patience=20), both already queued. Revisit after those complete.
