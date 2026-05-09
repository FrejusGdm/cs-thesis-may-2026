# C4: Conclusion

## Verdict
**collapsed** (CTC blank collapse)

## What happened
XLS-R 300M fine-tuned with our CTC head dropped its training loss from 22.72 to 4.74 over 6 epochs while dev CER stayed nailed at 100.00% throughout. The model minimized CTC loss by learning "always emit blank" — its argmax output was the empty string for every one of the 160 dev utterances on every epoch. Training was stopped after 6 epochs because there was no signal to save.

## Why
Root cause identified: **our manual CTC loss implementation is degenerate on this setup**. Specifically, the blank token dominated the logit distribution from initialization, and the gradient through the manual `ctc_loss` we wrote pushed the model into a blank-emitting local minimum it couldn't escape. Secondary contributors:

- XLS-R 300M was pretrained on contrastive-learned audio representations without any CTC-style supervised head; the `lm_head` had to be learned from scratch against a small vocab (115) and a small dataset (1277 utterances).
- Our patience=20 did not help because the model never left the collapsed region.
- Same pattern seen in C5 (wav2vec 2.0-large-960h) — this is a bug in our loss pipeline, not a one-off model issue.

## What to do next
Investigated. Root cause: manual CTC loss. Fix: use the HuggingFace `Wav2Vec2ForCTC` built-in CTC loss (`model(input_values, labels=...)` which internally computes the correct `torch.nn.functional.ctc_loss`). **C4v2 already queued** in the registry with the built-in loss + patience=20.

## Known follow-ups already queued
- **C4v2-s42**: XLS-R 300M rerun with the model's built-in CTC loss, `patience=20`, 50 epochs. Already in the registry as `queued` — this is the fix and will likely determine whether CTC on XLS-R is viable for Adja at all.
