# E4 — Whisper Ewe → Adja bridge (best ASR to date)

Last updated: **2026-04-29** (correction added — original E4 24.90% is logged-only).

> **Correction (2026-04-29):** The original E4 weights were never uploaded to
> Hugging Face — only the metrics file made it to `JosueG/adja-asr-results/E4/`.
> Two reproduction attempts (E4v4 2026-04-17, E4_v5 2026-04-28) plateau at
> CER ~37.18% / 37.35%. The 24.90% number is therefore a logged-only result.
> The deployable Whisper Adja ASR is `JosueG/whisper-ewe-adja-e4v4` at
> CER 37.18%. Working hypothesis: the original E4 trained with the EOS-mask
> hallucination bug we later fixed in `scripts/hf_jobs/whisper_finetune.py`;
> that bug acted as accidental regularization. See `results/comparison.md`
> and `experiments/registry.md` E4_v5 row for the full writeup.

## Background

Whisper-small fine-tuned directly on Adja (the C2 experiment) hit CER 27%
/ WER 74%. Not bad, but we wanted to try starting from a Whisper variant
already fine-tuned on a related language.

Dodzi Raynard had released `dodziraynard/whisper-small-ee`, a Whisper-small
fine-tuned on Ewe. We used this as our E4 base and fine-tuned on Adja.

## The result

**CER 24.90% / WER 73.09%** — our best direct-fine-tune ASR result. Still
best at epoch 50 when we stopped (was still improving); the training run
could have gone longer.

## Why it worked

Ewe and Adja are closest relatives in the Gbe family. They share:
- Phoneme inventory (ɛ, ɔ, ŋ, ɖ)
- Tone system
- General syllable structure

When Whisper was fine-tuned on Ewe, the encoder already learned to map Gbe
phonology to intermediate embeddings. The Adja fine-tune essentially teaches
the decoder to emit slightly different Gbe orthography — a much smaller
adjustment than starting from English-pretrained Whisper.

## What went wrong later (important gotcha)

When we tried to reproduce E4 with a slightly updated training script (E4v2),
we got **CER 76-150%** with hallucination loops in the output. Root cause:
the processor was set to `padding="max_length"` on the raw audio, padding
all inputs to 30 seconds with silence. The decoder learned that silence =
endless output, and at inference time it generated infinite repetitive loops.

Fix (in `scripts/hf_jobs/whisper_finetune.py`): use `padding=True` (pad to
longest in batch) and manually pad mel spectrogram to exactly 3000 frames
after feature extraction. Do NOT pad raw audio. See the long comment in
that file for the detailed gotcha.

## What we did next

1. Added the LM shallow-fusion track (D4) — combine E4's output with a
   character-level Adja language model during beam search.
2. D4_E4v2_lm hit CER 42.9% — big LM win, but on the broken E4v2 AM.
3. D4_C4v2_lm_optuna (LM on XLS-R CTC, not E4) hit **CER 22.67%** — best
   overall ASR result.

## E7 — what we're doing next

Now: `E7_whisper_largev3_ewe_stage1.py` on HF Jobs A100 (1.5B model, 15k
labeled Ewe rows) — same strategy but with a much bigger and better Whisper
variant, and official WaxalNLP data (vs dodziraynard's unofficial model).

On HPC we'll run `hpc/scripts/large/whisper_largev3_ewe_hpc.py` with longer
training (50 epochs vs 20) once we have the cluster set up.

## Files

- Registry: `../../experiments/registry.md` row E4
- Run ledger: `../../results/run-ledger.md` 2026-04-16 E4 entry
- Fine-tune script: `../../scripts/hf_jobs/whisper_finetune.py`
- E7 successor: `../../scripts/hf_jobs/E7_whisper_largev3_ewe_stage1.py`
- Padding gotcha explanation: `../../docs/whisper-training-gotchas.md`
