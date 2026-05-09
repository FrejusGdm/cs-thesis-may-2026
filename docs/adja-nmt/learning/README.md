# Learning — study notes for the Adja speech project

Last updated: **2026-04-21**

This is where I (Josue, CS senior at Dartmouth) keep my own learning materials
so I actually understand what we are running and why, not just run it. Built
alongside the experiments so that a future-me (or a collaborator) can catch up.

## If you are brand new to this project

Read in this order — don't skip ahead, each file builds on the last:

1. `reading-order.md` — the full curriculum with time estimates and prereqs.
2. `concepts/01-what-is-a-tokenizer.md` — the layer that blocked CSM and Orpheus.
3. `concepts/02-what-is-an-audio-codec.md` — what Mimi and SNAC actually do.
4. `concepts/03-three-layer-tts.md` — how tokenizer / LM / codec fit together.
5. `concepts/04-ctc-vs-attention-vs-rnnt.md` — ASR decoding families.
6. `concepts/05-ssl-pretraining-wav2vec.md` — wav2vec 2.0 / XLS-R / MMS.
7. `concepts/06-lora-and-peft.md` — parameter-efficient fine-tuning.
8. `concepts/07-tonal-languages-for-ML.md` — why Gbe-family matters here.
9. `concepts/08-low-resource-transfer.md` — the thesis of this project.

Then move to `experiments-explained/` for dated write-ups of the actual runs.

## If you want to understand why a specific experiment was run

Every submitted experiment has a file in `experiments-explained/` telling the
story: what we were testing, what the outcome was, and what to take away.

- `T1-csm-why-it-failed.md` — the first big negative result.
- `T3-spark-why-it-worked.md` — the control that flipped our theory.
- `T2-orpheus-multilingual-variants.md` — comparing EN / zh / fr bases.
- `E4-whisper-ewe-adja-bridge.md` — best ASR so far and how we got there.
- `ssl-pretraining-on-unlabeled-ewe.md` — why 183k utterances of Ewe matter.
- `audio-lm-what-and-why.md` — the speculative track using unlabeled audio.

## If you want to read the primary sources

`papers/README.md` is an annotated bibliography — papers grouped by topic,
with notes on which sections matter for this project.

## Cross-links

- Main `CLAUDE.md` at repo root — project rules and conventions.
- Master experiment plan: `../experiments/asr-tts-getting-right-2026-04-21.md`.
- Current experiment status: `../experiments/registry.md`.
- HPC side: `../hpc/README.md`.
- Session logs: `../session-logs/`.

## How this folder is maintained

- New concept: add a `concepts/NN-topic.md` file (NN = next number).
- New paper: add to `papers/<area>/` with a one-paragraph summary.
- New experiment launched: create `experiments-explained/<exp>.md` with
  **hypothesis, setup, expected outcome** BEFORE the run finishes. Fill in the
  **actual outcome** when results come back. Keep both so you can learn from
  surprises.
