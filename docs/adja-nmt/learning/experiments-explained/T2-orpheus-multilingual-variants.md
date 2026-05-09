# T2 (Orpheus) — comparing multilingual variants for Adja

Last updated: **2026-04-21**

## The setup

Orpheus ships in 7 language variants, all using the same SNAC codec and same
Llama-3.2-3B backbone. Only the fine-tune data differs:

- `canopylabs/orpheus-3b-0.1-ft` (English)
- `canopylabs/3b-zh-ft-research_release` (Mandarin, **tonal**)
- `canopylabs/3b-fr-ft-research_release` (French)
- `canopylabs/3b-de-ft-research_release` (German)
- `3b-hi-ft-research_release`, `3b-ko-ft-research_release`, `3b-es_it-ft-research_release`

## What we ran

Two strategies, 3 bases each:

### Strategy A — direct Adja fine-tune (control)
- T2 English → Adja (done earlier, got dev loss 5.42, noise output)
- T2 Chinese (zh) → Adja (submitted 2026-04-21, Job 69e7c944)
- T2 French (fr) → Adja (not yet — can be run later if we want the 3rd control)

### Strategy B — Ewe bridge two-stage
- T2 English → Ewe → Adja (Job 69e7d107 Stage 1 on 2026-04-21)
- T2 Chinese (zh) → Ewe → Adja (Job 69e7d1d1)
- T2 French (fr) → Ewe → Adja (Job 69e7d1dc)

All three Stage 1 jobs run in parallel on L40S. Stage 2 submissions wait
until Stage 1 checkpoints push to `JosueG/adja-tts-checkpoints/`.

## Hypotheses being tested

1. **Does tonal prior transfer?** Mandarin has 4 tones; Adja/Ewe are tonal.
   If the zh base beats en and fr at the Adja task, the LM's tonal priors
   transferred across unrelated language families.
2. **Does related-language pretraining (Ewe) add value on top of the base?**
   Compare T2-direct-{en,zh} vs T2-ewe-bridge-{en,zh}. If the bridge wins
   by a larger margin for en than for zh, Ewe supplies the tonal info the
   en base lacks.
3. **Does any of this beat Spark TTS?** Spark is our current best. Orpheus
   with either a tonal prior, a bridge, or a tokfix — ideally all three —
   should eventually match or beat Spark's intelligibility.

## What we're NOT testing yet

- German, Hindi, Korean, Spanish/Italian bases. Low ROI — they don't have
  tonal priors and aren't closer to Adja than English.
- Full fine-tune (vs LoRA r=64). We did this for English (fullft reached
  dev loss 5.42 at 1.88 epochs, then diverged). Not worth repeating for
  every base unless LoRA shows promising results first.

## Expected outcomes (guesses, not predictions)

- All 3 Stage 1 jobs converge similarly on Ewe (1215 utts is plenty).
- Stage 2 on Adja likely shows zh > en ≈ fr, with a bridge-vs-direct
  improvement of ~1.0-1.5 dev-loss points.
- Intelligibility threshold probably still not reached without tokfix —
  Llama BPE byte fragmentation issue persists regardless of base language
  because all variants share the same tokenizer.

## What results mean for the thesis / paper

- If bridge > direct: we have a reusable recipe for any Gbe-family TTS
  (and by extension, low-resource tonal TTS).
- If zh > en: we have evidence that tonal-to-tonal LM transfer works even
  cross-family, a novel claim.
- If tokfix + bridge + zh gives the best result: we have a clean story
  showing each ingredient's contribution.

## Files

- Stage 1 script (parameterized): `../../scripts/hf_jobs/T2_orpheus_ewe_stage1.py`
- Stage 2 script: `../../scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py`
- Direct zh→Adja: `../../scripts/hf_jobs/T2_orpheus_zh_adja.py`
- Tokfix: `../../scripts/hf_jobs/T2_orpheus_tokfix.py`
- Related concept: `../concepts/07-tonal-languages-for-ML.md`
