# AT1 — Audio-LM pretraining: CSM backbone on unlabeled Ewe

Last updated: 2026-04-26

---

## What AT1 is

AT1 is a 3-stage pipeline that adapts the AudioLM / VALL-E paradigm
(Borsos et al. 2022; Wang et al. 2023) to low-resource Gbe-family TTS:

```
Stage A  ── 183k unlabeled Ewe clips ──────────────────────────────
           Encode audio → Mimi codec tokens (no text).
           Train CSM LM backbone on next-token prediction over the
           flattened code stream. No text labels required.
           The model learns the acoustic prior for Gbe-family speech.

Stage B  ── ~1,215 labeled Ewe TTS + ~1,600 labeled Adja TTS ──────
           Supervised (text, audio) fine-tune on top of the Stage A
           checkpoint. CSM re-learns text conditioning with a Gbe-
           family acoustic prior already baked in.

Inference ─ generate Adja speech from Adja text ───────────────────
```

Compared to the T1/T2 Ewe-bridge experiments (Stage 1 Ewe → Stage 2 Adja
transfer), AT1 adds the self-supervised Stage A pretraining step and uses
183x more Ewe audio during Stage A. The hypothesis is that a model that has
seen 100-200h of unlabeled Gbe speech will generalize better in Stage B
than one that starts cold from the English-centric CSM checkpoint.

---

## Why 183k unlabeled Ewe clips

- The WaxalNLP `ewe_asr` unlabeled split contains 183k utterances of
  real Ewe speech (estimated 100-200h total).
- Ewe is the closest well-resourced language to Adja: both are Gbe-family
  (Niger-Congo), share phonological features including tone, and have
  overlapping vocabulary.
- 183k utterances is well below AudioLM's original pretraining scale
  (60k h), but is enough to shift the codec-token distribution toward
  Gbe-family speech patterns, based on evidence from T1-stage1 showing
  that even 1,215 Ewe TTS clips produced intelligible Ewe output.

---

## Why AudioLM / VALL-E architecture

Standard TTS fine-tuning minimizes reconstruction loss on (text, audio) pairs
directly. With only ~6h of labeled (text, audio) data, the model has to
simultaneously learn:
  - the phonology and tone of a new language family
  - how text tokens map to codec tokens in a new script/orthography
  - natural prosody and speaker identity

AudioLM (Borsos et al., arxiv.org/abs/2209.03143) showed that a language model
pre-trained on audio codec tokens alone — with no text — learns a strong
acoustic prior. When text conditioning is added afterward, the supervised stage
converges faster and produces more natural output because the acoustic prior
is already in place.

VALL-E (Wang et al., arxiv.org/abs/2301.02111) applied this to zero-shot TTS:
a codec LM trained on 60k h of English can synthesize new speakers from a
3-second prompt. AT1 is a scaled-down version of this for a low-resource
Gbe-family language.

---

## Prerequisites — run M0 first

Before submitting AT1, run the Mimi reconstruction gate test locally:

```bash
HF_TOKEN=<token> python scripts/sagemaker_jobs/M0_mimi_reconstruction_test.py
```

Gate thresholds (spectral convergence SC):
- SC < 0.30 — excellent; proceed to AT1
- SC 0.30–0.60 — acceptable; proceed with caution
- SC > 0.60 — Mimi is likely dropping tonal content; fall back to
              `audio_lm_orpheus_ewe.py` (SNAC codec) on HPC

If M0 fails, AT1 is dead — the LM will be trained on a corrupted codec
representation and Stage B will not recover.

---

## Cost breakdown (AT1 full run)

| Stage | Instance | Est. runtime | On-demand | Spot (~70%) |
|---|---|---|---|---|
| A — pretrain (3 epochs, 183k utts) | ml.p3.8xlarge | ~28h | $411 | ~$288 |
| B — fine-tune (20 epochs, ~17k utts) | ml.p3.8xlarge | ~6h | $88 | ~$62 |
| Total | | ~34h | ~$499 | ~$350 |

Spot is the default in `launch.py`. AT1 on spot costs approximately $350.

---

## Submitting

```bash
# Smoke test first — confirms script imports, CUDA, ~$1
python scripts/sagemaker_jobs/launch.py --experiment AT1 --smoke

# Full run on spot
python scripts/sagemaker_jobs/launch.py --experiment AT1
```

See `best-TTS-African-Languages/setup/sagemaker-setup.md` for IAM role,
AWS credentials, and result retrieval.

---

## Success criteria

1. Stage A train loss decreases monotonically; no NaN or explosion.
2. Stage B best_eval_loss is lower than the T1-stage2 / T2-stage2 baselines.
3. Generated Adja audio is intelligible to a native speaker (listening test).

---

## Risks and mitigations

**Catastrophic forgetting in Stage B.**
After training on pure audio tokens in Stage A, the model has pushed text
conditioning out of its distribution. Stage B re-introduces text, but the
LM may never fully recover.
Mitigation: Stage B uses cosine LR schedule with 100 warmup steps and early
stopping (patience=5 on eval loss). If Stage B loss plateaus > Stage A cold
start, Stage A pretraining is net-negative for this model size / data volume.

**Codec fidelity (tone).**
Mimi operates at 12.5 Hz frame rate / 8 codebooks. If the codec quantization
loses F0 contour information for tonal Gbe languages, Stage A learns a
tonally-impoverished acoustic prior.
Mitigation: M0 gate test. If SC > 0.60 on Adja samples, switch to the SNAC
(Orpheus) track — SNAC uses 3 codebooks at 75 Hz and may preserve tone better.

**Spot interruption.**
A 34h spot job can be interrupted mid-run.
Mitigation: save_strategy="epoch" in Stage A (saves every epoch, ~9h).
Consider adding checkpoint_s3_uri to the estimator in launch.py for
SageMaker managed checkpointing.

**Data volume.**
183k unlabeled Ewe clips is 100-200h — far below AudioLM's 60k h pretraining.
Stage A may not shift the acoustic prior enough to matter.
This is the core speculative assumption. Stage A is still worth running if
M0 passes, since the marginal cost vs T1/T2 is ~$350 for potentially +1-2
listening-test points.

---

## Related experiments

- **T1** (`scripts/hf_jobs/T1_csm_ewe_adja_stage2.py`) — Ewe-bridge without
  audio-LM pretraining. AT1 Stage B should beat T1 Stage 2 if pretraining helps.
- **T2** (`scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py`) — Orpheus 3B variant.
- **M0** (`scripts/sagemaker_jobs/M0_mimi_reconstruction_test.py`) — gate test.

---

## References

- Borsos et al. (2022). **AudioLM: a Language Modeling Approach to Audio Generation.**
  arxiv.org/abs/2209.03143
- Wang et al. (2023). **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E).**
  arxiv.org/abs/2301.02111
- Defossez et al. (2024). **Mimi codec (from Moshi).**
  arxiv.org/abs/2410.00037
- Sesame AI Labs. **CSM (Conversational Speech Model).**
  github.com/SesameAILabs/csm
