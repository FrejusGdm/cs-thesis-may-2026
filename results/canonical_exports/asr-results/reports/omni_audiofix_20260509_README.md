# OmniASR Audio-Fix Rerun Log

Date: 2026-05-09

This log tracks the full OmniASR fine-tuning matrix resubmitted after the HF parquet audio audit.

## Reason

`JosueG/adja-tts-orpheus` stores audio arrays as int-like PCM magnitudes inside float arrays. Prior Omni manifest materialization wrote those arrays directly to WAV, so the WAVs were clipped before training. The reruns below use pre-WAV range scaling in both Omni launchers:

- `scripts/hf_jobs/omni_asr_finetune.py`
- `scripts/hf_jobs/omni_asr_finetune_recipe.py`

## Submitted Jobs

| Exp ID | Model | Job ID | Hardware | Status at submission |
|--------|-------|--------|----------|----------------------|
| `Omni_AUDIOFIX_CTC300M_20260509_M2` | `omniASR_CTC_300M_v2` | `69ff5af6317220dbbd1a71ab` | `h200` | running |
| `Omni_AUDIOFIX_LLM300M_20260509_M0` | `omniASR_LLM_300M_v2` | `69ff5af6317220dbbd1a71ad` | `h200` | running |
| `Omni_AUDIOFIX_CTC3B_20260509_M2` | `omniASR_CTC_3B_v2` | `69ff5af5317220dbbd1a71a9` | `h200` | running |
| `Omni_AUDIOFIX_LLM3B_20260509_M0` | `omniASR_LLM_3B_v2` | `69ff5af5aff1cd33e8f31fc8` | `h200` | running |
| `Omni_AUDIOFIX_CTC7B_20260509_M2` | `omniASR_CTC_7B_v2` | `69ff5893aff1cd33e8f31fa7` | `h200` | running |
| `Omni_AUDIOFIX_LLM7B_20260509_M0` | `omniASR_LLM_7B_v2` | `69ff598eaff1cd33e8f31fb2` | `h200x4` | running |

## Scope Boundary

These are fine-tuning reruns only. `Omni_ZS_*` and `Omni_ICL_*` are inference-only/prompting baselines and are not part of this training rerun matrix.

## Monitoring

```bash
hf jobs inspect 69ff5af6317220dbbd1a71ab
hf jobs inspect 69ff5af6317220dbbd1a71ad
hf jobs inspect 69ff5af5317220dbbd1a71a9
hf jobs inspect 69ff5af5aff1cd33e8f31fc8
hf jobs inspect 69ff5893aff1cd33e8f31fa7
hf jobs inspect 69ff598eaff1cd33e8f31fb2
```
