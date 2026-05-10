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
| `Omni_AUDIOFIX2_CTC300M_20260509_M2` | `omniASR_CTC_300M_v2` | `69ff5bfaaff1cd33e8f31fe6` | `h200` | running |
| `Omni_AUDIOFIX2_LLM300M_20260509_M0` | `omniASR_LLM_300M_v2` | `69ff5bf9aff1cd33e8f31fe0` | `h200` | running |
| `Omni_AUDIOFIX2_CTC3B_20260509_M2` | `omniASR_CTC_3B_v2` | `69ff5bfa317220dbbd1a71b5` | `h200` | running |
| `Omni_AUDIOFIX2_LLM3B_20260509_M0` | `omniASR_LLM_3B_v2` | `69ff5bfaaff1cd33e8f31fe4` | `h200` | running |
| `Omni_AUDIOFIX2_CTC7B_20260509_M2` | `omniASR_CTC_7B_v2` | `69ff5bfaaff1cd33e8f31fe2` | `h200` | running |
| `Omni_AUDIOFIX2_LLM7B_20260509_M0` | `omniASR_LLM_7B_v2` | `69ff5bfaaff1cd33e8f31fe8` | `h200x4` | running |

## Canceled First Wave

The first `Omni_AUDIOFIX_*` submission wave was canceled before completion after the launcher was patched to remove secret env vars before the fairseq trainer starts. Replacement `Omni_AUDIOFIX2_*` jobs above are the active runs.

## Scope Boundary

These are fine-tuning reruns only. `Omni_ZS_*` and `Omni_ICL_*` are inference-only/prompting baselines and are not part of this training rerun matrix.

## Monitoring

```bash
hf jobs inspect 69ff5bfaaff1cd33e8f31fe6
hf jobs inspect 69ff5bf9aff1cd33e8f31fe0
hf jobs inspect 69ff5bfa317220dbbd1a71b5
hf jobs inspect 69ff5bfaaff1cd33e8f31fe4
hf jobs inspect 69ff5bfaaff1cd33e8f31fe2
hf jobs inspect 69ff5bfaaff1cd33e8f31fe8
```
