# Omni AUDIOFIX2 ASR Results

Date: 2026-05-09

These are the completed OmniASR fine-tuning results after the audio-range fix.
The fix matters because the Orpheus-derived HF audio arrays are stored as
integer-like PCM magnitudes inside float arrays; writing them directly to WAV
clips the training audio. The AUDIOFIX2 jobs used range scaling before WAV
materialization.

## Headline

The best clean main run is **`Omni_AUDIOFIX2_LLM7B_20260509_M0`**, with
**test CER 19.86** and **test WER 62.28**. This is now the best measured ASR
result in the release. It is a result/checkpoint provenance item, not yet a
standalone public model repo.

The separate `PUBLISH_R3` rerun has slightly lower CER (`19.84`) but higher WER
(`62.74`), so the main LLM7B run remains the headline result for now.

## Result Table

| Run | Job | Status | Test WER | Test CER | Private result folder |
| --- | --- | --- | --- | --- | --- |
| Omni_AUDIOFIX2_LLM7B_20260509_M0 | 69ff5bfaaff1cd33e8f31fe8 | completed | 62.28 | 19.86 | Omni_AUDIOFIX2_LLM7B_20260509_M0_llm_1778343264 |
| Omni_AUDIOFIX2_LLM7B_PUBLISH_20260509_R3 | not recorded in release notes | completed | 62.74 | 19.84 | Omni_AUDIOFIX2_LLM7B_PUBLISH_20260509_R3_llm_1778370222 |
| Omni_AUDIOFIX2_LLM3B_20260509_M0 | 69ff5bfaaff1cd33e8f31fe4 | completed | 69.08 | 22.91 | Omni_AUDIOFIX2_LLM3B_20260509_M0_llm_1778343024 |
| Omni_AUDIOFIX2_CTC3B_20260509_M2 | 69ff5bfa317220dbbd1a71b5 | completed | 73.26 | 23.79 | Omni_AUDIOFIX2_CTC3B_20260509_M2_ctc_1778343030 |
| Omni_AUDIOFIX2_LLM300M_20260509_M0 | 69ff5bf9aff1cd33e8f31fe0 | completed | 73.44 | 24.78 | Omni_AUDIOFIX2_LLM300M_20260509_M0_llm_1778343022 |
| Omni_AUDIOFIX2_CTC300M_20260509_M2 | 69ff5bfaaff1cd33e8f31fe6 | completed | 77.9 | 26.98 | Omni_AUDIOFIX2_CTC300M_20260509_M2_ctc_1778343024 |
| Omni_AUDIOFIX2_CTC7B_20260509_M2 | 69ff5bfaaff1cd33e8f31fe2 | error_cuda_oom | — | — | not uploaded |

## CTC7B Failure

`Omni_AUDIOFIX2_CTC7B_20260509_M2` failed on single-H200 CUDA OOM. The run hit
the H200 memory ceiling and did not upload an eval folder. It likely needs a
smaller recipe, more aggressive memory reduction, or multi-GPU/FSDP treatment.

## Public Release Policy

This public release exports reports and compact metrics only. The old
`JosueG/adja-asr-results` repo remains private provenance for checkpoints and
active experiments.
