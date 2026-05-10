# Release Operations Log

This log records public-release operations that change GitHub or Hugging Face
artifacts.

## 2026-05-09: Initial GitHub Release

- Created and pushed `FrejusGdm/cs-thesis-may-2026`.
- Exported publishable code, documentation, model indexes, and result summaries
  from the private thesis workspaces.
- Treated existing GitHub and Hugging Face repos as read-only inputs unless a
  new release repo was being created.

## 2026-05-09: Speech Dataset Correction

- Initial public dataset `JosueG/cs-thesis-may-2026-data` was created from
  `JosueG/adja-tts-mms-ready`.
- Manual Hugging Face playback review found that this processed derivative
  sounded noisy/incorrect.
- Correction: retire/delete `JosueG/cs-thesis-may-2026-data` and publish the
  Orpheus source instead as `JosueG/adja-speech-asr-tts`.
- Deletion completed: `JosueG/cs-thesis-may-2026-data` no longer resolves on
  Hugging Face.
- Replacement published: `JosueG/adja-speech-asr-tts` at revision
  `db4da3824020752eb9ee62d80f0d8d483785e414`.
- Replacement source: `JosueG/adja-tts-orpheus` at revision
  `5b77536822c3cb173d83214204f59d6df6c12529`.
- Replacement dataset keeps the Orpheus source representation: one unsplit
  1,597-row corpus exposed as the single Hugging Face `train` partition,
  with `text`, `audio`, and 48 kHz source audio arrays.

## Standing Rule

When an artifact is public-facing, prefer clear task-specific names over thesis
bundle names. The GitHub repo can be the broad thesis archive; datasets should
say what they actually are.

## 2026-05-09: Lightweight Model And Results Cleanup

- Prepared thesis-final model cards for the two NLLB MT models, C4v2 XLS-R
  ASR, E4v4 Whisper-Ewe ASR, and Spark T3 TTS.
- Selected the two NLLB repos and Spark T3 repo for public visibility.
- Kept ASR repos public and improved their cards instead of creating duplicate
  model repos.
- Added `results/thesis-results-map.md` as the public navigation layer for
  best models, experiment archive, datasets, result dumps, checkpoints, and
  generated audio.
- Added an Orpheus audio audit helper to export WAV examples and diagnose
  whether MMS-ready noise came from resampling, dtype/range interpretation,
  codec reconstruction, or dataset materialization.
- Created Hugging Face collection:
  `JosueG/cs-thesis-may-2026-69ff55e45e0d5b0eb7fa4344`.

## 2026-05-09: Canonical Result Reporting Exports

- Created ASR result export source under `results/canonical_exports/asr-results`.
- Created TTS result export source under `results/canonical_exports/tts-results`.
- Published the exports as Hugging Face dataset repos:
  `JosueG/cs-thesis-may-2026-asr-results` and
  `JosueG/cs-thesis-may-2026-tts-results`.
- Added both result repos to the thesis Hugging Face collection.
- Export policy: reports, metrics, selected predictions, and manifests only;
  no checkpoints, raw private audio, or old mixed experiment dependency folders.

## 2026-05-09: Omni AUDIOFIX2 Result Update

- Added completed `Omni_AUDIOFIX2_*` metrics to the ASR result archive.
- New best measured ASR result: `Omni_AUDIOFIX2_LLM7B_20260509_M0`, with test
  CER `19.86` and test WER `62.28`.
- Recorded `Omni_AUDIOFIX2_CTC7B_20260509_M2` as failed due to single-H200 CUDA
  OOM, with no uploaded eval folder.
- Kept C4v2 as the best public deployable ASR model repo until an Omni
  checkpoint is published as a clean standalone model.
