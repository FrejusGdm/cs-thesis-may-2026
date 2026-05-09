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
- Replacement dataset keeps the Orpheus source representation: train split,
  1,597 rows, `text` and `audio`, 48 kHz source audio arrays.

## Standing Rule

When an artifact is public-facing, prefer clear task-specific names over thesis
bundle names. The GitHub repo can be the broad thesis archive; datasets should
say what they actually are.
