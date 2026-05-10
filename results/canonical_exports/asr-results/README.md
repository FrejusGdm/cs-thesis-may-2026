---
language:
- adj
- fr
license: other
task_categories:
- automatic-speech-recognition
pretty_name: CS Thesis May 2026 ASR Results
tags:
- adja
- asr
- speech-recognition
- low-resource
- cs-thesis-may-2026
---

# CS Thesis May 2026 ASR Results

This dataset repo is the canonical public ASR result archive for the May 2026
Adja thesis release. It contains reports, metrics, and selected predictions
only. It intentionally does not contain checkpoints, raw private audio, or
active experiment dependency folders.

## Contents

- `reports/`: ASR experiment reports, conclusions, progress summaries, and
  pipeline notes.
- `reports/experiments/`: per-experiment report folders for C/E tracks.
- `predictions/`: selected public decode samples and qualitative ASR outputs.
- `manifests/`: export manifest with source paths and release revision.

## Best Public ASR Artifacts

- Primary ASR model: `JosueG/wav2vec2-xlsr-adja-c4v2`
- Complementary ASR model: `JosueG/whisper-ewe-adja-e4v4`
- Best measured ASR result: `Omni_AUDIOFIX2_LLM7B_20260509_M0`, test
  CER `19.86`, test WER `62.28`, exported under
  `reports/omni_audiofix2_20260509/`
- Canonical speech dataset: `JosueG/adja-speech-asr-tts`
- GitHub release: `FrejusGdm/cs-thesis-may-2026`

## Provenance

The old mixed `JosueG/adja-asr-results` repo remains provenance for checkpoints
and active experiments. This export is the public reporting layer.
