---
language:
- adj
license: other
task_categories:
- text-to-speech
pretty_name: CS Thesis May 2026 TTS Results
tags:
- adja
- tts
- speech-synthesis
- reverse-wer
- low-resource
- cs-thesis-may-2026
---

# CS Thesis May 2026 TTS Results

This dataset repo is the canonical public TTS result archive for the May 2026
Adja thesis release. It contains reports, reverse-WER metrics, and selected
configuration summaries only. It intentionally does not contain checkpoints,
large generated-audio dumps, raw private audio, or active experiment dependency
folders.

## Contents

- `reports/`: TTS comparison reports, reverse-WER summary, pipeline notes, and
  run ledger.
- `metrics/`: reverse-WER JSON files and small public metric/config files.
- `manifests/`: export manifest with source paths and release revision.

## Best Public TTS Artifact

- Primary TTS model: `JosueG/spark-tts-adja-t3`
- Canonical speech dataset: `JosueG/adja-speech-asr-tts`
- ASR judges: `JosueG/wav2vec2-xlsr-adja-c4v2` and
  `JosueG/whisper-ewe-adja-e4v4`
- GitHub release: `FrejusGdm/cs-thesis-may-2026`

## Provenance

The old mixed `JosueG/adja-tts-results` and `JosueG/adja-tts-checkpoints` repos
remain provenance for checkpoints, generated audio, and active experiments.
This export is the public reporting layer.
