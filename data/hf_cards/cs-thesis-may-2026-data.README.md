---
configs:
- config_name: speech_adja_asr_tts
  data_files:
  - split: train
    path: speech_adja_asr_tts/train-*
  - split: dev
    path: speech_adja_asr_tts/dev-*
  - split: test
    path: speech_adja_asr_tts/test-*
language:
- adj
- fr
license: other
task_categories:
- automatic-speech-recognition
- text-to-speech
pretty_name: CS Thesis May 2026 Data
---

# CS Thesis May 2026 Data

This is the canonical thesis data bundle for the May 2026 French-Adja machine
translation and Adja speech experiments.

## Components

### `speech_adja_asr_tts`

This component duplicates the speech dataset from `JosueG/adja-tts-mms-ready`.
The source repository is treated as a read-only provenance input; this dataset
repo is the public canonical thesis duplicate.

The component is primarily an Adja ASR dataset, and it is also usable for TTS
experiments because each row contains paired text and audio with speaker IDs.

Splits:

- `train`: 1,277 rows
- `dev`: 160 rows
- `test`: 160 rows

Schema:

- `text`: normalized transcription text
- `audio`: speech audio
- `speaker_id`: speaker identifier

## Related Data

- Existing canonical MT dataset: `JosueG/french-adja-parallel-corpus`
- Future MT part-two data should be added as a separate component instead of
  mutating the existing MT corpus.

## Release Policy

Existing experiment dependency repositories are not renamed, moved, deleted, or
modified by this release. Canonical thesis artifacts are created as new repos or
new exports so active training runs remain stable.

## Citation

If you use this dataset, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*.
