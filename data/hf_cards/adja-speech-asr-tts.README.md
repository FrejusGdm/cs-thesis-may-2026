---
configs:
- config_name: adja_speech_orpheus_48khz
  data_files:
  - split: train
    path: adja_speech_orpheus_48khz/train-*
language:
- adj
- fr
license: other
task_categories:
- automatic-speech-recognition
- text-to-speech
pretty_name: Adja Speech Dataset for ASR and TTS
---

# Adja Speech Dataset for ASR and TTS

This is the canonical public Adja speech dataset for the May 2026 thesis
release. It is intended for automatic speech recognition, text-to-speech, and
speech pipeline experiments.

## Components

### `adja_speech_orpheus_48khz`

This component duplicates the Orpheus speech source from `JosueG/adja-tts-orpheus`.
The source repository is treated as read-only provenance; this dataset repo is
the public canonical release surface.

The component is useful for ASR because each row pairs Adja text with speech,
and useful for TTS because the same pairs can train or evaluate speech
synthesis models.

Splits:

- `train`: 1,597 rows

Schema:

- `text`: normalized transcription text
- `audio`: speech audio stored from the Orpheus source at 48 kHz

## Provenance And Quality Note

The processed derivative `JosueG/adja-tts-mms-ready` is not used as the
canonical public source after release review found bad/noisy playback on
Hugging Face. This repo goes back to the Orpheus source data instead.

No existing experiment dependency repo is renamed, moved, or edited by this
release.

## Related Thesis Artifacts

- Code and result archive: `FrejusGdm/cs-thesis-may-2026`
- Existing canonical MT dataset: `JosueG/french-adja-parallel-corpus`

## Citation

If you use this dataset, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*.
