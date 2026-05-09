# Audio Audit Findings

Audit run: 2026-05-09

Compared:

- canonical Orpheus-derived dataset: `JosueG/adja-speech-asr-tts`,
  config `adja_speech_orpheus_48khz`, split `train`
- processed derivative: `JosueG/adja-tts-mms-ready`, split `train`

Local export folder:

- `/private/tmp/adja-audio-audit-compare`

## Findings

The first five Orpheus rows decode as 48 kHz mono arrays with integer-like
amplitudes. Raw peaks range from about `26980` to `65536`, so WAV export must
divide by `65536` before writing PCM. After that scaling, the exported WAV files
are valid 48 kHz mono PCM files and should be listenable.

The first five MMS-ready rows decode as float audio already in `[-1, 1]`, but
every sampled row peaks at exactly `1.0` or near it, and RMS is extremely high:
about `0.87` to `0.998`. That is a strong saturation/clipping signature and is
consistent with the manual report that MMS-ready playback sounded like noise.

## Current Interpretation

The failure is unlikely to be ordinary 48 kHz -> 16 kHz resampling by itself.
The more likely issue is a bad materialization or normalization path in the
MMS-ready derivative: array range interpretation, clipping/saturation, codec
reconstruction, or upload conversion.

Important caveat: the compared row indices are not guaranteed to be matching
utterances because MMS-ready is a processed train/dev/test derivative and
Orpheus is the unsplit source corpus. The diagnosis is based on amplitude
statistics and exported audio shape, not exact sentence-pair alignment.

## Release Decision

Keep `JosueG/adja-speech-asr-tts` as the canonical public speech dataset and
keep `JosueG/adja-tts-mms-ready` as provenance only.
