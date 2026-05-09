# Hugging Face Dataset Map

## Existing Canonical MT Dataset

`JosueG/french-adja-parallel-corpus`

This is the existing canonical French-Adja parallel corpus. It remains unchanged
by this release.

## Canonical Adja Speech Dataset

Target repo: `JosueG/adja-speech-asr-tts`

Current component:

- `adja_speech_orpheus_48khz`: duplicated speech dataset from `JosueG/adja-tts-orpheus`.
- `manifests/`: source revisions, checksums, row counts, schemas, and access policy.

The source repos are read-only inputs. The retired `JosueG/cs-thesis-may-2026-data`
dataset was created from `JosueG/adja-tts-mms-ready` and is not the canonical
speech release.
