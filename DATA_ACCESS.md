# Data Access

Large datasets are hosted on Hugging Face, not committed directly to this Git
repository.

## Existing Canonical MT Dataset

- `JosueG/french-adja-parallel-corpus`
- Status: existing gated public canonical MT dataset
- Release behavior: reference only; do not mutate

## Thesis Speech Dataset

The release speech dataset should be duplicated into a new canonical thesis
dataset repo from `JosueG/adja-tts-mms-ready`. The source repo is read-only for
this release.

The speech component should be described as an Adja ASR dataset that is also
usable for TTS because it contains paired text/audio and a small speaker set.

## Future MT Additions

Additional French-Adja translated sentences from other translators should be
added as a separate component, not merged destructively into the existing
canonical MT corpus.
