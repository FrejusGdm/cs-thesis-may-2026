# Data Access

Large datasets are hosted on Hugging Face, not committed directly to this Git
repository.

## Existing Canonical MT Dataset

- `JosueG/french-adja-parallel-corpus`
- Status: existing gated public canonical MT dataset
- Release behavior: reference only; do not mutate

## Thesis Speech Dataset

The release speech dataset is `JosueG/adja-speech-asr-tts`, duplicated from
the Orpheus source repo `JosueG/adja-tts-orpheus`. The source repo is read-only
for this release.

The speech component should be described as an Adja ASR dataset that is also
usable for TTS because it contains paired text/audio. The `adja-tts-mms-ready`
repo is a processed train/dev/test derivative and is documented as provenance,
not as the canonical public source, after audio quality review.

## Future MT Additions

Additional French-Adja translated sentences from other translators should be
added as a separate component, not merged destructively into the existing
canonical MT corpus.
