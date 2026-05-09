# Hugging Face Results Map

## Current Source Repos

- `JosueG/adja-asr-results`: ASR result/checkpoint dump, currently private.
- `JosueG/adja-tts-results`: TTS result/checkpoint/audio dump, exists in both model and dataset forms.
- `JosueG/adja-mt-results-private`: MT result/checkpoint dump, currently private.

## Planned Canonical Result Repos

- `JosueG/cs-thesis-may-2026-asr-results`
- `JosueG/cs-thesis-may-2026-tts-results`
- Optional: `JosueG/cs-thesis-may-2026-mt-results`

These should be new repos created by export/duplication. Existing result repos
must remain unchanged because active experiments may still depend on them.

Recommended structure for each canonical result repo:

- `README.md`: task summary, metrics, source models, source datasets.
- `reports/`: human-readable reports.
- `metrics/`: JSON/CSV metrics with schemas.
- `predictions/`: selected predictions or decode samples safe for release.
- `artifacts/`: small non-sensitive generated examples.
- `manifests/`: source repo IDs, revisions, checksums, export date.
