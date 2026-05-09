# Hugging Face Results Map

This pass does not create new result repos or copy checkpoints. It makes the
best model artifacts public, improves public cards, and adds a single map for
where results live.

## Public Best Artifacts

- MT forward: `JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42`
- MT reverse: `JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42`
- ASR primary: `JosueG/wav2vec2-xlsr-adja-c4v2`
- ASR complementary: `JosueG/whisper-ewe-adja-e4v4`
- TTS primary: `JosueG/spark-tts-adja-t3`

## Current Source Repos

- `JosueG/adja-asr-results`: ASR result/checkpoint dump, private provenance.
- `JosueG/adja-tts-results`: TTS result/checkpoint/audio dump, private provenance
  in both model and dataset forms.
- `JosueG/adja-mt-results-private`: MT result/checkpoint dump, private
  provenance.
- `JosueG/adja-tts-checkpoints`: TTS checkpoint provenance.
- `JosueG/qwen3-adja-results`: Qwen ASR/MT experimental provenance.

## Canonical Result Repos

- `JosueG/cs-thesis-may-2026-asr-results`
- `JosueG/cs-thesis-may-2026-tts-results`
- Optional: `JosueG/cs-thesis-may-2026-mt-results`

The ASR and TTS result repos are public reporting exports created from this
GitHub release. Existing result repos must remain unchanged because active
experiments may still depend on them.

Recommended structure for each canonical result repo:

- `README.md`: task summary, metrics, source models, source datasets.
- `reports/`: human-readable reports.
- `metrics/`: JSON/CSV metrics with schemas.
- `predictions/`: selected predictions or decode samples safe for release.
- `artifacts/`: small non-sensitive generated examples when reviewed.
- `manifests/`: source repo IDs, revisions, checksums, export date.

The exported source folders are also checked into this GitHub release under
`results/canonical_exports/`.
