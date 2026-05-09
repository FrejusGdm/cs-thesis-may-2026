# Models

Models remain in separate Hugging Face model repos and are grouped here by task.
This avoids mixing incompatible model architectures while making the thesis
artifact graph easy to inspect.

## Best Public Models

These are the public thesis-facing model artifacts for the first cleanup pass.
Model cards live in each Hugging Face repo; local copies are in
`models/hf_cards/`.

## Machine Translation

- `JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42`: French -> Adja,
  NLLB-600M, thesis-final seed-42 checkpoint.
- `JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42`: Adja -> French,
  NLLB-600M, thesis-final seed-42 checkpoint.

## ASR / STT

- `JosueG/wav2vec2-xlsr-adja-c4v2`: primary deployable Adja ASR model,
  C4v2 XLS-R CTC.
- `JosueG/whisper-ewe-adja-e4v4`: complementary Adja ASR model,
  Whisper-Ewe -> Adja transfer with hallucination caveats.

## TTS

- `JosueG/spark-tts-adja-t3`: thesis-best Spark T3 Adja TTS family.

## Hugging Face Collection

The public collection is managed as `CS Thesis May 2026` under `JosueG`.
Hugging Face collection slugs include a generated suffix. The current public
collection is `JosueG/cs-thesis-may-2026-69ff55e45e0d5b0eb7fa4344`.

## Policy

Weights are not copied or consolidated in this pass. Existing best model repos
remain the weight locations. This cleanup updates cards, public visibility for
the selected thesis-final repos, and collection membership only.
