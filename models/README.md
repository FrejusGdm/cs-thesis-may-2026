# Models

Models remain in separate Hugging Face model repos and are grouped here by task.
This avoids mixing incompatible model architectures while making the thesis
artifact graph easy to inspect.

## Machine Translation

- `JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42`
- `JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42`

## ASR / STT

- `JosueG/wav2vec2-xlsr-adja-c4v2`
- `JosueG/whisper-ewe-adja-e4v4`

## TTS

- `JosueG/spark-tts-adja-t3`

## Policy

Existing model repos are not moved or mutated by the release workflow. A
Hugging Face collection should group selected thesis-final models.
