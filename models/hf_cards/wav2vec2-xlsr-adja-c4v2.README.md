---
language:
- adj
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- adja
- asr
- speech-recognition
- low-resource
- wav2vec2
- xls-r
- cs-thesis-may-2026
datasets:
- JosueG/adja-speech-asr-tts
base_model: facebook/wav2vec2-xls-r-300m
---

# Wav2Vec2 XLS-R Adja ASR C4v2

This is the thesis-final deployable CTC ASR model for Adja speech. It is the
main public ASR weight location used by the May 2026 CS thesis release.

## Thesis Role

This model is the primary Adja speech-to-text artifact. It is used as an ASR
baseline, an endpoint model in the S2TT pipeline, and one of the reverse-WER
judges for TTS evaluation.

## Model And Data

- **Task:** automatic speech recognition
- **Base model:** `facebook/wav2vec2-xls-r-300m`
- **Training data:** Orpheus Adja speech lineage, public canonical dataset
  `JosueG/adja-speech-asr-tts`
- **Input audio:** use 16 kHz audio at inference time unless your pipeline
  handles resampling explicitly
- **Release repo:** `FrejusGdm/cs-thesis-may-2026`

## Headline Result

The C4v2 acoustic model reports **25.05% CER** and **72.39% WER** with greedy
decoding in the project registry. With a character language model and Optuna
shallow-fusion tuning, the best reported ASR setting reaches **22.67% normalized
CER** and **70.76% normalized WER**.

See:

- `results/adja-nmt/D4_ctc_lm/conclusion.md`
- `docs/source-repos/adja-nmt/experiment-registry.md`

## Limitations

- The public model repo is the acoustic model surface; some LM decoding
  experiments live in the result reports rather than inside this model repo.
- Deployed endpoint behavior can differ from offline evaluation because of
  tokenizer reconstruction, chunking, and resampling.
- CER is more informative than WER for this release because Adja word-boundary
  and orthographic normalization choices are still evolving.

## Citation

If you use this model, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*. https://github.com/FrejusGdm/cs-thesis-may-2026
