---
language:
- adj
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- adja
- ewe
- asr
- whisper
- low-resource
- cs-thesis-may-2026
datasets:
- JosueG/adja-speech-asr-tts
base_model: openai/whisper-small
---

# Whisper-Ewe -> Adja ASR E4v4

This is the public Whisper-Ewe-to-Adja transfer ASR model for the May 2026 CS
thesis release. It is the deployable reproduction of the Ewe-transfer ASR path.

## Thesis Role

This model is the complementary Adja ASR artifact. It tests whether transfer
from Ewe, a related Gbe language, helps Adja speech recognition, and it is used
as a second ASR judge for TTS reverse-WER evaluation.

## Model And Data

- **Task:** automatic speech recognition
- **Base model:** Whisper-small lineage, previously adapted to Ewe before Adja
  fine-tuning
- **Training data:** Orpheus Adja speech lineage, public canonical dataset
  `JosueG/adja-speech-asr-tts`
- **Input audio:** use 16 kHz audio at inference time unless your pipeline
  handles resampling explicitly
- **Release repo:** `FrejusGdm/cs-thesis-may-2026`

## Headline Result

The deployable E4v4 reproduction reaches **37.18% dev CER** and **83.61% dev
WER** at epoch 20. It is worse than the original logged E4 number, whose
checkpoint was not preserved as a standalone deployable model, but it remains a
useful complementary ASR judge.

See:

- `results/adja-nmt/E4v4_whisper_ewe_fixed/conclusion.md`
- `docs/source-repos/adja-nmt/experiment-registry.md`

## Limitations

- The original E4 log reported 24.90% CER, but that checkpoint is not available
  as a public deployable model. This repo should be cited as E4v4, not the lost
  original E4 run.
- E4v4 can hallucinate on short or out-of-distribution clips.
- Use C4v2 alongside E4v4 when judging TTS outputs; agreement between both ASRs
  is more meaningful than either model alone.

## Citation

If you use this model, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*. https://github.com/FrejusGdm/cs-thesis-may-2026
