---
language:
- adj
license: apache-2.0
library_name: transformers
pipeline_tag: text-to-speech
tags:
- adja
- tts
- speech-synthesis
- spark-tts
- low-resource
- cs-thesis-may-2026
datasets:
- JosueG/adja-speech-asr-tts
base_model: unsloth/Spark-TTS-0.5B
---

# Spark TTS Adja T3

This is the thesis-final Spark TTS Adja model repo for the May 2026 CS thesis
release. It is the public weight location for the best working Adja TTS family
found during the project.

## Thesis Role

This model is the primary text-to-speech artifact. It represents the T3 Spark
path: the only TTS family in the project that produced clearly intelligible
Adja in native-listening review.

## Model And Data

- **Task:** text-to-speech
- **Base model:** `unsloth/Spark-TTS-0.5B` lineage
- **Training data:** Orpheus Adja speech lineage, public canonical dataset
  `JosueG/adja-speech-asr-tts`
- **Release repo:** `FrejusGdm/cs-thesis-may-2026`

## Headline Result

The Spark T3 family is the thesis-best TTS direction. The canonical T3 run is
reported with **36.14% C4v2 reverse-CER** and **28.92% E4v4 reverse-CER**; the
longer T3 early-stop run reaches **33.73% C4v2 reverse-CER**. Native listening
review marked Spark T3 as intelligible Adja, while most other TTS families
collapsed into noise or non-Adja output.

See:

- `results/adja-nmt/tts-comparison.md`
- `results/adja-nmt/pipeline-comparison.md`

## Limitations

- Reverse-WER and reverse-CER inherit ASR model error. Treat them as relative
  diagnostics, not absolute speech-quality scores.
- The dataset is small and speaker coverage is limited.
- Generated speech should be reviewed by fluent or native Adja speakers before
  use outside research.

## Citation

If you use this model, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*. https://github.com/FrejusGdm/cs-thesis-may-2026
