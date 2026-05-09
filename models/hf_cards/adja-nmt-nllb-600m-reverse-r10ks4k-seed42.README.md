---
language:
- adj
- fr
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
tags:
- adja
- nllb
- low-resource
- machine-translation
- cs-thesis-may-2026
datasets:
- JosueG/french-adja-parallel-corpus
base_model: facebook/nllb-200-distilled-600M
---

# NLLB-600M Adja -> French

This is the thesis-final Adja-to-French machine translation checkpoint for the
May 2026 CS thesis release. It is the reverse-direction companion to the
French-to-Adja NLLB model.

## Thesis Role

This model is the reverse MT artifact: Adja source text to French target text.
It is used for Adja text interpretation, reverse translation checks, and the
speech-pipeline experiments where ASR output is translated into French.

## Model And Data

- **Task:** machine translation, Adja -> French
- **Base model:** `facebook/nllb-200-distilled-600M`
- **Direction:** `aj_Latn` -> `fra_Latn`
- **Training data:** French-Adja thesis corpus lineage, with the public
  canonical MT reference at `JosueG/french-adja-parallel-corpus`
- **Release repo:** `FrejusGdm/cs-thesis-may-2026`

## Headline Result

This seed-42 NLLB-600M checkpoint is the public reverse-direction MT weight
location for the thesis. It is the intended model to pair with Adja ASR when
building Adja speech -> French text pipelines.

See:

- `results/mt/summary/all_results.csv`
- `results/adja-nmt/pipeline-comparison.md`

## Limitations

- The model should be treated as a research artifact, not a production
  translator.
- ASR noise compounds MT errors in speech pipelines.
- The model uses the release tokenizer/config; load this repo directly so the
  Adja language token and generation settings match the checkpoint.

## Citation

If you use this model, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*. https://github.com/FrejusGdm/cs-thesis-may-2026
