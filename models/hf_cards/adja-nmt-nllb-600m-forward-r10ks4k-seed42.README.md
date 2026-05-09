---
language:
- fr
- adj
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

# NLLB-600M French -> Adja

This is the thesis-final French-to-Adja machine translation checkpoint for the
May 2026 CS thesis release. It is one of the two NLLB models used to make the
French-Adja MT work inspectable and reproducible.

## Thesis Role

This model is the forward MT artifact: French source text to Adja target text.
It supports the thesis experiments on grammar-guided dataset construction,
low-resource NMT fine-tuning, and the speech-to-text-to-speech pipeline.

## Model And Data

- **Task:** machine translation, French -> Adja
- **Base model:** `facebook/nllb-200-distilled-600M`
- **Direction:** `fra_Latn` -> `aj_Latn`
- **Training data:** French-Adja thesis corpus lineage, with the public
  canonical MT reference at `JosueG/french-adja-parallel-corpus`
- **Release repo:** `FrejusGdm/cs-thesis-may-2026`

## Headline Result

The seed-42 NLLB-600M full/module runs are the strongest MT baseline in the
release. The public GitHub result tables report about 21-22 BLEU and 29 chrF
on the held-out MT evaluation for the full-corpus NLLB setting.

See:

- `results/mt/summary/all_results.csv`
- `results/mt/README_thesis_qualitative_decode.md`

## Limitations

- Adja remains low-resource; outputs should be reviewed before community,
  educational, or publication use.
- Metrics are sensitive to corpus split, normalization, and reference coverage.
- The model uses the release tokenizer/config; load this repo directly so the
  Adja language token and generation settings match the checkpoint.

## Citation

If you use this model, cite:

Josue Godeme. 2026. *CS Thesis May 2026: French-Adja MT and Adja Speech
Experiments*. https://github.com/FrejusGdm/cs-thesis-may-2026
