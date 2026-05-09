# Preprocessing Pipeline

Scripts used to clean and prepare raw data **before** it enters the experiment splitting pipeline (`data/prepare_splits.py`).

## Purpose

The corpus generation scripts (`grammatical_dataset-generation/code-run-{1,2}/`) produce raw French sentences and the translator provides Adja translations. Before these can be used in experiments, several cleaning steps are needed. This folder documents and automates those steps for reproducibility.

## Expected Workflow

```
Raw translated data (private, not in repo)
        │
        ▼
┌─────────────────────────┐
│  preprocessing/ scripts  │  ← You are here
│  (cleaning, dedup, etc.) │
└─────────────────────────┘
        │
        ▼
Cleaned CSVs (private, not in repo)
        │
        ▼
┌─────────────────────────┐
│  data/prepare_splits.py  │  80-10-10 group-aware split
└─────────────────────────┘
        │
        ▼
Train / Val / Test splits (private, not in repo)
```

## Scripts

Add your cleaning scripts here. Recommended naming convention: number them in execution order.

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_normalize_whitespace.py` | Strip extra spaces, normalize Unicode | Raw CSVs | Normalized CSVs |
| `02_deduplicate.py` | Remove exact and near-duplicate pairs | Normalized CSVs | Deduped CSVs |
| `03_validate_alignment.py` | Check Fr-Adja pairs are aligned, flag empty cells | Deduped CSVs | Validated CSVs |
| ... | ... | ... | ... |

*(Update this table as you add scripts.)*

## Running

```bash
# Run all preprocessing steps in order:
python 01_normalize_whitespace.py --input ../path/to/raw.csv --output data/normalized.csv
python 02_deduplicate.py --input data/normalized.csv --output data/deduped.csv
python 03_validate_alignment.py --input data/deduped.csv --output data/validated.csv

# Then proceed to splitting:
python ../data/prepare_splits.py --input data/validated.csv
```

## What to Document

For methodology transparency, each script should log:

- **Input count**: number of sentence pairs before processing
- **Output count**: number of sentence pairs after processing
- **What was removed/changed and why** (e.g., "removed 12 duplicate pairs", "normalized 34 Unicode characters")
- **Any manual decisions** (e.g., "kept shorter variant when duplicates differed only in trailing punctuation")

This information feeds directly into the paper's data preparation section.

## Data Directory

Intermediate outputs go in `preprocessing/data/` (gitignored). Only scripts are committed.
