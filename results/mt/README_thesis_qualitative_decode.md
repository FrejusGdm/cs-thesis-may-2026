# Thesis qualitative decode export (paper-era pooled data only)

Checkpoint outputs under `experiments/results/april2026/` (e.g. `FULL/` with train sizes near **19–20K**) correspond to **expanded corpora** after additional Couffo data—not the ACL / thesis **10K Tatoeba-seeded + 4K structured** regime.

For tables and qualitative decodes aligned with `\ref{tab:mt-robustness}` and `aggregate_subset_metrics.py` (**“paper-era data only (no new6k)”**), use **`experiments/results/rebuttal_rerun/nllb-600m/exp1/`**.

## Script

[`../tools/export_thesis_decode_table.py`](../tools/export_thesis_decode_table.py)

From repo root `neurosymbolic-ai-paper-experiments`:

```bash
python experiments/tools/export_thesis_decode_table.py \
  --caption-style thesis \
  --latex-out /path/to/.../mt_qualitative_decodes_rebuttal.tex \
  --tatoeba-latex-out /path/to/.../mt_qualitative_decodes_tatoeba.tex
```

`--latex-out` is the fused held-out qualitative table (`idx` 0--1454). `--tatoeba-latex-out` writes the **same columns** filtered to Tatoeba-sourced held-out lines (`idx` $\geq$ 455); selection still ranks mixed-over-random sentence chrF++. Omit either flag only if you do not want that artefact written.

Use `--base .../rebuttal_repaired_.../nllb-600m/exp1` when consuming repaired JSONLs. For appendix or internal repro notes, keep `--caption-style full` (optionally `--caption-provenance '...'`) so paths and regime text appear in the caption.

Defaults:

- `--seed 42`
- Loads `test_predictions.jsonl` under:
  - `RANDOM-10K/seed{N}/`
  - `STRUCTURED-4K-ONLY/seed{N}/`
  - `RANDOM-10K_STRUCTURED-4K/seed{N}/`

Merged on `idx`; **assert** identical French `src` and reference Adja `ref`.

## Thesis destination

Overwrite or regenerate:

- `adja-nmt/.../mt_qualitative_decodes_rebuttal.tex` (full fused benchmark slice)
- `adja-nmt/.../mt_qualitative_decodes_tatoeba.tex` (Tatoeba held-out slice)

and `\input{...}` both from `06_results.tex` (combined first, Tatoeba second).

## Sync log template (paste under this section when regenerating)

_Last sync_: 2026-05-02

- **Date**: 2026-05-02
- **JSONL triple** (Rand / Struct-only / Mixed):
  `rebuttal_rerun/nllb-600m/exp1/RANDOM-10K/seed42/test_predictions.jsonl`,
  `.../STRUCTURED-4K-ONLY/seed42/test_predictions.jsonl`,
  `.../RANDOM-10K_STRUCTURED-4K/seed42/test_predictions.jsonl`
- **Seed**: 42
- **CLI**:
  `python experiments/tools/export_thesis_decode_table.py --latex-out ~/.../cs-thesis-josue-2026/chapters/tex/03_mt_sections/mt_qualitative_decodes_rebuttal.tex`
- **Thesis**: `06_results.tex` + fragment `mt_qualitative_decodes_rebuttal.tex`
