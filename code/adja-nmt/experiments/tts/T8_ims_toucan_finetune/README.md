# T8: IMS-Toucan Fine-tune for Adja

Status: **ready for smoke submission**. Canonical launcher:
`scripts/hf_jobs/T8_ims_toucan_finetune.py`.

## Why T8 matters

IMS-Toucan is the most principled match in the open-source stack for the Adja
constraint:

- low-resource target language
- tonal phonology
- multilingual transfer as the main learning mechanism
- explicit language embeddings instead of only a generic text tokenizer

For this repo, T8 is the clearest test of whether **multilingual phonological
generalization** beats English-centric codec-LM adaptation.

## Upstream fit to Adja

- The upstream frontend works with ISO-639-3 language identifiers.
- `ajg` (Aja / Adja, Benin) exists in the upstream `iso_lookup.json`.
- Unsupported frontend details can fall back to **Transphone**, which makes this
  a realistic zero-shot-to-fine-tune path rather than a tokenizer dead-end.

This means the first run should use **direct Adja (`ajg`)**, not Ewe as a fake
frontend language.

## Canonical HF Jobs smoke

The smoke run is useful only if it proves:

1. repo install succeeds
2. Adja audio is materialized and cached
3. `prepare_tts_corpus(..., lang="ajg")` succeeds
4. Toucan training enters the real loop
5. at least one checkpoint is written

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T8_ims_toucan_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 3h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    -- \
    bash -lc "echo '$SCRIPT_B64' | base64 -d > /tmp/T8.py && python3 /tmp/T8.py --dry-run --push-to-hub --results-prefix T8_ims_toucan_smoke_$(date +%Y-%m-%d)"
```

## Canonical fine-tune run

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T8_ims_toucan_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 8h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    -- \
    bash -lc "echo '$SCRIPT_B64' | base64 -d > /tmp/T8.py && python3 /tmp/T8.py --push-to-hub --results-prefix T8_ims_toucan_ajg_$(date +%Y-%m-%d)"
```

## How the launcher is shaped

The launcher does **not** patch the upstream repo permanently. Instead it:

1. clones IMS-Toucan into `/tmp`
2. installs the training dependencies needed for headless HF Jobs
3. exports the Adja dataset as wav files + a path->transcript dictionary
4. calls the same internal building blocks used by
   `Recipes/finetuning_example_simple.py`
5. fine-tunes from the public `Flux9665/ToucanTTS` checkpoint

This keeps the repo-native launcher self-contained while still mirroring the
upstream recipe.

## Default training choices

- language code: `ajg`
- seed: `42`
- dataset split: `80/10/10`
- fine-tune from: `Flux9665/ToucanTTS`
- batch size: conservative by default to avoid A100 OOM surprises

## What counts as success

- cached corpus directory exists
- save directory contains a checkpoint
- training loop prints update/step progress
- `metrics.json` uploads to the results repo

For the first smoke, this is enough. Sample generation is a second-stage check.

## Known risks

1. **Dependency footprint is large** compared to XTTS-v2.
2. **The first run may spend significant time in preprocessing.**
3. **The frontend may succeed technically but still underperform if Transphone's
   zero-shot phonology for Adja is weak.**

## Tracking

After each cycle update:

- `experiments/tts/attempt-log.md`
- `experiments/registry.md`
- `results/run-ledger.md`
- `results/tts-comparison.md`

## References

- IMS-Toucan repo: https://github.com/DigitalPhonetics/IMS-Toucan
- Multilingual Toucan demo: https://multilingualtoucan.github.io/
- Upstream training entry: `run_training_pipeline.py`
