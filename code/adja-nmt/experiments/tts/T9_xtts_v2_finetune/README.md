# T9: XTTS-v2 Fine-tune for Adja

Status: **ready for smoke submission**. Canonical launcher:
`scripts/hf_jobs/T9_xtts_v2_finetune.py`.

## Why T9 exists

XTTS-v2 is the most practical small-data multilingual TTS baseline in this set:

- cross-lingual voice cloning is the core use case
- the model is widely used with very small speaker/language datasets
- the public training recipe is much easier to operationalize than IMS-Toucan

This is not the most linguistically principled option for Adja, but it is one of
the strongest "can a practical multilingual TTS stack adapt with ~2h?" tests.

## Important limitation

The public Coqui recipe only fine-tunes the **GPT encoder** side of XTTS. That is
still useful here: it measures whether Adja can be absorbed into the upstream
conditioning/autoregressive stack without retraining the whole system.

## Canonical HF Jobs smoke

The smoke should prove:

1. Coqui TTS installs cleanly
2. XTTS files download
3. Adja data materializes into LJSpeech-style metadata
4. trainer enters the real training loop
5. a checkpoint directory is created

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T9_xtts_v2_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -lc "echo '$SCRIPT_B64' | base64 -d > /tmp/T9.py && python3 /tmp/T9.py --dry-run --push-to-hub --results-prefix T9_xtts_smoke_$(date +%Y-%m-%d)"
```

## Canonical full run

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T9_xtts_v2_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 8h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -lc "echo '$SCRIPT_B64' | base64 -d > /tmp/T9.py && python3 /tmp/T9.py --push-to-hub --results-prefix T9_xtts_full_$(date +%Y-%m-%d)"
```

## Data contract

The launcher materializes Adja into a Coqui-compatible LJSpeech-style directory:

```text
xtts_adja/
  metadata.csv
  wavs/
    clip_000001.wav
    ...
```

`metadata.csv` rows are:

```text
clip_000001|normalized text|normalized text
```

That matches Coqui's `formatter="ljspeech"` expectation.

Repo conventions preserved:

- NFC normalization
- seed 42
- 80/10/10 split
- dry-run mode with very small subsets

## Why XTTS-v2 is a serious Adja candidate

- official model card explicitly targets cross-language cloning
- the model is designed to work from short conditioning clips
- 24 kHz output is already aligned with common multilingual TTS practice

## Why it is not automatically the best fit

1. Adja is outside XTTS's official language list.
2. The recipe does not give us the same explicit language-family prior as Ewe or
   IMS-Toucan.
3. The public training route is a practical heuristic, not a targeted
   low-resource tonal recipe.

## What counts as a successful smoke

- install succeeds
- XTTS base files download
- materialized `metadata.csv` and `wavs/` exist
- `trainer.fit()` starts
- checkpoint files appear
- `metrics.json` uploads to the results repo

## Tracking

After every submission, update:

- `experiments/tts/attempt-log.md`
- `experiments/registry.md`
- `results/run-ledger.md`
- `results/tts-comparison.md`

## References

- XTTS-v2 model card: https://huggingface.co/coqui/XTTS-v2
- Coqui XTTS docs: https://tts.readthedocs.io/en/latest/models/xtts.html
- Coqui TTS recipe: `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`
