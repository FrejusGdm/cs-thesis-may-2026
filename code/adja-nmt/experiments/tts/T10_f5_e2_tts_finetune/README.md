# T10/T11: F5-TTS and E2-TTS for Adja

Status: **ready with one documented caveat**. Canonical launcher:
`scripts/hf_jobs/T10_f5_e2_tts_finetune.py`.

## Why this track exists

F5-TTS and E2-TTS let us test a genuinely different TTS objective:

- **flow matching** instead of next-token codec LM training
- a clean upstream fine-tune CLI
- a direct comparison between two related model families inside the same repo

This makes them useful research ablations even if they are not the safest first
success path for Adja.

## T10 vs T11

- **T10** = `F5TTS_Base`
- **T11** = `E2TTS_Base`

The same launcher handles both. `F5TTS_Base` is the default because the upstream
repo positions F5 as the stronger practical baseline.

## Important vocabulary caveat

The released checkpoints are English/Chinese-oriented. Adja introduces unseen
characters (`ɛ`, `ɔ`, `ŋ`, `ɖ`, tone-marked forms), so a naive fine-tune can
fail with vocabulary-size mismatch or silently map everything to unknowns.

The launcher handles this explicitly by:

1. materializing Adja data in the upstream `raw.arrow + duration.json + vocab.txt`
   format
2. comparing Adja's custom `vocab.txt` against the base checkpoint
3. expanding the pretrained text embedding matrix when new symbols are needed

Without that step, this experiment is not meaningful.

## Canonical HF Jobs smoke

The smoke proves:

1. F5-TTS installs in the HF container
2. Adja wav/text data is materialized into the upstream dataset bundle
3. custom vocab expansion succeeds when needed
4. the trainer runs at least **2 real updates**

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T10_f5_e2_tts_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -lc "echo '$SCRIPT_B64' | base64 -d > /tmp/T10.py && python3 /tmp/T10.py --dry-run --push-to-hub --results-prefix T10_f5_smoke_$(date +%Y-%m-%d)"
```

## E2 smoke

```bash
... python3 /tmp/T10.py --model-family E2TTS_Base --dry-run --push-to-hub \
    --results-prefix T11_e2_smoke_$(date +%Y-%m-%d)
```

## Longer run

```bash
... python3 /tmp/T10.py --push-to-hub --results-prefix T10_f5_$(date +%Y-%m-%d)
```

Keep E2 as a second run unless the user specifically wants both immediately.

## Data contract

The launcher converts `JosueG/adja-tts-orpheus` into the upstream F5 dataset
bundle:

- `raw.arrow`
- `duration.json`
- `vocab.txt`

It uses:

- NFC normalization
- repo-standard seed 42 split
- absolute wav paths inside the Arrow dataset

## What counts as a successful smoke

- install completes
- Adja dataset bundle is created
- custom vocab expansion reports either:
  - no missing symbols, or
  - a new expanded pretrained checkpoint
- trainer logs real updates
- checkpoint directory contains `model_last.pt` or a saved update checkpoint

## Known risks

1. **Tokenizer/vocab compatibility is the main failure mode.**
2. **These checkpoints are not Gbe-family priors.**
3. **A successful smoke only proves the training path is live, not that the
   resulting speech will be intelligible.**

## Tracking

After every submission, update:

- `experiments/tts/attempt-log.md`
- `experiments/registry.md`
- `results/run-ledger.md`
- `results/tts-comparison.md`

## References

- F5-TTS repo: https://github.com/SWivid/F5-TTS
- F5-TTS paper: https://aclanthology.org/2025.acl-long.313/
- E2-TTS paper: https://arxiv.org/abs/2406.18009
