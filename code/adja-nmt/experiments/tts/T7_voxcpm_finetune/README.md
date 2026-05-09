# T7: VoxCPM Fine-tune for Adja

Status: **ready for smoke submission**. Canonical launcher:
`scripts/hf_jobs/T7_voxcpm_finetune.py`.

## Why T7 exists

VoxCPM gives us a very different TTS family from the codec-LM baselines already in
this repo:

- **VoxCPM2** is multilingual and tokenizer-free, with native 48 kHz output.
- The upstream repo exposes both **LoRA** and **full fine-tuning**.
- It is one of the most relevant "large multilingual capacity" tests for Adja.

This is not the most principled low-resource track for Adja. That remains
IMS-Toucan and MMS-TTS-Ewe. T7 exists to test whether a much larger multilingual
speech prior can overcome the same low-resource bottleneck that limited CSM,
Orpheus, and Spark.

## Default model choice

Default: `openbmb/VoxCPM2`

Why:

- it is the upstream multilingual release
- it is the only VoxCPM line that makes sense as a serious Adja experiment
- it supports LoRA and full SFT in the official docs

Fallbacks exposed by the script:

- `openbmb/VoxCPM1.5`
- `openbmb/VoxCPM-0.5B`

Use those only for faster diagnostics. They are not the main multilingual answer.

## Canonical HF Jobs smoke

This smoke is the minimum useful validation:

1. install VoxCPM cleanly inside the container
2. download the base model
3. materialize Adja audio into VoxCPM JSONL manifests
4. run at least **2 real training steps**
5. save a LoRA checkpoint plus `train.log`

```bash
SCRIPT_B64=$(base64 -w0 < scripts/hf_jobs/T7_voxcpm_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T7.py && python3 /tmp/T7.py --dry-run --push-to-hub --results-prefix T7_voxcpm2_smoke_$(date +%Y-%m-%d)"
```

## Canonical LoRA run

```bash
SCRIPT_B64=$(base64 -w0 < scripts/hf_jobs/T7_voxcpm_finetune.py)
hf jobs run \
    --flavor a100-large \
    --timeout 8h \
    --secrets HF_TOKEN \
    -d \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T7.py && python3 /tmp/T7.py --push-to-hub --results-prefix T7_voxcpm2_lora_$(date +%Y-%m-%d)"
```

## Full fine-tune follow-up

```bash
... python3 /tmp/T7.py --full-finetune --learning-rate 1e-5 --num-iters 1000 \
    --push-to-hub --results-prefix T7_voxcpm2_fullft_$(date +%Y-%m-%d)
```

Keep full FT as a second-stage run. LoRA is the cheaper first question.

## Data contract

The script converts the private HF dataset `JosueG/adja-tts-orpheus` into the
upstream VoxCPM JSONL manifest format:

```json
{"audio": "/abs/path/to/example.wav", "text": "NFC-normalized transcript", "dataset_id": 0}
```

It keeps the repo-standard:

- seed `42`
- 80/10/10 split
- Unicode NFC normalization

The base model's `config.json` is read to auto-detect the training
`sample_rate`, so the script does not hardcode a possibly wrong rate.

## What counts as a successful smoke

- environment install succeeds
- model snapshot download succeeds
- materialized train/val manifests exist
- VoxCPM prints real training progress
- a `checkpoints/latest/` folder exists
- `results_repo/<prefix>/metrics.json` uploads cleanly

## Known risks

1. **Install footprint is bigger than XTTS/F5.**
2. **VoxCPM2 is newer than XTTS-v2 and less battle-tested for tiny-language
   adaptation.**
3. **A quick smoke may spend more wall time downloading than training.**
4. **This is a large-model ablation, not the most linguistically informed Adja
   prior.**

## Tracking

After every submission, update:

- `experiments/tts/attempt-log.md`
- `experiments/registry.md`
- `results/run-ledger.md`
- `results/tts-comparison.md`

## References

- VoxCPM repo: https://github.com/OpenBMB/VoxCPM
- VoxCPM fine-tune guide: upstream `docs/finetune.md`
- VoxCPM2 docs: https://voxcpm.readthedocs.io/en/latest/
