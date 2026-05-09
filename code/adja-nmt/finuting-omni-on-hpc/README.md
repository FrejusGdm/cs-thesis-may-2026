# finuting-omni-on-hpc

HPC-ready OmniASR fine-tuning package for Dartmouth Discovery.

This folder is designed to hand off to another researcher who runs on SLURM/HPC with datasets stored on cluster paths (not Hugging Face datasets). It uses absolute paths throughout.

## What this package does

- Builds OmniASR manifest files (`.tsv`, `.wrd`, `.lang`) from local audio + transcript metadata.
- Runs official OmniASR fairseq2 recipe fine-tuning (CTC or LLM) from the upstream repo.
- Supports `--dry-run` to validate configs/paths without launching training.
- Writes config + run summary JSON under a run directory for reproducibility.

## Dartmouth Discovery assumptions

- Use absolute paths only (no `$(dirname "$0")` in SLURM).
- Preferred storage root:
  - `<HPC_WORKDIR>`
- GPU partition and type from project notes:
  - partition: `gpuq`
  - GPU: `nvidia_a100_80gb_pcie_3g.40gb:1`

These choices follow `learnings-from-the-past/hpc-gotchas.md`.

## Folder layout

- `prepare_manifest.py`: convert your dataset metadata into Omni manifests.
- `run_omni_finetune_hpc.py`: run training from prepared manifests.
- `slurm/train_omni_hpc.sbatch`: SLURM wrapper script (absolute paths).
- `examples/config_ctc_3b.yaml`: example config for CTC 3B.
- `examples/config_llm_3b.yaml`: example config for LLM 3B.

## 1) Prepare your dataset metadata

Create a TSV metadata file with one row per utterance and these fields:

- `split`: `train`, `dev`, or `test`
- `audio_path`: absolute path to waveform on HPC
- `text`: transcription
- `lang`: optional language code (defaults to `ajg_Latn`)

Example TSV row:

train\t<HPC_WORKDIR> transcript\tajg_Latn

## 2) Build manifests

Example command:

```
python finuting-omni-on-hpc/prepare_manifest.py \
  --input-tsv /ABS/PATH/metadata.tsv \
  --output-manifest-dir /ABS/PATH/omni_manifest \
  --default-lang ajg_Latn \
  --audio-root <HPC_WORKDIR> \
  --min-audio-samples 1600
```

Outputs:

- `/ABS/PATH/omni_manifest/train.tsv`, `train.wrd`, `train.lang`
- `/ABS/PATH/omni_manifest/dev.tsv`, `dev.wrd`, `dev.lang`
- `/ABS/PATH/omni_manifest/test.tsv`, `test.wrd`, `test.lang`
- `/ABS/PATH/omni_manifest/manifest_stats.json`

## 3) Dry-run training config validation

Dry-run checks paths/config and writes generated recipe YAML, but does not train.

```
python finuting-omni-on-hpc/run_omni_finetune_hpc.py \
  --omni-repo-dir /ABS/PATH/omnilingual-asr \
  --manifest-dir /ABS/PATH/omni_manifest \
  --output-dir /ABS/PATH/omni_runs/llm_r1 \
  --ft-mode llm \
  --model-name omniASR_LLM_3B_v2 \
  --language-code ajg_Latn \
  --dry-run
```

## 4) Run on SLURM

Edit `finuting-omni-on-hpc/slurm/train_omni_hpc.sbatch`:

- Set absolute paths for:
  - `PROJECT_ROOT`
  - `OMNI_REPO_DIR`
  - `DATA_TSV`
  - `WORK_DIR`
- Optionally set `CONDA_ENV` or module loads to match your cluster env.

Submit:

```
sbatch /ABS/PATH/finuting-omni-on-hpc/slurm/train_omni_hpc.sbatch
```

## Runner CLI (no hidden config loader)

Required:

- `--omni-repo-dir` absolute path to cloned `omnilingual-asr`
- `--manifest-dir` absolute path with train/dev/test manifests
- `--output-dir` absolute run directory
- `--ft-mode` (`ctc` or `llm`)
- `--model-name`

Useful knobs:

- `--language-code` (use `ajg_Latn` for Aja Benin)
- `--num-steps`
- `--batch-size`
- `--grad-accum`
- `--max-num-elements`
- `--max-audio-sec`
- `--min-audio-len`
- `--valid-splits`
- `--validate-after-n-steps`
- `--validate-every-n-steps`
- `--checkpoint-every-n-steps`
- `--publish-metrics-every-n-steps`
- `--data-parallelism` (`ddp` or `fsdp`)
- `--save-model-only` (`false`, `true`, `all_but_last`)

## Outputs

Each run writes:

- `<output_dir>/<ft_mode>_finetune_hpc.yaml` (generated recipe config)
- fairseq2 outputs/checkpoints under `<output_dir>/`

## Notes and pitfalls

- Keep `MIN_AUDIO_LEN` consistent with your validation expectations; this changes eval sample counts.
- If CTC validation fails with zero-length sequence errors under very low `MIN_AUDIO_LEN`, use a stricter threshold (e.g., `32000`) and document resulting kept counts.
- Keep dataset paths absolute to avoid SLURM path resolution surprises.

