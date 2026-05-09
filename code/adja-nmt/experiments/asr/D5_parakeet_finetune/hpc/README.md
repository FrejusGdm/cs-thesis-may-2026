# D5 Parakeet — HPC (Dartmouth Discovery) execution path

These files are the original collaborator-provided code, preserved as-is.
They run on Dartmouth Discovery (SLURM + Apptainer) using a NeMo container.

## File map

| File                       | Purpose                                                         |
|----------------------------|-----------------------------------------------------------------|
| `parakeet_finetune.def`    | Apptainer/Singularity definition (base: `nvcr.io/nvidia/nemo:25.02`) |
| `make-sif.script`          | SLURM job that builds the `.sif` image                          |
| `finetune_parakeet.py`     | Train + evaluate driver (CSV manifests in, NeMo training out)   |
| `evaluate_parakeet.py`     | Stand-alone eval from a `.ckpt` checkpoint                      |
| `run_finetune_mqr02.slurm` | SLURM launcher for finetune + evaluation                        |
| `run_eval.slurm`           | SLURM launcher for stand-alone evaluation                       |

## Prerequisites

- Access to a partition with A100 80GB (Dartmouth: `gpuq`).
- A scratch directory for the `.sif` build cache and the trained model.
- A directory of `.wav` files plus `train`/`dev`/`test` CSVs with columns
  `audio_filepath`, `text`, optional `duration`.

## One-time: build the container

```bash
sbatch experiments/asr/D5_parakeet_finetune/hpc/make-sif.script
```

This runs `singularity build --fakeroot ... parakeet_finetune.sif
parakeet_finetune.def` and produces `parakeet_finetune.sif`. Move/copy that
file to a stable location (the SLURM scripts default to
`<HPC_WORKDIR>`).

## Run finetune + eval

Edit the `USER CONFIGURATION` block in
[`run_finetune_mqr02.slurm`](run_finetune_mqr02.slurm) to point at your
own paths:

- `SIF_IMAGE`     — path to the built `.sif`
- `SCRIPT_PATH`   — path to `finetune_parakeet.py` on the cluster
- `TRAIN_CSV` / `DEV_CSV` / `TEST_CSV` — your manifests
- `AUDIO_DIR`     — directory containing the `.wav` files
- `OUTPUT_DIR`    — where checkpoints and eval outputs go

then:

```bash
sbatch experiments/asr/D5_parakeet_finetune/hpc/run_finetune_mqr02.slurm
```

The job binds your data, output, script, and three cache directories
(`hf_cache`, `torch_cache`, `numba_cache` under `$SLURM_TMPDIR`) into the
container so nothing leaks into `$HOME`. After training it prints the
contents of `evaluation_summary.txt` to the log.

## Stand-alone evaluation

If you already have a `.ckpt` from a previous run and just want fresh test
metrics, edit [`run_eval.slurm`](run_eval.slurm) (point `CHECKPOINT` at the
best `.ckpt` and `TEST_MANIFEST` at the previously generated test
manifest), then:

```bash
sbatch experiments/asr/D5_parakeet_finetune/hpc/run_eval.slurm
```

## Notes & gotchas

- The training script intentionally creates the PyTorch-Lightning `Trainer`
  **before** `ASRModel.from_pretrained(..., trainer=trainer)`. NeMo
  requires the trainer reference at model construction time; reversing the
  order silently disables checkpointing.
- `exp_manager` handles checkpointing — the script disables PL's own
  `enable_checkpointing` and `logger`. Best checkpoint is selected on
  `val_wer`.
- The CSV → JSON manifest converter computes `duration` from the audio if
  it isn't already in the CSV. If audio paths in the CSV aren't visible
  inside the container, expand the `--bind` list.
- `torch.load(..., weights_only=False)` is required for NeMo `.ckpt`
  files because they store config dicts alongside tensors.
- See the project root `learnings-from-the-past/hpc-gotchas.md` for general
  Discovery SLURM gotchas (notably `$(dirname "$0")` not working under
  `sbatch`).
